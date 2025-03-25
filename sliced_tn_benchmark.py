from mpi4py import MPI
import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from cuquantum import contract, CircuitToEinsum, Network
from run import run
import pandas as pd
from itertools import product

frontend = "qiskit"
backend = "cutn"
nwarmups_contraction = 1
nrepeats_contraction = 5
nrepeats_pathfinding = 1
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa"] #["qft", "qpe", "qaoa", "hidden_shift", "vqe", "bv", "hamiltonian_sim", "random"]
circuit_sizes = [64] #[2, 4, 8, 12, 16, 20, 24, 28, 32]
slices_per_gpu = [1] #[2, 4, 8, 16, 32]
reconfig_iterations = [32] #[0, 2, 4, 8, 16, 32, 64]
num_gpus = getDeviceCount()

# Set up the MPI communicator.
root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Note that all NCCL operations must be performed in the correct device context.
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()

# Set up the NCCL communicator.
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

# Remove memory manager to avoid memory leaks
cp.cuda.set_allocator(None)
# Sum the partial contribution from each process on root.
stream_ptr = cp.cuda.get_current_stream().ptr

for circuit_name, num_qubits in product(circuits, circuit_sizes):
    data_list = []
    if circuit_name == "random" and num_qubits >= 28:
        continue
    # compute the bitstring amplitude
    bitstring = ["0" if bit % 2 == 0 else "1" for bit in range(num_qubits)]

    expression, operands = None, None
    if rank == root:
        # Define quantum circuit and convert it to an Eninstein summation
        args = ["circuit", "--frontend", frontend, "--backend", backend, "--benchmark", 
                circuit_name, "--nqubits", f"{num_qubits}", "--nwarmups", 
                f"{nwarmups_contraction}", "--nrepeats", f"{nrepeats_contraction}", "--new"]
        runner = run(args, get_runner=True)
        runner.nqubits = num_qubits
        b = runner._benchmarks[circuit_name]
        benchmark_object = b['benchmark']
        benchmark_config = config
        benchmark_config['precision'] = "single"  # some frontends may need it
        runner.benchmark_name = circuit_name
        runner.benchmark_object = benchmark_object
        runner.benchmark_config = benchmark_config
        circuit = runner._load_or_generate_circuit(f"circuits/{circuit_name}_{num_qubits}")
        
        exact_amplitude = None
        converter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)
        expression, operands = converter.amplitude(bitstring)
        exact_amplitude = contract(expression, *operands, stream=stream_ptr)
        exact_amplitude = cp.asnumpy(exact_amplitude).item()

    comm.Barrier()
    expression = comm.bcast(expression, root)
    operands = comm.bcast(operands, root)
    
    for slice_per_gpu, num_iterations, seed in product(slices_per_gpu, reconfig_iterations, range(nrepeats_pathfinding)):
        samples = num_iterations
        # print(f"pathfinder samples: {num_iterations}, batch: {seed+1}/{nrepeats_pathfinding}")
    
        # Set the operand data on root.
        if rank == root:
            operands = operands
        else:
            operands = [cp.empty(o.shape, dtype='complex128') for o in operands]

        # Broadcast the operand data. We pass in the CuPy ndarray data pointers to the NCCL APIs.
        for operand in operands:
            comm_nccl.broadcast(operand.data.ptr, operand.data.ptr, operand.size*operand.dtype.itemsize, nccl.NCCL_FLOAT64, root, stream_ptr)

        # # Create network object.
        with Network(expression, *operands, stream=stream_ptr) as network:
            start_pathfinding = cp.cuda.Event()
            end_pathfinding = cp.cuda.Event()
            
            comm.Barrier()
            start_pathfinding.record()

            # Compute the path on all ranks with n samples for hyperoptimization. Force slicing to enable parallel contraction.
            path, info = network.contract_path(optimize={'samples': samples, 
                                                         'threads': 8,
                                                        'slicing': {'min_slices': slice_per_gpu * size}, 
                                                        # 'reconfiguration':{'num_iterations':num_iterations}, 
                                                        # 'path':{'num_iterations':num_iterations},
                                                        # 'seed':seed,
                                                        # 'smart':False,
                                                        })

            # Select the best path from all ranks. Note that we still use the MPI communicator here for simplicity.
            opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

            # Broadcast info from the sender to all other ranks.
            info = comm.bcast(info, sender)

            # Set path and slices.
            path, info = network.contract_path(optimize={'path': info.path, 
                                                         'slicing': info.slices, 
                                                        #  'reconfiguration':{'num_iterations':num_iterations}, 
                                                        #  'path':{'num_iterations':num_iterations},
                                                            # 'seed':seed,
                                                        })

            comm.Barrier()
            end_pathfinding.record()
            end_pathfinding.synchronize()

            # Calculate this process's share of the slices.
            num_slices = info.num_slices
            chunk, extra = num_slices // size, num_slices % size
            slice_begin = rank * chunk + min(rank, extra)
            slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
            slices = range(slice_begin, slice_end)

            for i in range(nwarmups_contraction + nrepeats_contraction):
                start_contract = cp.cuda.Event()
                end_contract = cp.cuda.Event()

                comm.Barrier()
                start_contract.record()

                # Contract the group of slices the process is responsible for.
                result = network.contract(slices=slices, stream=stream_ptr)

                comm_nccl.reduce(result.data.ptr, result.data.ptr, result.size * result.dtype.itemsize, nccl.NCCL_FLOAT64, nccl.NCCL_SUM, root, stream_ptr)

                comm.Barrier()
                end_contract.record()
                end_contract.synchronize()

                # Check correctness.
                if rank == root and i >= nwarmups_contraction:
                    parallel_amplitude = result.item()
                    pathfinding_elapsed_gpu_time = float(cp.cuda.get_elapsed_time(start_pathfinding, end_pathfinding)) / 1000
                    contract_elapsed_gpu_time = float(cp.cuda.get_elapsed_time(start_contract, end_contract)) / 1000
                    print(f"Circuit {circuit_name}({num_qubits} qubits) on {slice_per_gpu} slices per GPU and {size} gpus ({num_gpus} gpus on {int(size / num_gpus)} nodes), total time required: {contract_elapsed_gpu_time} s\nResult exact: {exact_amplitude}\nResult paral: {parallel_amplitude}\nDifference: {exact_amplitude-parallel_amplitude}\n")
                    data_list.append({"circuit":circuit_name, "num_qubits":num_qubits, 
                                        "slice_per_gpu":slice_per_gpu, "n_gpus":size, "samples":samples,
                                        "contract_time_seconds":contract_elapsed_gpu_time, 
                                        "pathfinding_time_seconds":pathfinding_elapsed_gpu_time, 
                                        "parallel_amplitude":parallel_amplitude, 
                                        "exact_amplitude":exact_amplitude})

    if rank == root:
        try:
            old_df = pd.read_pickle("data/sliced_tn_data.pkl")
        except:
            old_df = pd.DataFrame()
        df = pd.DataFrame(data_list)    
        df = pd.concat([old_df, df], ignore_index=True)
        pd.to_pickle(df, "data/sliced_tn_data.pkl")  

comm_nccl.destroy()