# %%
"""
Example illustrating slice-based parallel tensor network contraction with cuQuantum using MPI.

$ mpiexec -n 4 python example2_mpi.py
"""
# Sphinx
import numpy as np
import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

import qiskit
from cuquantum import CircuitToEinsum
from cuquantum import Network
from run import run
import pandas as pd
import pickle

frontend = "qiskit"
backend = "cutn"
nwarmups = 1
nrepeats = 2
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "vqe", "bv", "hamiltonian_sim", "random"]
circuit_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
slice_limits = [2, 4, 8, 16, 32]
num_gpus = getDeviceCount()

# Set up the MPI communicator.
root = 0
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Set up the NCCL communicator.
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

# %%
for circuit_name in circuits:
    data_list = []

    for num_qubits in circuit_sizes:
        if circuit_name == "random" and num_qubits >= 28:
            continue
        for slice_limit in slice_limits:
            # Define quantum circuit and convert it to an Eninstein summation
            args = ["circuit", "--frontend", frontend, "--backend", backend, "--benchmark", circuit_name, "--nqubits", f"{num_qubits}", "--nwarmups", f"{nwarmups}", "--nrepeats", f"{nrepeats}", "--new"]
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
            circuit.draw()

            # Hyperparameters
            samples = 8
            # min_slices = max(16, size)
            min_slices = slice_limit

            for i in range(nwarmups + nrepeats):

                myconverter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)
                expression, operands_conv = myconverter.amplitude("0"*num_qubits) #.statevector() is bottlenecked by memory

                # Note that all NCCL operations must be performed in the correct device context.
                device_id = rank % getDeviceCount()
                cp.cuda.Device(device_id).use()

                # Set the operand data on root.
                if rank == root:
                    operands = operands_conv
                else:
                    operands = [cp.empty(o.shape, dtype='complex128') for o in operands_conv]

                # Broadcast the operand data. We pass in the CuPy ndarray data pointers to the NCCL APIs.
                stream_ptr = cp.cuda.get_current_stream().ptr
                for operand in operands:
                    comm_nccl.broadcast(operand.data.ptr, operand.data.ptr, operand.size*16, nccl.NCCL_CHAR, root, stream_ptr)

                # # Create network object.
                with Network(expression, *operands) as network:
                    comm.Barrier()
                    if rank == root:
                        start_pathfinding = MPI.Wtime()

                    # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
                    path, info = network.contract_path(optimize={'samples': samples, 'slicing': {'min_slices': min_slices}})

                    # Select the best path from all ranks. Note that we still use the MPI communicator here for simplicity.
                    opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
                    if rank == root:
                        print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

                    # Broadcast info from the sender to all other ranks.
                    info = comm.bcast(info, sender)

                    # Set path and slices.
                    path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})

                    comm.Barrier()
                    if rank == root:
                        end_pathfinding = MPI.Wtime()

                    # Calculate this process's share of the slices.
                    num_slices = info.num_slices
                    chunk, extra = num_slices // size, num_slices % size
                    slice_begin = rank * chunk + min(rank, extra)
                    slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
                    slices = range(slice_begin, slice_end)

                    print(f"Process {rank} is processing slice range: {slices}.")

                    comm.Barrier()
                    if rank == root:
                        start_contract = MPI.Wtime()

                    # Contract the group of slices the process is responsible for.
                    result = network.contract(slices=slices, release_workspace=True)

                    # Sum the partial contribution from each process on root.
                    stream_ptr = cp.cuda.get_current_stream().ptr
                    comm_nccl.reduce(result.data.ptr, result.data.ptr, result.size*16, nccl.NCCL_CHAR, nccl.NCCL_SUM, root, stream_ptr)

                    cp.cuda.runtime.deviceSynchronize()
                    comm.Barrier()
                    if rank == root:
                        end_contract = MPI.Wtime()

                    # Check correctness.
                    if rank == root and i >= nwarmups:
                        print(f"Circuit {circuit_name}({num_qubits} qubits) on {min_slices} slices and {num_gpus} gpus, total time required: {end_contract - start_contract}\n")
                        data_list.append({"circuit":circuit_name, "num_qubits":num_qubits, "n_slices":min_slices, "n_gpus":num_gpus, "contract_time_seconds":end_contract - start_contract, "pathfinding_time_seconds":end_pathfinding - start_pathfinding})



    if rank == root:
        try:
            old_df = pd.read_pickle("data/sliced_tn_data.pkl")
        except:
            old_df = pd.DataFrame()
        df = pd.DataFrame(data_list)    
        df = pd.concat([old_df, df], ignore_index=True)
        pd.to_pickle(df, "data/sliced_tn_data.pkl")  

comm_nccl.destroy()

# %%
