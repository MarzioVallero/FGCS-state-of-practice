import numpy as np
import cupy as cp
from run import run
import pandas as pd
from cuquantum import contract
from cuquantum import CircuitToEinsum
from cuquantum.cutensornet.experimental import NetworkState, MPSConfig

np.random.seed(0)
cp.random.seed(0)
cp.cuda.Device(0).use()

frontend = "qiskit"
backend = "cutn"
nwarmups = 1
nrepeats = 10
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "vqe", "hamiltonian_sim", "bv", "random"]
circuit_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
rel_cutoffs = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

# Max number of eigenvalues truncated by SVD, quantum tensors are at max order-4
max_extent = 4

for circuit_name in circuits:
    data_list = []

    for num_qubits in circuit_sizes:
        # compute the bitstring amplitude
        bitstring = '0' * num_qubits

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

        exact_amplitude = None
        converter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)
        expression, operands = converter.amplitude(bitstring)
        exact_amplitude = contract(expression, *operands)

        for rel_cutoff in rel_cutoffs:                  
            for i in range(nwarmups + nrepeats):
                # select MPS as simulation method with truncation parameters
                mps_config = MPSConfig(max_extent=max_extent, rel_cutoff=rel_cutoff) # APPROX

                # create a NetworkState object and use it in a context manager
                with NetworkState.from_circuit(circuit, dtype='complex128', backend='cupy', config=mps_config) as state:
                    start_gpu = cp.cuda.Event()
                    end_gpu = cp.cuda.Event()
                    
                    start_gpu.record()
                    amplitude = state.compute_amplitude(bitstring, release_workspace=True)

                    end_gpu.record()
                    end_gpu.synchronize()
                    elapsed_gpu_time = float(cp.cuda.get_elapsed_time(start_gpu, end_gpu)) / 1000

                    if i >= nwarmups:
                        print(f"Circuit {circuit_name}({num_qubits} qubits) with {rel_cutoff} SVD cutoff (max_extent: {max_extent}), total time required: {elapsed_gpu_time}\n")
                        data_list.append({"circuit":circuit_name, "num_qubits":num_qubits, "rel_cutoff":rel_cutoff, "max_extent":max_extent, "contract_time_seconds":elapsed_gpu_time, "approx_amplitude":amplitude, "exact_amplitude":exact_amplitude})

    try:
        old_df = pd.read_pickle("data/approx_tn_data.pkl")
    except:
        old_df = pd.DataFrame()
    df = pd.DataFrame(data_list)    
    df = pd.concat([old_df, df])
    pd.to_pickle(df, "data/approx_tn_data.pkl")  
