import cupy as cp
from run import run
import pandas as pd
from cuquantum import contract
from cuquantum import CircuitToEinsum
from cuquantum.cutensornet.experimental import NetworkState, MPSConfig
from itertools import product

frontend = "qiskit"
backend = "cutn"
nwarmups = 0
nrepeats = 1
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "vqe", "hamiltonian_sim", "bv", "random"]
circuit_sizes = [32] # [2, 4, 8, 12, 16, 20, 24, 28, 32]
parameters = {"gesvd": {"max_extent":[1, 2, 3, 4], "abs_cutoff":[1e-32], "gesvdj_tol":[0], "gesvdj_max_sweeps":[0], "gesvdr_niters":[0]}, 
              "gesvdj":{"max_extent":[4], "abs_cutoff":[1e-32], "gesvdj_tol":[1e-2, 1e-4, 1e-6, 1e-8], "gesvdj_max_sweeps":[10, 50, 100, 500, 1000], "gesvdr_niters":[0]}, 
              "gesvdr":{"max_extent":[4], "abs_cutoff":[1e-32], "gesvdj_tol":[0], "gesvdj_max_sweeps":[0], "gesvdr_niters":[1, 2, 3, 4, 5, 6]}
              }

# Remove memory manager to avoid memory leaks
cp.cuda.set_allocator(None)

for circuit_name in circuits:
    data_list = []

    for num_qubits in circuit_sizes:
        # compute the bitstring amplitude
        bitstring = ["0" if bit % 2 == 0 else "1" for bit in range(num_qubits)]

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
        exact_amplitude = exact_amplitude.item()

        for algorithm, args in parameters.items():
            for max_extent, abs_cutoff, gesvdj_tol, gesvdj_max_sweeps, gesvdr_niters in product(args["max_extent"], args["abs_cutoff"], args["gesvdj_tol"], args["gesvdj_max_sweeps"], args["gesvdr_niters"]):
                mps_config = MPSConfig(algorithm=algorithm, 
                                       max_extent=max_extent, abs_cutoff=abs_cutoff, 
                                       gesvdj_tol=gesvdj_tol, gesvdj_max_sweeps=gesvdj_max_sweeps, 
                                       gesvdr_niters=gesvdr_niters)
                
                for i in range(nwarmups + nrepeats):
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
                            print(f"Circuit {circuit_name}({num_qubits} qubits) with SVD ({algorithm}) cutoff (max_extent: {max_extent}), total time required: {elapsed_gpu_time} s\nResult exact: {exact_amplitude}\nResult paral: {amplitude}\nDifference: {exact_amplitude-amplitude}\n")
                            data_list.append({"circuit":circuit_name, "num_qubits":num_qubits, "algorithm":algorithm, "max_extent":max_extent, "abs_cutoff":abs_cutoff, "gesvdj_tol":gesvdj_tol, "gesvdj_max_sweeps":gesvdj_max_sweeps, "gesvdr_niters":gesvdr_niters, "contract_time_seconds":elapsed_gpu_time, "approx_amplitude":amplitude, "exact_amplitude":exact_amplitude})

    try:
        old_df = pd.read_pickle("data/approx_tn_data.pkl")
    except:
        old_df = pd.DataFrame()
    df = pd.DataFrame(data_list)    
    df = pd.concat([old_df, df], ignore_index=True)
    pd.to_pickle(df, "data/approx_tn_data.pkl")  
