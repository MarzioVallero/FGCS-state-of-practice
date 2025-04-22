#%% Import
# import os
# import matplotlib.pyplot as plt
# import subprocess
# import dill
# import re
# import numpy as np
from run import run

#%%
circuits_to_test = ["qft"] #, "qpe", "qaoa", "hidden_shift", "random", "vqe", "hamiltonian_sim", "bv"]
max_num_qubits = 3
qubit_sizes = range(2, max_num_qubits+1)
frontends = ["qiskit"] #["qiskit",   "qiskit",   "qiskit",  "qiskit", "cirq",      "cirq",      "cirq",      "pennylane",               "pennylane",                 "pennylane",                  "qulacs"]
backends =  ["cusvaer"] #["aer-cuda", "aer-cusv", "cusvaer", "cutn",   "qsim-cuda", "qsim-cusv", "qsim-mgpu", "pennylane-lightning-gpu", "pennylane-lightning-qubit", "pennylane-lightning-kokkos", "qulacs-gpu"]
nwarmups = 1
nrepeats = 1
exec_time_limit = 0


# %%
for circuit in circuits_to_test:
    for frontend, backend in zip(frontends, backends):
        cpu_time_series = []
        gpu_time_series = []
        gpu_mem_series = []
        circ2ein_time_series = []
        contract_path_time_series = []
        for num_qubits in qubit_sizes:
            args = ["circuit", "--frontend", frontend, "--backend", backend, "--benchmark", circuit, "--nqubits", f"{num_qubits}", "--nwarmups", f"{nwarmups}", "--nrepeats", f"{nrepeats}", "--new"]
            if backend != 'cirq' and backend != 'qsim':
                args.append("--ngpus")
                args.append("1")
            if backend == "cusvaer":
                args.append("--cusvaer-global-index-bits")
                args.append("2,0") # X,Y where 2^X is GPUs per node, 2^Y is number of nodes
                args.append("--cusvaer-p2p-device-bits")
                args.append("4") # Number of GPUs on node with direct interconnect
            # gpu_mod = "--ngpus 1" if (backend != 'cirq' and backend != 'qsim') else ""
            # command = f"/home/marzio.vallero/miniconda3/envs/callisto/bin/python /home/marzio.vallero/misty-rainforest/run.py circuit --frontend {frontend} --backend {backend} --benchmark {circuit} --nqubits {num_qubits} {gpu_mod} --nwarmups {nwarmups} --nrepeats {nrepeats} --new"
            # print(command)
            # try:
                # process_output = subprocess.check_output(command, shell=True)
                # process_output = process_output.decode("utf-8")
            run(args)
                # print(f"\033[H\033[JCompleted {circuit} on {backend} with qubit size {num_qubits}\n{process_output}", end="")
                # cpu_time = re.findall("(?<=\[CPU\] Averaged elapsed time: )\d+\.\d+", process_output)
                # gpu_time = re.findall("(?<=\[GPU\] Averaged elapsed time: )\d+\.\d+", process_output)
                # gpu_mem = re.findall("(?<=\ occupancy: )\d+", process_output)
                # cpu_time_series.append(float(cpu_time[0]))
                # gpu_time_series.append(float(gpu_time[0]))
                # gpu_mem_series.append(float(gpu_mem[0]))
                # if backend == "cutn":
                #     circ2ein_time = re.findall("(?<=CircuitToEinsum took )\d+\.\d+", process_output)
                #     contract_path_time = re.findall("(?<=contract_path\(\) took )\d+\.\d+", process_output)
                #     circ2ein_time_series.append(float(circ2ein_time[0]))
                #     contract_path_time_series.append(float(contract_path_time[0]))        
            # except:
            #     continue
            # if (exec_time_limit != 0 and float(cpu_time[0]) * (nwarmups + nrepeats) > exec_time_limit):
            #     print(f"Execution time limit {exec_time_limit} exceeded, next instances will be skipped.")
            #     break

        # with open(f"./data/{backend}_{max_num_qubits}_{circuit}.pkl", 'wb') as file:
        #     dill.dump([qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series, circ2ein_time_series, contract_path_time_series], file)

# %%
