# %%
import cupy as cp
import numpy as np

from cuquantum import contract, contract_path, CircuitToEinsum, tensor
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet.experimental import contract_decompose

np.random.seed(0)
cp.random.seed(0)

# We will reuse library handle in this notebook to reduce the context initialization time.
handle = cutn.create()
options = {'handle': handle}

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
nrepeats = 1
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "vqe", "hamiltonian_sim", "bv", "random"]
circuit_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
slice_limits = [2, 4, 8, 16, 32]
num_gpus = getDeviceCount()

# %%
for circuit_name in circuits:
    data_list = []

    for num_qubits in circuit_sizes:
        # Define quantum circuit and convert it to an Eninstein summation
        pass
        
        
        # Check correctness.
        # if rank == root:
        #     print(f"Circuit {circuit_name}({num_qubits} qubits) on {min_slices} slices and {num_gpus} gpus, total time required: {end_contract - start_contract}\n")
        #     data_list.append({"circuit":circuit_name, "num_qubits":num_qubits, "n_slices":min_slices, "n_gpus":num_gpus, "contract_time_seconds":end_contract - start_contract, "pathfinding_time_seconds":end_pathfinding - start_pathfinding})

    try:
        old_df = pd.read_pickle("data/approx_tn_data.pkl")
    except:
        old_df = pd.DataFrame()
    df = pd.DataFrame(data_list)    
    df = pd.concat([old_df, df])
    pd.to_pickle(df, "data/approx_tn_data.pkl")  
