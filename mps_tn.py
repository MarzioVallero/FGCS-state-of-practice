# %%
from cuquantum.cutensornet.experimental import NetworkState, MPSConfig
import cupy as cp
import numpy as np

np.random.seed(0)
cp.random.seed(0)

from cupy.cuda.runtime import getDeviceCount
from run import run

frontend = "qiskit"
backend = "cutn"
nwarmups = 1
nrepeats = 1
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "vqe", "hamiltonian_sim", "bv", "random"]
circuit_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
slice_limits = [2, 4, 8, 16, 32]
num_gpus = getDeviceCount()

# Breaking outer loops
circuit_name = "qft"
num_qubits = 32

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

# select exact MPS with gesvdj SVD algorithm as the simulation method
# we also use a low relative cutoff for SVD to filter out noise
config = MPSConfig(algorithm='gesvdj', rel_cutoff=1e-8) # EXACT

# select MPS as simulation method with truncation parameters
# config = MPSConfig(max_extent=4, rel_cutoff=1e-5) # APPROX

# create a NetworkState object and use it in a context manager
with NetworkState.from_circuit(circuit, dtype='complex128', backend='cupy', config=config) as state:
    # Optional, compute the final mps representation
    mps_tensors = state.compute_output_state()

    # Optional, compute the state vector
    # sv = state.compute_state_vector()
    # print(f"state vector type: {type(sv)}, shape: {sv.shape}")

    # compute the bitstring amplitude
    bitstring = '0' * num_qubits
    amplitude = state.compute_amplitude(bitstring)
    print(f"Bitstring amplitude for {bitstring}: {amplitude}")

# %%
