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

# Define quantum circuit and convert it to an Eninstein summation
circuit_name = "qft"
frontend = "qiskit"
backend = "cutn"
nwarmups = 1
nrepeats = 1
num_qubits = 32

from run import run
args = ["circuit", "--frontend", frontend, "--backend", backend, "--benchmark", circuit_name, "--nqubits", f"{num_qubits}", "--nwarmups", f"{nwarmups}", "--nrepeats", f"{nrepeats}", "--new"]
runner = run(args, get_runner=True)
runner.nqubits = num_qubits
b = runner._benchmarks[circuit_name]
benchmark_object = b['benchmark']
benchmark_config = b['config']
benchmark_config['precision'] = "single"  # some frontends may need it
runner.benchmark_name = circuit_name
runner.benchmark_object = benchmark_object
runner.benchmark_config = benchmark_config
circuit = runner._load_or_generate_circuit(f"circuits/{circuit_name}_{num_qubits}")
circuit.draw()

#%%
start = MPI.Wtime()

root = 0
comm = MPI.COMM_WORLD

rank, size = comm.Get_rank(), comm.Get_size()

# Hyperparameters
samples = 8
min_slices = 32 #max(16, size)

myconverter = CircuitToEinsum(circuit, dtype='complex128', backend=cp)
expression, operands_conv = myconverter.amplitude("0"*num_qubits) #.statevector() is bottlenecked by memory

# # Broadcast the operand data.
# operands = comm.bcast(operands, root)

# # Assign the device for each process.
# device_id = rank % getDeviceCount()

# # Create network object.
# network = Network(expression, *operands, options={'device_id' : device_id})

# # Compute the path on all ranks with 8 samples for hyperoptimization. Force slicing to enable parallel contraction.
# path, info = network.contract_path(optimize={'samples': samples, 'slicing': {'min_slices': min_slices}})

# # Select the best path from all ranks.
# opt_cost, sender = comm.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)
# if rank == root:
#     print(f"Process {sender} has the path with the lowest FLOP count {opt_cost}.")

# # Broadcast info from the sender to all other ranks.
# info = comm.bcast(info, sender)

# # Set path and slices.
# path, info = network.contract_path(optimize={'path': info.path, 'slicing': info.slices})

# # Calculate this process's share of the slices.
# num_slices = info.num_slices
# chunk, extra = num_slices // size, num_slices % size
# slice_begin = rank * chunk + min(rank, extra)
# slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
# slices = range(slice_begin, slice_end)

# print(f"Process {rank} is processing slice range: {slices}.")

# # Contract the group of slices the process is responsible for.
# result = network.contract(slices=slices)

# # Sum the partial contribution from each process on root.
# result = comm.reduce(sendobj=result, op=MPI.SUM, root=root)

# end = MPI.Wtime()

# # Check correctness.
# if rank == root:
#    print(f"Total time required: {end-start}")

#    result_np = np.einsum(expression, *operands, optimize=True)
#    print("Does the cuQuantum parallel contraction result match the numpy.einsum result?", np.allclose(result, result_np))

#%%
# Note that all NCCL operations must be performed in the correct device context.
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()

# Set up the NCCL communicator.
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)

# Set the operand data on root.
if rank == root:
    operands = operands_conv
else:
    operands = [cp.empty(o.shape, dtype='complex128') for o in operands_conv]

# Broadcast the operand data. We pass in the CuPy ndarray data pointers to the NCCL APIs.
stream_ptr = cp.cuda.get_current_stream().ptr
for operand in operands:
    comm_nccl.broadcast(operand.data.ptr, operand.data.ptr, operand.size*16, nccl.NCCL_CHAR, root, stream_ptr)

# Create network object.
network = Network(expression, *operands)

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

# Calculate this process's share of the slices.
num_slices = info.num_slices
chunk, extra = num_slices // size, num_slices % size
slice_begin = rank * chunk + min(rank, extra)
slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
slices = range(slice_begin, slice_end)

print(f"Process {rank} is processing slice range: {slices}.")

# Contract the group of slices the process is responsible for.
result = network.contract(slices=slices)

# Sum the partial contribution from each process on root.
stream_ptr = cp.cuda.get_current_stream().ptr
comm_nccl.reduce(result.data.ptr, result.data.ptr, result.size*16, nccl.NCCL_CHAR, nccl.NCCL_SUM, root, stream_ptr)


# Check correctness.
if rank == root:
    print(f"Process {rank}:\n{result}")
#     result_cp = cp.einsum(expression, *operands_conv, optimize=True)
#     print("Does the cuQuantum parallel contraction result match the cupy.einsum result?", cp.allclose(result, result_cp))
#     print(result_cp)