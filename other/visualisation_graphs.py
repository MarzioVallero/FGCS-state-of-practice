#%% Import
import os
import matplotlib.pyplot as plt
import dill
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
import json

#%%
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "random", "vqe", "hamiltonian_sim", "bv"] # ["qft","iqft","ghz","simon","hidden_shift","qaoa","qpe","quantum_volume","random","vqe","hamiltonian_sim","bv"]
max_num_qubits = 32
qubit_sizes = range(2, max_num_qubits+1)
backends = ["aer-cuda", "aer-cusv", "cusvaer", "cutn"]#,   "qsim-cuda", "qsim-cusv", "qsim-mgpu", "pennylane-lightning-gpu", "pennylane-lightning-qubit", "pennylane-lightning-kokkos", "qulacs-gpu"]

# %% Define utility functions
def show_day_hour_min(x, pos):
        if x < 1:
            x = x*10e2
            return f'{x:.0f} ms'
        else:
            if x < 60:
                return f'{x:.0f} s'
            else:
                x = np.round(x / 60)
                if x < 60:
                    return f'{x:.0f} min'
                else:
                    x = np.round(x / 60)
                    return f'{x:.0f} h'
            
def show_memory_size(x, pos):
    if x < 1024:
        return f'{x:.0f} bytes'
    else:
        x = np.round(x / 1024)
        if x < 1024:
            return f'{x:.0f} KB'
        else:
            x = np.round(x / 1024)
            if x < 1024:
                return f'{x:.0f} MB'
            else:
                x = np.round(x / 1024)
                return f'{x:.0f} GB'

# %% Create time-memory plots for each circuit
colours = ["limegreen", "navy", "hotpink", "gold", "darkorange"]
markers = ["s", "*", "h", "o", "p"]

for circuit in circuits:
    # Produce series
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.set_xlabel('# qubits')
    ax1.set_ylabel('time (seconds)')
    ax1.set_yscale("log")
    ax1.set_xticks(qubit_sizes)

    ax2.set_xlabel('# qubits')
    ax2.set_ylabel('memory (bytes)')
    ax2.set_yscale("log")
    ax2.set_xticks(qubit_sizes)

    for backend, colour, marker in zip(backends, colours, markers):
        try:
            with open(f"./data/{circuit}.json", 'rb') as file:
                dictionary = json.load(file)
                # qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series, circ2ein_time_series, contract_path_time_series = dill.load(file)
            qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series, circ2ein_time_series, contract_path_time_series = list, list, list, list, list, list
            for k, exp_dict in dictionary.items():
                qubit_sizes.append(int(k))
                for exp_hash, exp_data in exp_dict.items():
                    cpu_time_series.append()
                    # TODO MAKE THIS DATA IMOPRT WORK
        except:
            continue
        if backend == "cutn":
            series = [sum(i) for i in zip(gpu_time_series, circ2ein_time_series)]
        elif (backend != "cirq" and backend != "qsim"):
            series = gpu_time_series
        else:
            series = cpu_time_series
        print("len series: ", len(series), "range: ", range(2, len(series) + 1))
        ax1.plot(range(2, len(series) + 2), series, colour, marker=marker, label=f'{backend} time')
        ax2.plot(range(2, len(gpu_mem_series)+2), np.array(gpu_mem_series), colour, marker=marker, label=f'{backend} memory')

        print(qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series)

    ticks_time = [10e-4, 10e-3, 10e-2, 10e-1, 1, 5, 10, 30, 60, 60*10,60*30, 60*60]
    extra_ticks_time = [5, 30, 60, 60*10,60*30, 60*60]
    ax1.set_yticks(extra_ticks_time, minor=True)
    ax1.set_yticks(ticks_time)
    ax1.tick_params(axis='y', which='minor', labelcolor='gray', labelsize=9)
    ax1.yaxis.set_minor_formatter(show_day_hour_min)
    ax1.yaxis.set_major_formatter(show_day_hour_min)
    ax1.grid(True, which='minor', axis='y', color='gray', ls='--', lw=1, alpha=1)
    ax1.grid(True, which='major', axis='y', ls='--', lw=1, alpha=1)
    
    plt.draw()

    ticks_mem = [2*16, 256, 1024, 1024*4, 1024*32, 1024*128, 1024*1024, 1024*1024*4, 1024*1024*32, 1024*1024*256, 1024*1024*1024, 4*1024*1024*1024, 16*1024*1024*1024, 64*1024*1024*1024]
    ax2.set_yticks(ticks_mem)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.yaxis.set_major_formatter(show_memory_size)
    ax2.grid(True, which='major', axis='y', ls='--', lw=1, alpha=1)

    ax1.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend()
    ax1.set_title(f"{circuit} GPU time")
    ax2.set_title(f"{circuit} GPU memory usage")
    if not os.path.isdir(f"./plots/{circuit}"):
        os.makedirs(f"./plots/{circuit}")
    plt.savefig(f"./plots/{circuit}/{circuit}_performance_{max_num_qubits}_qubits.pdf", format="pdf", bbox_inches="tight")
    plt.close()

# %% Create memory usage graph for all circuits
circuits = ["qaoa", "random", "qpe", "qft", "hidden_shift", "hamiltonian_sim", "vqe", "bv"]
colours = ["green", "blue", "red", "black", "green", "black", "darkorange", "royalblue"]
markers = ["o", "o", "o", "x", "o", "o", "x", "o"]
has_plot_sv = False

# Produce series
fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

ax1.set_xlabel('# qubits')
ax1.set_ylabel('memory')
ax1.set_yscale("log")
ax1.set_xticks(qubit_sizes)

for circuit, colour, marker in zip(circuits, colours, markers):
    for backend in ["qsim-cusv", "cutn"]:
        if has_plot_sv and backend == "qsim-cusv":
            continue
        try:
            with open(f"./data/{backend}_{max_num_qubits}_{circuit}.pkl", 'rb') as file:
                qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series, circ2ein_time_series, contract_path_time_series = dill.load(file)
        except:
            continue
        if backend == "cutn":
            series = [sum(i) for i in zip(gpu_time_series, circ2ein_time_series)]
        elif (backend != "cirq" and backend != "qsim"):
            series = gpu_time_series
        else:
            series = cpu_time_series

        if not has_plot_sv and backend == "qsim-cusv":
            has_plot_sv = True

        ax1.plot(range(2, len(gpu_mem_series)+2), np.array(gpu_mem_series), (colour if backend == "cutn" else "magenta"), marker= (marker if backend == "cutn" else "*"), label=f'{circuit if backend == "cutn" else "statevector"} {backend}', mfc=("white" if circuit in ["qaoa", "qpe"] else colour))
    
plt.draw()
ticks_mem = [2*16, 256, 1024, 1024*4, 1024*32, 1024*128, 1024*1024, 1024*1024*4, 1024*1024*32, 1024*1024*256, 1024*1024*1024, 4*1024*1024*1024, 16*1024*1024*1024, 64*1024*1024*1024]
ax1.set_yticks(ticks_mem)
ax1.tick_params(axis='y', labelsize=9)
ax1.yaxis.set_major_formatter(show_memory_size)
ax1.grid(True, which='major', axis='y', ls='--', lw=1, alpha=1)

ax1.legend()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend()
ax1.set_title(f"GPU memory usage")
if not os.path.isdir(f"./plots/{circuit}"):
    os.makedirs(f"./plots/{circuit}")
plt.savefig(f"./plots/overall_memory_comparison_{max_num_qubits}_qubits.pdf", format="pdf", bbox_inches="tight")
plt.close()

# %% Heatmap for compute time
circuits = ["qaoa", "random", "qpe", "qft", "hidden_shift", "hamiltonian_sim", "vqe", "bv"]
colours = ["green", "blue", "red", "black", "green", "black", "darkorange", "royalblue"]
markers = ["o", "o", "o", "x", "o", "o", "x", "o"]
backends = ["qsim-cuda", "qsim-cusv", "cutn"]
heatmap_data = np.ones(shape=(2, len(circuits), max_num_qubits-1)) * np.inf
cmap = ListedColormap(['navy','hotpink','limegreen'])

# Produce series
fig, ax1 = plt.subplots(1, 1, figsize=(32, 20))

for row_index, (circuit, colour, marker) in enumerate(zip(circuits, colours, markers)):
    for backend in backends:
        try:
            with open(f"./data/{backend}_{max_num_qubits}_{circuit}.pkl", 'rb') as file:
                qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series, circ2ein_time_series, contract_path_time_series = dill.load(file)
        except:
            continue
        if backend == "cutn":
            series = [sum(i) for i in zip(gpu_time_series, circ2ein_time_series)]
        elif (backend != "cirq" and backend != "qsim"):
            series = gpu_time_series
        else:
            series = cpu_time_series

        series = np.array(series)
        print(series)
        for column_index, _ in enumerate(qubit_sizes):
            if series[column_index] < heatmap_data[1][row_index][column_index]:
                if backend == "qsim-cuda":
                    heatmap_data[0][row_index][column_index] = 0
                elif backend == "qsim-cusv":
                    heatmap_data[0][row_index][column_index] = 0.5
                elif backend == "cutn":
                    heatmap_data[0][row_index][column_index] = 1
                heatmap_data[1][row_index][column_index] = series[column_index]


# ax1.imshow(heatmap_data)
sns.heatmap(heatmap_data[0], annot=heatmap_data[1], square=True, cbar=False, cmap=cmap, fmt=".2e", ax=ax1)
ax1.set_xticks(ticks=np.arange(0, len(qubit_sizes))+0.5, labels=qubit_sizes)
ax1.set_yticks(ticks=np.arange(0, len(circuits))+0.5, labels=circuits, rotation=0)
print(heatmap_data)

plt.draw()
if not os.path.isdir(f"./plots/{circuit}"):
    os.makedirs(f"./plots/{circuit}")
plt.savefig(f"./plots/performance_heatmap_{max_num_qubits}_qubits.pdf", format="pdf", bbox_inches="tight")
plt.close()

# %% Speedup heatmap
circuits = ["qaoa", "random", "qpe", "qft", "hidden_shift", "hamiltonian_sim", "vqe", "bv"]
colours = ["green", "blue", "red", "black", "green", "black", "darkorange", "royalblue"]
markers = ["o", "o", "o", "x", "o", "o", "x", "o"]
backends = ["qsim-cusv", "cutn"]
base_comparison_backend = "qsim-cuda"
heatmap_data = np.ones(shape=(2*len(circuits), max_num_qubits-1)) * np.inf
cmap = "Spectral"

# Produce series
fig, ax1 = plt.subplots(1, 1, figsize=(25, 15))

for row_index, (circuit, colour, marker) in enumerate(zip(circuits, colours, markers)):
    with open(f"./data/{base_comparison_backend}_{max_num_qubits}_{circuit}.pkl", 'rb') as file:
        baseline_qubit_sizes, baseline_cpu_time_series, baseline_gpu_time_series, baseline_gpu_mem_series, baseline_circ2ein_time_series, baseline_contract_path_time_series = dill.load(file)
        if base_comparison_backend == "cutn":
            baseline_series = [sum(i) for i in zip(baseline_gpu_time_series, baseline_circ2ein_time_series)]
        elif (base_comparison_backend != "cirq" and base_comparison_backend != "qsim"):
            baseline_series = baseline_gpu_time_series
        else:
            baseline_series = baseline_cpu_time_series

    for backend_index, backend in enumerate(backends):
        try:
            with open(f"./data/{backend}_{max_num_qubits}_{circuit}.pkl", 'rb') as file:
                qubit_sizes, cpu_time_series, gpu_time_series, gpu_mem_series, circ2ein_time_series, contract_path_time_series = dill.load(file)
        except:
            continue
        if backend == "cutn":
            series = [sum(i) for i in zip(gpu_time_series, circ2ein_time_series)]
        elif (backend != "cirq" and backend != "qsim"):
            series = gpu_time_series
        else:
            series = cpu_time_series

        series = np.array(series)
        for column_index, _ in enumerate(qubit_sizes):
            heatmap_data[2*row_index+backend_index][column_index] = (baseline_series[column_index] / series[column_index]) - 1.0

#the choice of the levels depends on the data:
negative_colors = plt.cm.RdYlGn(np.linspace(0, 0.5, 256))
positive_colors = plt.cm.RdYlGn(np.linspace(0.5, 1, 256))
all_colors = np.vstack((negative_colors, positive_colors))
cmap_nonlin = LinearSegmentedColormap.from_list('terrain_map', all_colors)

# make the norm:  Note the center is offset so that the land has more
# dynamic range:
divnorm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=100.0)

vertical_labels = [f"{circuit}_{backend}" for circuit in circuits for backend in backends]

# ax1.imshow(heatmap_data)
sns.heatmap(heatmap_data, annot=True, square=True, norm=divnorm, cmap=cmap_nonlin, cbar=False, fmt=".1f", ax=ax1)
ax1.set_xticks(ticks=np.arange(0, len(qubit_sizes))+0.5, labels=qubit_sizes)
ax1.set_yticks(ticks=np.arange(0, len(vertical_labels))+0.5, labels=vertical_labels, rotation=0)
print(heatmap_data)

plt.draw()
if not os.path.isdir(f"./plots/{circuit}"):
    os.makedirs(f"./plots/{circuit}")
plt.savefig(f"./plots/speedup_heatmap_{max_num_qubits}_qubits.pdf", format="pdf", bbox_inches="tight")
plt.close()
# %%
