# %%
"""
Dataset generation for debugging
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
import random

frontend = "qiskit"
backend = "cutn"
nwarmups = 1
nrepeats = 1
config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
circuits = ["qft", "qpe", "qaoa", "hidden_shift", "vqe", "bv", "hamiltonian_sim", "random"]
circuit_sizes = [2, 4, 8, 12, 16, 20, 24, 28, 32]
slice_limits = [2, 4, 8, 16, 32]
gpus = [1,2,4,8]

for num_gpus in gpus:
    for circuit_name in circuits:
        data_list = []

        for num_qubits in circuit_sizes:
            if circuit_name == "random" and num_qubits >= 28:
                continue
            for slice_limit in slice_limits:
                data_list.append({"circuit":circuit_name, "num_qubits":num_qubits, "n_slices":slice_limit, "n_gpus":num_gpus, "contract_time_seconds":(random.random()*num_qubits)/(num_gpus), "pathfinding_time_seconds":random.random()})

        try:
            old_df = pd.read_pickle("data/example.pkl")
        except:
            old_df = pd.DataFrame()
        df = pd.DataFrame(data_list)    
        df = pd.concat([old_df, df], ignore_index=True)
        pd.to_pickle(df, "data/example.pkl")  

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle("data/sliced_tn_data.pkl")
df['circuit'] = df['circuit'].str.replace("hidden_shift", "hs", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("hamiltonian_sim", "ham", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("random", "rand", case=False, regex=False)

# %%
sns.set_theme(style="whitegrid")

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    data=df, x="num_qubits", y="contract_time_seconds", hue="n_gpus", col="n_slices",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75, log_scale=True
)
g.despine(left=True)
# %%
sns.set_theme(style="whitegrid")

# Set up a grid to plot survival probability against several variables
g = sns.PairGrid(df, y_vars="contract_time_seconds",
                 x_vars=["n_gpus", "n_slices", "circuit"],
                 hue_kws=["n_gpus", "n_slices", "circuit"],
                 palette="YlGnBu_d",
                 height=5, aspect=.8)

# Draw a seaborn pointplot onto each Axes
g.map(sns.violinplot, log_scale=True)
sns.despine(fig=g.fig, left=True)
# %%
sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
ax.set_xscale("log")
ax.set_xlim(xmin=10**(-4.00001), xmax=df["contract_time_seconds"].max()+1)

# Plot the orbital period with horizontal boxes
sns.boxplot(
    df, x="contract_time_seconds", y="circuit", hue="n_gpus",
    whis=[0, 100], width=0.8, palette="vlag", gap=0.3, 
)

# Add in points to show each observation
sns.stripplot(df, x="contract_time_seconds", y="circuit", hue="n_gpus", size=2, palette="vlag", legend=False)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

# %%
sns.set_theme(style="darkgrid")

# Plot the responses for different events and regions
g = sns.lineplot(x="n_gpus", y="contract_time_seconds",
             hue="n_slices",
             data=df)

g.set(yscale='log')
# %%
