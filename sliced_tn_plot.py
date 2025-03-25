# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle("data/sliced_tn_data (strong scaling).pkl")
df['circuit'] = df['circuit'].str.replace("hidden_shift", "hs", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("hamiltonian_sim", "ham", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("random", "rand", case=False, regex=False)
df['error'] = abs(df['parallel_amplitude'] - df['exact_amplitude'])

# %%
sns.set_theme(style="whitegrid")

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    data=df, x="n_gpus", y="contract_time_seconds", hue="circuit", # col="circuit",
    capsize=.2, palette="bright", errorbar=("pi", 90),
    kind="point", height=4, aspect=1.25, log_scale=True
)
g.despine(left=True)
g.set(xlabel='# GPUs', ylabel='Contraction time (s)')

g.savefig('plots/pointplot_sliced.pdf')

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

df = pd.read_pickle("data/sliced_tn_data (pathfinding samples scaling).pkl")
df['circuit'] = df['circuit'].str.replace("hidden_shift", "hs", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("hamiltonian_sim", "ham", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("random", "rand", case=False, regex=False)
df['error'] = abs(df['parallel_amplitude'] - df['exact_amplitude'])
df = df.drop(columns=['num_qubits', 'slice_per_gpu', 'n_gpus', 'parallel_amplitude', 'exact_amplitude', 'error'])
df = pd.melt(df, id_vars=["circuit", "samples"], value_name="time").rename(columns={"variable": "operation"})
df['operation'] = df['operation'].str.replace("contract_time_seconds", "contraction", case=False, regex=False)
df['operation'] = df['operation'].str.replace("pathfinding_time_seconds", "pathfinding", case=False, regex=False)

sns.set_theme(style="whitegrid")

# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="circuit", hue="operation", legend_out=True, 
                     col_wrap=4, height=2.5, aspect=1.1, sharex=False, sharey=False)

# Draw a line plot to show the trajectory of each random walk
# grid.map(sns.lineplot, "samples", "time", marker="o", palette="bright", err_style="bars", legend=True)
grid.map(sns.pointplot, "samples", "time", marker="o", errorbar=("pi", 90))
grid.fig.tight_layout()
grid.set(yscale='log')
grid.add_legend()

grid.savefig('plots/pathfinding_vs_contract_time.pdf')

# %%
