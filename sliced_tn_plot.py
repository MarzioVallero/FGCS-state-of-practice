# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle("data/sliced_tn_data.pkl")
df['circuit'] = df['circuit'].str.replace("hidden_shift", "hs", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("hamiltonian_sim", "ham", case=False, regex=False)
df['circuit'] = df['circuit'].str.replace("random", "rand", case=False, regex=False)
df['error'] = abs(df['parallel_amplitude'] - df['exact_amplitude'])

# %%
sns.set_theme(style="whitegrid")

# Draw a pointplot to show pulse as a function of three categorical factors
g = sns.catplot(
    data=df, x="num_qubits", y="contract_time_seconds", hue="n_gpus", col="n_slices",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75, log_scale=True
)
g.despine(left=True)

g.savefig('plots/pointplot_sliced.pdf')

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

g.savefig('plots/violin_sliced.pdf')

# %%
sns.set_theme(style="ticks")

# Initialize the figure with a logarithmic x axis
f, ax = plt.subplots(figsize=(7, 6))
ax.set_xscale("log")
ax.set_xlim(xmin=10**(-4.00001), xmax=df["contract_time_seconds"].max()+1)

# Plot the orbital period with horizontal boxes
sns.boxplot(
    df, x="contract_time_seconds", y="circuit", hue="n_gpus",
    whis=[0, 100], width=0.8, palette="Spectral_r", gap=0.3, 
)

# Add in points to show each observation
sns.stripplot(df, x="contract_time_seconds", y="circuit", hue="n_gpus", size=2, palette="Spectral_r", legend=False)

# Tweak the visual presentation
ax.xaxis.grid(True)
ax.set(ylabel="")
sns.despine(trim=True, left=True)

f.savefig('plots/distr_sliced.pdf')
