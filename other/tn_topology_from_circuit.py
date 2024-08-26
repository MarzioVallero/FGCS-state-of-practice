#%%
from benchmarks import qft, hidden_shift, qaoa, qpe, random, vqe, hamiltonian_simulation, bernstein_vazirani
import networkx as nx
# import networkx as nx
import matplotlib.pyplot as plt
from math import log2
from compute_metrics import circuit_to_dag, remove_single_qubit_gates
import time
import os
import re

#%%
nqubits = 32
circuit_objects = [vqe.VQE()] #[qft.QFT(), qpe.QPE(), qaoa.QAOA(), hidden_shift.HiddenShift(), random.Random(), vqe.VQE(), hamiltonian_simulation.HamiltonianSimulation(), bernstein_vazirani.BernsteinVazirani()] #[ghz.GHZ(), qft.QFT(), iqft.IQFT(), simon.Simon(), hidden_shift.HiddenShift(), qaoa.QAOA(), qpe.QPE(), random.Random(), vqe.VQE(), hamiltonian_simulation.HamiltonianSimulation(), bernstein_vazirani.BernsteinVazirani()]
plot_graph = True
single_qubit_gates = False

for circuit_object in circuit_objects:
    t1 = time.perf_counter()
    graph = circuit_to_dag(circuit_object, nqubits)
    if not single_qubit_gates:
        graph = remove_single_qubit_gates(graph)
   
    # pos = nx.shell_layout(graph, nlist=[inner, middle, outer])
    pos = nx.nx_agraph.graphviz_layout(graph, prog="fdp", args="-Glen=100 -Gmaxiter=10000 -Glen=1")
    print(f"DAG conversion time: {time.perf_counter() - t1}")

    # print(graph.nodes(data=True))

    circuit_name = re.findall(r"(?<=benchmarks.)(\w*)", str(circuit_object))[0]
    node_size_multiplier = 10
    color_state_map = [graph.degree[node] for node in list(graph.nodes())]
    color_state_map[len(color_state_map)-nqubits:len(color_state_map)] = (5 for _ in range(nqubits))
    node_size_map = [graph.degree[node]*node_size_multiplier for node in list(graph.nodes())]
    dynamic_dim = round(2*log2(len(graph.nodes)))

    if plot_graph:
        plt.figure(3, figsize=(10, 5)) 
        # ax = plt.gca()
        # for e in graph.edges:
        #     ax.annotate("",
        #                 xy=pos[e[0]], xycoords='data',
        #                 xytext=pos[e[1]], textcoords='data',
        #                 arrowsize=1,
        #                 arrowprops=dict(arrowstyle="<|-", color="0.5",
        #                                 # shrinkA=10, shrinkB=8,
        #                                 patchA=None, patchB=None,
        #                                 connectionstyle="arc3,rad=rrr".replace('rrr',str(0.4*e[2])),),
        #                 )
        nx.draw_networkx_edges(
            graph, pos,
            edge_color="black", arrowstyle="<|-",
            arrowsize=4, node_size=node_size_map
        )
        nx.draw_networkx_nodes(
            graph, pos,
            node_size=node_size_map, 
            node_color=color_state_map,
            edgecolors="black",
            cmap="Spectral",
            alpha=1,
        )
        labels = {}    
        for node in graph.nodes():
            labels[node] = node
        # nx.draw_networkx_labels(graph, pos, labels=labels)
        plt.axis('off')
        plt.draw()
        if not os.path.isdir(f"./plots/{circuit_name}"):
            os.makedirs(f"./plots/{circuit_name}")
        plt.savefig(f"./plots/{circuit_name}/{circuit_name}_topology_{nqubits}_qubits.pdf", format="pdf", bbox_inches="tight")
        plt.close()

    n_input, n_output, n_mid, n_multi_qubit_gates = (0, 0, 0, 0)
    for node in graph.nodes.items():
        if node[1]["kind"] == "input":
            n_input += 1
        elif node[1]["kind"] == "measure":
            n_output += 1
        else:
            n_mid += 1
        if len(node[1]["qubits"]) > 1:
            n_multi_qubit_gates += 1
    print(f"Statistics for {circuit_name}:\nn_input: {n_input}, n_mid: {n_mid}, n_output: {n_output}, n_single_qubit_gates: {n_mid-n_multi_qubit_gates}, n_multi_qubit_gates: {n_multi_qubit_gates}\n")

# %%
