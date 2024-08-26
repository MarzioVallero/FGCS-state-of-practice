# %%
import os
from benchmarks import *
import networkx as nx
import re
import numpy as np
from copy import deepcopy
from multiprocessing.pool import Pool

# %% Circuit to DAG converter
def flatten(L):
    for item in L:
        if isinstance(item, list):
            for subitem in flatten(item): yield subitem
        else:
            yield item

def circuit_to_dag(circuit_object, nqubits):
    config = {"measure":True, "unfold":False, "p":1, "ansatz":"x", "num_layers":1, "time_step":1, "total_time":1}
    gate_sequence = circuit_object.generateGatesSequence(nqubits, config)

    print(f"There are {len(gate_sequence)} gates to be processed")

    graph = nx.MultiDiGraph()
    tensors = gate_sequence
    nodes_in_processing = {}
    for qubit in range(0, nqubits):
        graph.add_node(f"q{qubit}", kind="input", qubits=[qubit])
        nodes_in_processing[qubit] = f"q{qubit}"
    for index, tensor in enumerate(tensors):
        if tensor[0] == "measure":
            for qubit in flatten(tensor[1]):
                graph.add_node(f"m_{qubit}", kind=tensor[0], qubits=[qubit])
                graph.add_edge(nodes_in_processing[qubit], f"m_{qubit}", qubit=qubit)
                nodes_in_processing[qubit] = 0
        else:
            qubits_used_by_gate = [x for x in flatten(tensor[1][-2:]) if x in range(0, nqubits)]
            graph.add_node(f"{tensor[0]}_{index}", kind=tensor[0], qubits=qubits_used_by_gate)
            for qubit in qubits_used_by_gate:
                graph.add_edge(nodes_in_processing[qubit], f"{tensor[0]}_{index}", qubit=qubit)
                nodes_in_processing[qubit] = f"{tensor[0]}_{index}"

    return graph

def remove_single_qubit_gates(graph):
    g = graph.copy()

    while any(degree==2 for _, degree in g.degree):

        g0 = g.copy() #<- simply changing g itself would cause error `dictionary changed size during iteration` 
        for node, degree in g.degree():
            if degree==2:

                if g.is_directed(): #<-for directed graphs
                    a0,b0 = list(g0.in_edges(node))[0]
                    a1,b1 = list(g0.out_edges(node))[0]

                else:
                    edges = g0.edges(node)
                    edges = list(edges.__iter__())
                    a0,b0 = edges[0]
                    a1,b1 = edges[1]

                e0 = a0 if a0!=node else b0
                e1 = a1 if a1!=node else b1

                g0.remove_node(node)
                g0.add_edge(e0, e1)
        g = g0

    return g

# %% Metrics
def program_communication(dag, nqubits) -> float:
    graph = nx.Graph()
    for node in dag.nodes:
        if len(dag.nodes[node]["qubits"]) > 1:
            q1, q2 = dag.nodes[node]["qubits"]
            graph.add_edge(q1, q2)

    degree_sum = sum([graph.degree(n) for n in graph.nodes])

    return degree_sum / (nqubits * (nqubits - 1))

def critical_depth(dag, nqubits) -> float:
    n_ed = 0

    two_q_gates = [node for node in dag if len(dag.nodes[node]["qubits"]) > 1]
    try:
        path = nx.dag_longest_path(dag)
        n_ed += len([node for node in path if node in two_q_gates])
    except KeyError:
        pass

    n_e = len(two_q_gates)

    if n_ed == 0:
        return 0

    return n_ed / n_e

def entanglement_ratio(dag, nqubits) -> float:
    dag = deepcopy(dag)

    # remove input and output tensors
    for node, degree in dict(dag.degree()).items():
        if degree == 1:
            dag.remove_node(node)

    two_q_gates = [node for node in dag if len(dag.nodes[node]["qubits"]) > 1]

    return len(two_q_gates) / len(dag.nodes)

def parallelism(dag, nqubits) -> float:
    dag = deepcopy(dag)

    # remove input and output tensors
    for node, degree in dict(dag.degree()).items():
        if degree == 1:
            dag.remove_node(node)

    return max(1 - (len(nx.dag_longest_path(dag)) / len(dag.nodes)), 0)

def entanglement_variance(dag, nqubits) -> float:
    two_q_gates = [node for node in dag if len(dag.nodes[node]["qubits"]) > 1]

    avg_cnot = 2* len(two_q_gates) / nqubits
    numerator = 0
    for qubit in range(0, nqubits):
        value = len([node for node in two_q_gates if qubit in dag.nodes[node]["qubits"]])
        numerator += np.square(value-avg_cnot)
    numerator = np.log(numerator+1)

    return numerator/ nqubits

# %%
def run_instance(data):
    (metric_func, circuit, num_qubits ) = data
    res = metric_func(circuit, num_qubits)
    return res

def repeat_and_average(metric_func, circuit, num_qubits, repetitions):
    p = Pool(repetitions)
    res_list = p.map(run_instance, [(metric_func, circuit, num_qubits)] * repetitions)
    mean_res = np.mean(np.array(res_list))
    return mean_res

def decision_function(metrics):
    decision_list = []
    for num_qubits in range(len(metrics["entanglement_ratio"])):
        if metrics["entanglement_variance"][num_qubits][1] > 0.2:
            if metrics["entanglement_ratio"][num_qubits][1] > 0.5:
                decision_list.append(f"sv1")
                continue
        if metrics["critical_depth"][num_qubits][1] > 0.4:
            if metrics["parallelism"][num_qubits][1] > 0.5:
                if metrics["program_communication"][num_qubits][1] > 0.15:
                    decision_list.append(f"sv2")
                else:
                    decision_list.append(f"tn2")
            else:
                decision_list.append(f"tn3")
        else:
            decision_list.append(f"sv3")

    return decision_list

def compute_metrics_graphs():
    import matplotlib.pyplot as plt
    from time import time
    from benchmarks import qft, iqft, hidden_shift, qaoa, qpe, random, vqe, hamiltonian_simulation, bernstein_vazirani

    time_limit_per_circuit = 60 * 60 # 30 minutes

    circuits_to_test = [qaoa.QAOA(), random.Random(), qpe.QPE(), qft.QFT(),  vqe.VQE(), hamiltonian_simulation.HamiltonianSimulation(), hidden_shift.HiddenShift(), bernstein_vazirani.BernsteinVazirani()] #[ghz.GHZ(), qft.QFT(), iqft.IQFT(), simon.Simon(), hidden_shift.HiddenShift(), qaoa.QAOA(), qpe.QPE(), random.Random(), vqe.VQE(), hamiltonian_simulation.HamiltonianSimulation(), bernstein_vazirani.BernsteinVazirani()]
    max_num_qubits = 32
    qubit_sizes = range(2, max_num_qubits+1)
    metrics = [program_communication, critical_depth, entanglement_ratio, parallelism, entanglement_variance]
    colours = ["brown", "dodgerblue", "green", "gold", "magenta"]
    markers = ["o", "p", "D", "8", "s"]

    for circuit in circuits_to_test:
        results = {}
        print(circuit)
        circuit_name = re.findall(r"(?<=benchmarks.)(\w*)", str(circuit))[0]
        for num_qubits in qubit_sizes:
            start_time = time()
            try:
                dag_circuit = circuit_to_dag(circuit, num_qubits)
            except Exception as e:
                print(f"Exception: {str(e)}")
                continue
            print(f"Computing metrics for {circuit_name} of size {num_qubits}")
            for metric in metrics:
                metric_name = re.findall(r"(?<=<function )(\w*)", str(metric))[0]
                if metric_name not in results.keys():
                    results[metric_name] = []
                metric_result = metric(dag_circuit, num_qubits) # Single sample
                # metric_result = repeat_and_average(metric, dag_circuit, num_qubits, 100)
                results[metric_name].append((num_qubits, metric_result))
                end_time = time()
                # print(f"Time needed for {metric_name} {circuit_name} {num_qubits} qubits: {end_time - start_time}")
            
            if end_time - start_time > time_limit_per_circuit:
                break
        
        fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5))

        ax1.set_xlabel('# qubits')
        ax1.set_ylabel('metrics')
        #ax1.set_ylim([-0.1, 1.1])
        ax1.set_xticks(qubit_sizes)

        for metric, colour, marker in zip(metrics, colours, markers):
            metric_name = re.findall(r"(?<=<function )(\w*)", str(metric))[0]
            ax1.plot(list(zip(*results[metric_name]))[0], list(zip(*results[metric_name]))[1], colour, marker=marker, label=f'{metric_name}')

        plt.draw()
        ax1.legend()
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        ax1.set_title(f"{circuit_name} metrics")
        if not os.path.isdir(f"./plots/metrics"):
            os.makedirs(f"./plots/metrics")
        plt.savefig(f"./plots/metrics/{circuit_name}_metrics_{max_num_qubits}_qubits.pdf", format="pdf", bbox_inches="tight")
        plt.close()

        print(circuit_name, decision_function(results))

# %%
if __name__ == "__main__":
    compute_metrics_graphs()
# %%
