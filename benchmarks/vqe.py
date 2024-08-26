# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark
import cirq
from collections import Counter
from copy import deepcopy
import numpy as np
import scipy.optimize as opt
from _utils import Gate


class VQE(Benchmark):
    """Proxy benchmark of a full VQE application that targets a single iteration
    of the whole variational optimization.

    The benchmark is parameterized by the number of qubits, n. For each value of
    n, we classically optimize the ansatz, sample 3 iterations near convergence,
    and use the sampled parameters to execute the corresponding circuits on the
    QPU. We take the measured energies from these experiments and average their
    values and compute a score based on how closely the experimental results are
    to the noiseless values.
    """

    @staticmethod
    def generateGatesSequence(nqubits, config):
        circuit = []
        num_qubits = nqubits
        num_layers = config["num_layers"]
        hamiltonian = VQE.gen_tfim_hamiltonian(num_qubits)
        _params = VQE.gen_angles(num_qubits, num_layers)

        z_ansatz, x_ansatz = VQE._gen_ansatz(_params, num_qubits, num_layers)

        if (config["ansatz"] == "x"):
            return x_ansatz
        else:
            return z_ansatz

    def gen_tfim_hamiltonian(num_qubits) -> list:
        r"""Generate an n-qubit Hamiltonian for a transverse-field Ising model (TFIM).

            $H = \sum_i^n(X_i) + \sum_i^n(Z_i Z_{i+1})$

        Example of a 6-qubit TFIM Hamiltonian:

            $H_6 = XIIIII + IXIIII + IIXIII + IIIXII + IIIIXI + IIIIIX + ZZIIII
                  + IZZIII + IIZZII + IIIZZI + IIIIZZ + ZIIIIZ$
        """
        hamiltonian = []
        for i in range(num_qubits):
            hamiltonian.append(["X", i, 1])  # [Pauli type, qubit idx, weight]
        for i in range(num_qubits - 1):
            hamiltonian.append(["ZZ", (i, i + 1), 1])
        hamiltonian.append(["ZZ", (num_qubits - 1, 0), 1])
        return hamiltonian

    def _gen_ansatz(params, num_qubits, num_layers) -> list:
        qubits = range(num_qubits)
        z_circuit = []

        param_counter = 0
        for _ in range(num_layers):
            # Ry rotation block
            for i in range(num_qubits):
                z_circuit.append(Gate(id='ry', params=2 * params[param_counter], targets=qubits[i]) )
                param_counter += 1
            # Rz rotation block
            for i in range(num_qubits):
                z_circuit.append(Gate(id='rz', params=2 * params[param_counter], targets=qubits[i]) )
                param_counter += 1
            # Entanglement block
            for i in range(num_qubits - 1):
                z_circuit.append(Gate(id='cnot', controls=qubits[i], targets=qubits[i + 1]) )
            # Ry rotation block
            for i in range(num_qubits):
                z_circuit.append(Gate(id='ry', params=2 * params[param_counter], targets=qubits[i]) )
                param_counter += 1
            # Rz rotation block
            for i in range(num_qubits):
                z_circuit.append(Gate(id='rz', params=2 * params[param_counter], targets=qubits[i]) )
                param_counter += 1

        x_circuit = deepcopy(z_circuit)
        for q in qubits:
            x_circuit.append(Gate(id='h', targets=q))

        # Measure all qubits
        z_circuit.append(Gate(id='measure', targets=list(range(num_qubits))))
        x_circuit.append(Gate(id='measure', targets=list(range(num_qubits))))

        return [z_circuit, x_circuit]

    def _parity_ones(bitstr: str) -> int:
        one_count = 0
        for i in bitstr:
            if i == "1":
                one_count += 1
        return one_count % 2

    def _calc(bit_list, bitstr, probs: Counter) -> float:
        energy = 0.0
        for item in bit_list:
            if VQE._parity_ones(item) == 0:
                energy += probs.get(bitstr, 0)
            else:
                energy -= probs.get(bitstr, 0)
        return energy

    def _get_expectation_value_from_probs(probs_z: Counter, probs_x: Counter) -> float:
        avg_energy = 0.0

        # Find the contribution to the energy from the X-terms: \sum_i{X_i}
        for bitstr in probs_x.keys():
            bit_list_x = [bitstr[i] for i in range(len(bitstr))]
            avg_energy += VQE._calc(bit_list_x, bitstr, probs_x)

        # Find the contribution to the energy from the Z-terms: \sum_i{Z_i Z_{i+1}}
        for bitstr in probs_z.keys():
            # fmt: off
            bit_list_z = [bitstr[(i - 1): (i + 1)] for i in range(1, len(bitstr))]
            # fmt: on
            bit_list_z.append(bitstr[0] + bitstr[-1])  # Add the wrap-around term manually
            avg_energy += VQE._calc(bit_list_z, bitstr, probs_z)

        return avg_energy

    def _get_opt_angles(num_qubits, num_layers) -> tuple:
        def f(params, num_qubits, num_layers) -> float:
            z_circuit, x_circuit = VQE._gen_ansatz(params, num_qubits, num_layers)
            z_probs = VQE.get_ideal_counts(z_circuit)
            x_probs = VQE.get_ideal_counts(x_circuit)
            energy = VQE._get_expectation_value_from_probs(z_probs, x_probs)

            return -energy  # because we are minimizing instead of maximizing

        init_params = [
            np.random.uniform() * 2 * np.pi for _ in range(num_layers * 4 * num_qubits)
        ]

        # Skip the optimisation loop, as we are only interested in the simulation performance
        # out = opt.minimize(f, init_params, args=(num_qubits, num_layers), method="COBYLA")

        # return out["x"], out["fun"]

        return init_params, True

    def gen_angles(num_qubits, num_layers) -> list:
        """Classically simulate the variational optimization and return
        the final parameters.
        """
        params, _ = VQE._get_opt_angles(num_qubits, num_layers)
        return params

    def circuit(_params) -> list:
        """Construct a parameterized ansatz.

        Returns a list of circuits: the ansatz measured in the Z basis, and the
        ansatz measured in the X basis. The counts obtained from evaluated these
        two circuits should be passed to `score` in the same order they are
        returned here.
        """
        return VQE._gen_ansatz(_params)

    def score(counts: list, _params) -> float:
        """Compare the average energy measured by the experiments to the ideal
        value obtained via noiseless simulation. In principle the ideal value
        can be obtained through efficient classical means since the 1D TFIM
        is analytically solvable.
        """
        counts_z, counts_x = counts
        shots_z = sum(counts_z.values())
        probs_z = {bitstr: count / shots_z for bitstr, count in counts_z.items()}
        shots_x = sum(counts_x.values())
        probs_x = {bitstr: count / shots_x for bitstr, count in counts_x.items()}
        experimental_expectation = VQE._get_expectation_value_from_probs(
            Counter(probs_z),
            Counter(probs_x),
        )

        circuit_z, circuit_x = VQE.circuit(_params)
        ideal_expectation = VQE._get_expectation_value_from_probs(
            VQE.get_ideal_counts(circuit_x),
            VQE.get_ideal_counts(circuit_z),
        )

        return float(
            1.0 - abs(ideal_expectation - experimental_expectation) / abs(2 * ideal_expectation)
        )

    def get_ideal_counts(circuit: cirq.Circuit) -> Counter:
        ideal_counts = {}
        for i, amplitude in enumerate(circuit.final_state_vector(ignore_terminal_measurements=True)):
            bitstring = f"{i:>0{len(circuit.all_qubits())}b}"
            probability = np.abs(amplitude) ** 2
            ideal_counts[bitstring] = probability
        return Counter(ideal_counts)