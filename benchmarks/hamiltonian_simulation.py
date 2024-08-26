# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark
import cirq
import numpy as np
from _utils import Gate


class HamiltonianSimulation(Benchmark):
    """Quantum benchmark focused on the ability to simulate 1D
    Transverse Field Ising Models (TFIM) of variable length.

    Device performance is based on how closely the experimentally obtained
    average magnetization (along the Z-axis) matches the noiseless value.
    Since the 1D TFIM is efficiently simulatable with classical algorithms,
    computing the noiseless average magnetization remains scalable over a large
    range of benchmark sizes.
    """

    @staticmethod
    def generateGatesSequence(nqubits, config):
        """Args:
        num_qubits: int
            Size of the TFIM chain, equivalent to the number of qubits.
        time_step: int
            Size of the timestep in attoseconds.
        total_time:
            Total simulation time of the TFIM chain in attoseconds.
        """
        circuit = []
        num_qubits = nqubits
        time_step = config["time_step"]
        total_time = config["total_time"]

        circuit = HamiltonianSimulation.circuit(num_qubits, time_step, total_time)

        return circuit

    def circuit(num_qubits, time_step, total_time) -> cirq.Circuit:
        """Generate a circuit to simulate the evolution of an n-qubit TFIM
        chain under the Hamiltonian:

        H(t) = - Jz * sum_{i=1}^{n-1}(sigma_{z}^{i} * sigma_{z}^{i+1})
               - e_ph * cos(w_ph * t) * sum_{i=1}^{n}(sigma_{x}^{i})

        where,
            w_ph: frequency of E" phonon in MoSe2.
            e_ph: strength of electron-phonon coupling.
        """
        hbar = 0.658212  # eV*fs
        jz = (
            hbar * np.pi / 4
        )  # eV, coupling coeff; Jz<0 is antiferromagnetic, Jz>0 is ferromagnetic
        freq = 0.0048  # 1/fs, frequency of MoSe2 phonon

        w_ph = 2 * np.pi * freq
        e_ph = 3 * np.pi * hbar / (8 * np.cos(np.pi * freq))

        qubits = range(num_qubits)
        circuit = []

        # Build up the circuit over total_time / time_step propagation steps
        for step in range(int(total_time / time_step)):
            # Simulate the Hamiltonian term-by-term
            t = (step + 0.5) * time_step

            # Single qubit terms
            psi = -2.0 * e_ph * np.cos(w_ph * t) * time_step / hbar
            for qubit in qubits:
                circuit.append(Gate(id='h', targets=qubit))
                circuit.append(Gate(id='rz', params=psi, targets=qubit))
                circuit.append(Gate(id='h', targets=qubit))

            # Coupling terms
            psi2 = -2.0 * jz * time_step / hbar
            for i in range(len(qubits) - 1):
                circuit.append(Gate(id='cnot', controls=qubits[i], targets=qubits[i + 1]))
                circuit.append(Gate(id='rz', params=psi2, targets=qubits[i + 1]))
                circuit.append(Gate(id='cnot', controls=qubits[i], targets=qubits[i + 1]))

        # End the circuit with measurements of every qubit in the Z-basis
        circuit.append(Gate(id='measure', targets=list(range(num_qubits))))

        return circuit
