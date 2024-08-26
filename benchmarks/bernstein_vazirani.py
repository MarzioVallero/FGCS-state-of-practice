# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from .benchmark import Benchmark
from random import randint
from _utils import Gate


class BernsteinVazirani(Benchmark):
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
        qubit_list = range(nqubits-1)
        circuit = []
        secret = ''.join(str(randint(0, 1)) for _ in range(nqubits-1))

        def oracle(secret, circuit):            
            for i in range(nqubits-1):
                if secret[::-1][i] == '1':
                    circuit.append(Gate(id='cnot', controls=i, targets=nqubits-1) )

            return circuit

        for q in qubit_list:
            circuit.append(Gate(id='h', targets=q))
        circuit.append(Gate(id='x', targets=nqubits-1))
        circuit.append(Gate(id='h', targets=nqubits-1))

        circuit = oracle(secret, circuit)

        for q in qubit_list:
            circuit.append(Gate(id='h', targets=q))

        circuit.append(Gate(id='measure', targets=list(qubit_list)))

        return circuit