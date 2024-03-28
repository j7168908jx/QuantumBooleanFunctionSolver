"""
This module provides functions to generate quantum circuits
for Grover's algorithm.
"""

from qiskit import QuantumCircuit
from qiskit import QuantumRegister, AncillaRegister
from qiskit.circuit.gate import Gate

from loguru import logger
from numbers import Integral
from typing import Callable, List, Optional

import numpy as np

from generate_eq import EQ, EQs
from StoreOpQuantumCircuit import StoreOpQuantumCircuit
from utils import determine_maximum_allowed_eqs, determine_oracle_ancilla_usage


def polynomial_gate(eq: EQ, n: int, label: str) -> Gate:
    """convert an `n`-var equation to a `n+1` quantum gate

    such that when input state |x> satisfies eq(x) = 0, the ancilla outputs |1>
    (applying X gate to the ancilla)

    `label` is used to annotate the gate
    """
    qc = QuantumCircuit(n+1)
    op = n
    for item in eq:
        if isinstance(item, Integral):
            control = [
                i for i, s in enumerate(f"{bin(item)[2:]:0>{n}s}")
                if s == "1"
            ]
            if len(control) > 0:
                qc.mcx(control, target_qubit=op, mode='noancilla')
            else:
                qc.x(op)
        else:
            raise ValueError(f"Unknown item {item} in eq {eq}")
    qc.x(n)
    # # for debugging
    # if len(eq) > 0:
    #     plt.figure()
    #     ax = plt.gca()
    #     ax.set_title(f"gate: {label}")
    #     qc.draw("mpl", ax=ax)
    #     plt.title("eq")
    #     plt.show()
    return qc.to_gate(label=label)

def gen_householder_circuit(n: int) -> QuantumCircuit:
    """Create a `n` qubit circuit for the householder part"""
    householder_op = QuantumCircuit(QuantumRegister(n))

    householder_op.barrier()
    for i in range(n):
        householder_op.h(i)
        householder_op.x(i)
    householder_op.h(n-1)
    householder_op.mcx(list(range(n-1)), n-1, ancilla_qubits=None,
                        mode='noancilla')
    householder_op.h(n-1)
    for i in range(n):
        householder_op.x(i)
        householder_op.h(i)
    householder_op.barrier()
    return householder_op

def gen_state_preparation_circuit(n: int) -> QuantumCircuit:
    """An `n`-qubit circuit that applys Hadamard gate to all `n` qubits
    """
    state_prepare = QuantumCircuit(QuantumRegister(n))

    for i in range(n):
        state_prepare.h(i)
    state_prepare.barrier()
    return state_prepare

def _gen_op_single_job(args) -> QuantumCircuit:
    """Unparse the arguments and creates a Grover operator (one iteration)
    """
    eq_idx, part_name, obj_qubits, gen_oracle, n = args
    oc = gen_oracle(eq_idx)
    householder_op = gen_householder_circuit(n)
    op = oc.compose(householder_op, obj_qubits, inplace=False)
    op.name = f"G{eq_idx}"
    logger.debug(f"part {part_name} using eqs: {eq_idx}")
    return op

def gen_random_grover_op(n: int, r: int,
                         gen_oracle: Callable[[int], QuantumCircuit],
                         part_size: int,
                         obj_qubits: List[int],
                         iters: int
                         ) -> List[QuantumCircuit]:
    """gen a random Grover operator for `n` vars and `r` equations

    Args:
        n (int): number of variables/qubits
        r (int): number of equations
        gen_oracle (callable): callable functions   # todo
        part_size (int): how many eqs each random part should contain
        obj_qubits (List[int]): indices of target qubits
        iters (int): number of Grover iterations

    """

    ops = []
    idx = np.arange(r)

    inner_iterations = int(np.ceil(r/part_size))
    outer_iterations = int(np.ceil(iters/inner_iterations))
    parallel_args = []
    for m in range(outer_iterations):
        np.random.shuffle(idx)
        logger.debug(f"shuffled index seq: {idx.tolist()}")

        # 0, 3, 6, 10 for part_size=5, r=15, 4 inner iter
        all_st = np.linspace(0, r-part_size, inner_iterations, dtype=int)
        for k, st in enumerate(all_st):
            if len(parallel_args) < iters:
                used_eq_idx = idx[st:st+part_size]
                parallel_args.append((
                    list(used_eq_idx), f"{m}-{k}", obj_qubits, gen_oracle, n
                ))

    # note: may use parallel library to generate operators
    # import multiprocessing as mp
    # with mp.Pool() as pool:
    ops = list(map(_gen_op_single_job, parallel_args))

    return ops


def generate_oracle(
    eqs: EQs, n: int, level: int = 3, arrange=True) -> QuantumCircuit:
    """generate the oracle circuit for the given equations.

    This function is parsed and called and `OracleGenerator`.

    It recursively call a sub-function to generate the oracle circuit,
    level by level.

    Args:
        eqs (EQs): list of equations
        n (int): number of variables/qubits
        level (int): recursive level of the oracle
        arrange (bool): whether to use aggresive algorithm to
            rearrange the gates to reduce the depth

    """

    x = list(range(n))

    def v(qc, level_now: int, avail_ancilla: List[int],
          target_ancilla: Optional[int],
          gate_index: int):

        if gate_index >= len(eqs):
            return gate_index

        qc.barrier()

        ancilla_num = len(avail_ancilla)
        if ancilla_num in [0, 1]:
            if target_ancilla:
                qc.append(gates[gate_index], x + [target_ancilla])
                gate_index += 1
            else:
                # special case for only 1 eq
                qc.append(gates[gate_index], x + [target_ancilla])
                qc.p(np.pi, target_ancilla)
                qc.append(gates[gate_index], x + [target_ancilla])
                gate_index += 1
            return gate_index

        level_now = min(ancilla_num-1, level_now)

        if level_now <= 1:
            applied_gates = []
            for anc in avail_ancilla:
                qc.append(gates[gate_index], x + [anc])
                applied_gates.append([gates[gate_index], x + [anc]])
                gate_index += 1
            if target_ancilla:
                qc.mcx(avail_ancilla, target_ancilla)
            else:
                qc.mcp(np.pi, avail_ancilla[:-1], avail_ancilla[-1])
            for rec in applied_gates[::-1]:
                qc.append(*rec)
        else:
            applied_circuits = []
            actually_used_ancilla = []
            for e, target_anc in enumerate(avail_ancilla[::-1]):
                kwargs = dict(
                    qc=qc, level_now=level_now-1,
                    avail_ancilla=avail_ancilla[:-e-1],
                    target_ancilla=target_anc,
                    gate_index=gate_index)
                t = v(**kwargs)
                if t > gate_index:
                    actually_used_ancilla.append(target_anc)
                gate_index = t
                applied_circuits.append(kwargs)
            if target_ancilla:
                qc.mcx(actually_used_ancilla, target_ancilla)
            else:
                if len(actually_used_ancilla) > 1:
                    qc.mcp(np.pi, actually_used_ancilla[:-1],
                           actually_used_ancilla[-1])
                else:
                    qc.p(np.pi, actually_used_ancilla[-1])
            for cir in applied_circuits[::-1]:
                v(**cir)

        return gate_index

    qr = QuantumRegister(n, name="x")
    ancilla_usage = determine_oracle_ancilla_usage(len(eqs), level=level)
    ancilla = AncillaRegister(ancilla_usage, name="oracleWorker")
    maximum_allowed_eqs = determine_maximum_allowed_eqs(ancilla_usage, level)

    gates = [
        polynomial_gate(eq, n, label=f"eq{i + 1}")
        for i, eq in enumerate(eqs)
        ] + [
        polynomial_gate(np.empty((0,)), n, label=f"φeq{i + 1}")
        for i in range(len(eqs), maximum_allowed_eqs)]

    if arrange:
        qc = StoreOpQuantumCircuit(qr, ancilla)
        v(qc, level_now=level, avail_ancilla=list(range(n, n + ancilla_usage)),
          target_ancilla=None, gate_index=0)
        qc.barrier()
        qc = qc.qc
    else:
        qc = QuantumCircuit(qr, ancilla)
        v(qc, level_now=level, avail_ancilla=list(range(n, n + ancilla_usage)),
          target_ancilla=None, gate_index=0)
        qc.barrier()

    # # for debugging
    # plt.figure()
    # ax = plt.gca()
    # qc.draw(output="mpl", ax=ax)
    # plt.show()
    #
    # print(f"totally decomposed oracle gate count: "
    #       f"{qc.decompose().decompose()
    #           .decompose().decompose().decompose().count_ops()}")
    return qc
