from qiskit import assemble, transpile, QuantumCircuit
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram

from qiskit import Aer
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.basis.unroller import Unroller

import pickle
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict, Union

EQ = np.ndarray
EQs = List[np.ndarray]


# ==================== Utility Functions ====================

def evaluate(eqs: EQs, x: int, n: int, full: bool) -> Union[List[int], bool]:
    """Evaluate results of eqs(x).
    monomial evaluation:
        x:                         01101 (x2,x3,x5=1)
        monomial:                  10100 (x1 * x3)
        not monomial:              01011 (2**(n+1) - 1 - monomial)
        x OR (not monomial):       01111
        result: above+1==2**(n+1): 0 (32 != 64)

    polynomial evaluation:
        monomial results:       011001001001
        sum and % 2:           5 % 2 = 1

    return full right-hand-side when `full=True`,
    otherwise return whether all right-hand-side is 0
    """
    r = []
    for eq in eqs:
        not_eq = 2**(n+1) - 1 - eq
        result = np.bitwise_or(x, not_eq) + 1 == 2 ** (n + 1)
        result = np.sum(result)
        if not full and result % 2 == 1:
            return False
        r.append(result % 2)

    return True if not full else r


def see_what_is_happening_here(qc: QuantumCircuit, init_x: str, init_h: str,
                               measure_num=-1,
                               qasm_noise=True):
    """Function for debugging a quantum circuit.
    With debug mode, set breakpoint at anywhere,
    print the counts(frequency) of current quantum state.
    """
    qc = qc.copy()
    qc2 = QuantumCircuit(QuantumRegister(len(init_x)))
    for i, s in enumerate(init_x):
        if s == "1":
            qc2.x(i)
    for i, s in enumerate(init_h):
        if s == "1":
            qc2.h(i)
    qc2.barrier()
    qc = qc.compose(qc2, front=True)
    if measure_num == -1:
        qc.measure_all()
    else:
        qc.add_register(ClassicalRegister(measure_num))
        qc.measure(list(range(measure_num)), list(range(measure_num)))
    qc.draw("mpl", initial_state=True)
    plt.title("Current Test QC")
    plt.show()

    qasm_sim = Aer.get_backend("statevector_simulator")
    transpiled = transpile(qc, qasm_sim)
    qobj = assemble(transpiled)
    results = qasm_sim.run(qobj, shots=1000).result()
    #
    counts = results.get_counts()
    print(counts)
    plot_histogram(counts, figsize=(7, 7))
    plt.show()


def _memorize(f, max_size=10000):
    """A general helper function. Memorize computed function results.
    """
    memo = {}

    def helper(*args, **kwargs):
        if len(memo) > max_size:
            memo.popitem()

        try:
            ha = hash(pickle.dumps([args, kwargs]))
        except AttributeError:
            result = f(*args, **kwargs)
        else:
            if ha not in memo:
                memo[ha] = result = f(*args, **kwargs)
            else:
                return memo[ha]
        return result
    return helper


@_memorize
def determine_oracle_ancilla_usage(eq_num: int, level: int) -> int:
    """For given num of eqs and level, determine minimum ancilla usage.

    For level=2: r * (r-1) / 2 + 1 = eq_num
    Generally:
      f_{l+1}(r) = f_l(r-1) + ... + f_l(1) + f_l(0), f_l(0) = f_l(1) = 1
      f_1(r) = r, f_1(0) = 1

    """
    if eq_num == 0:
        raise ValueError
    f_l = list(range(eq_num + 1))
    f_l[0] = 1
    for _ in range(level-1):
        for j in range(1, eq_num + 1)[::-1]:
            f_l[j] = sum(f_l[:j])

    for e, i in enumerate(f_l):
        if i >= eq_num:
            return e or 1  # avoid the case e == 0
    raise ValueError


@_memorize
def determine_maximum_allowed_eqs(ancilla_num: int, level: int) -> int:
    """For given num of ancilla and level, determine allowed num of eqs.
    """
    f_l = list(range(ancilla_num + 1))
    f_l[0] = 1
    for _ in range(level-1):
        for j in range(1, ancilla_num + 1)[::-1]:
            f_l[j] = sum(f_l[:j])

    return f_l[-1]


def is_good_state_reverse(eqs, x, n):
    """A simple function validate if x is solution of eqs.
    """
    return evaluate(eqs, int(x[::-1][:n], 2), n, False)


def count_max(r):
    """Determine which index has the maximum value in r.
    """
    return np.bincount(r).argmax()


def compute_depth(qc):
    from qiskit_aer import StatevectorSimulator
    dag = circuit_to_dag(qc)
    unroller = Unroller(StatevectorSimulator().configuration().basis_gates)
    depth = unroller.run(dag).depth()
    return depth


def eqs_to_str(eqs: EQs, n: int) -> str:
    usable_monomial = [0] + [2 ** i for i in range(n)] + \
                      [2 ** i + 2 ** j for j in range(n) for i in range(j)]
    monomial_name_map = {i: f"{bin(i)[2:]:0>{n}s}" for i in
                         usable_monomial}  # type: Dict[int, str]

    s = "Equations: \n"
    for eq in eqs:
        row = ""
        for monomial in sorted(eq, reverse=True):
            str_represent = monomial_name_map[monomial]
            index = [i+1 for i in range(n) if str_represent[i] == "1"]
            if len(index) > 1:
                row += "".join([f"x{i: <{len(str(n))}d}" for i in index])
                row += " + "
            elif len(index) == 1:
                row += "".join([f"x{i: <{len(str(n))}d}" for i in index])
                row += " " * (len(str(n))+1) + " + "
            else:
                row += "1 + "
        s += row[:-2] + "= 0\n"
    return s


def sols_to_str(xs: List[int], n: int) -> str:
    if len(xs) == 0:
        return "Solution: None"
    s = "Solution: \n"
    for x in xs:
        xx = f"{bin(x)[2:]:0>{n}s}"
        s += "".join([f"x{i+1} = {r}, " for i, r in enumerate(xx)]) + "\n"
    return s
