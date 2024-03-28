from qiskit import QuantumCircuit

from rearrange import rearrange


class StoreOpQuantumCircuit:
    """Helper Class to rearrange gate orders.

    If rearrange parameter is set to True, this class overloads original qc
    so when gates are appended to qc, they are first stored but not inserted,
    and finally reordered and inserted right before a MCX gate occurs.
    """

    def __init__(self, *args, **kwargs):
        self.qc = QuantumCircuit(*args, **kwargs)
        self.ops = []

    def add_all_history_instructions(self):
        works = [i["qargs"].copy() for i in self.ops]
        m = self.qc.num_ancillas
        n = self.qc.num_qubits - m
        order = rearrange(n, m, works)
        self.ops = [self.ops[i] for i in order]
        for op in self.ops:
            self.qc.append(**op)
        self.ops = []

    def __getattr__(self, name):
        def func(*args, **kwargs):
            if name == "append":
                gate, qubit_map = args
                ops = gate.definition.data
                for gate, qargs, cargs in ops:
                    self.ops.append({
                        "instruction": gate,
                        "qargs": [qubit_map[i.index] for i in qargs],
                        "cargs": cargs
                    })
            else:
                self.add_all_history_instructions()
                self.qc.__getattribute__(name)(*args, **kwargs)

        return func
