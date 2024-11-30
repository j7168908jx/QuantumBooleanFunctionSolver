from qiskit import transpile, QuantumCircuit
from qiskit import ClassicalRegister, QuantumRegister, AncillaRegister
from qiskit_aer import StatevectorSimulator
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.basis.unroller import Unroller

from loguru import logger
import time
import numpy as np
from typing import List, Tuple, Dict

from utils import see_what_is_happening_here, is_good_state_reverse, \
    determine_maximum_allowed_eqs, determine_oracle_ancilla_usage
from determine_iterations import *
from circuit_component import gen_state_preparation_circuit, \
    gen_random_grover_op, generate_oracle


simulator = StatevectorSimulator()
if "GPU" in simulator.available_devices():
    logger.warning("**USING GPU SIMULATOR**")
    simulator.set_options(device="GPU")
else:
    logger.warning("**USING CPU SIMULATOR**")

# ==================== Grover Evaluation ====================


def grover_circuit(eqs,
                   n: int,
                   level: int,
                   num_sols: int,
                   shots: int,
                   split: float,
                   num_ancilla=None,
                   threshold=0.999,
                   iterations=None,
                   arrange=True,
                   ) -> Tuple[QuantumCircuit, int]:
    r = len(eqs)
    obj_qubits = list(range(n))

    test_qc = QuantumCircuit(QuantumRegister(n), ClassicalRegister(n))
    test_qc.compose(gen_state_preparation_circuit(n), qubits=obj_qubits,
                    inplace=True, front=True)


    if num_ancilla is None:
        part_size = int(np.ceil(r / split))
        max_ancilla = determine_oracle_ancilla_usage(part_size, level=level)
        logger.info(f"using split = {split}.")
    else:
        max_ancilla = num_ancilla
        logger.info(f"using ancilla = {num_ancilla}.")

    part_size = determine_maximum_allowed_eqs(max_ancilla, level=level)
    part_size = min(part_size, r)
    logger.info(f"determined ancilla usage: {max_ancilla} qubits."
                f"Each part use {part_size} eqs.")

    if iterations is None:
        iterations = determine_minimum_iterations_approx(
            n=n, r=r, part_size=part_size,
            threshold=threshold, M=num_sols, shots=shots, max_iter=(3*n)**2
        )
        if iterations == (3*n)**2:
            raise TimeoutError("determine_minimum_iterations_approx failed!")
        logger.info(f"determined iteration approx: {iterations}.")
    else:
        logger.warning(f"using predetermined iteration: {iterations}")

    oracle_generator = OracleGenerator(eqs, n, level, arrange=arrange)
    grover_ops = gen_random_grover_op(
        n=n, r=r, part_size=part_size,
        obj_qubits=obj_qubits, iters=iterations, gen_oracle=oracle_generator
    )
    logger.info(f"actual grover ops count: {len(grover_ops)}")

    if max_ancilla > 0:
        test_qc.add_register(AncillaRegister(max_ancilla))

    for grover_op in grover_ops:
        test_qc.compose(grover_op, inplace=True)

    test_qc.measure(obj_qubits, list(range(n)))

    # # for debugging
    # grover_ops[0].draw("mpl", fold=100)
    # test_qc.draw("mpl", fold=100)
    # plt.savefig("1.png")

    return test_qc, len(grover_ops)

def run_circuit(test_qc: QuantumCircuit, shots: int, dry_run: bool):
    logger.info("transpile start")
    if dry_run:
        qc = QuantumCircuit(test_qc.num_qubits, test_qc.num_clbits)
        transpiled = transpile(qc, simulator)
        logger.warning("Running an empty circuit in dry run mode.")
    else:
        transpiled = transpile(test_qc, simulator)
    logger.info("transpile complete")
    logger.info("running simulator...")
    results_future = simulator.run(transpiled, shots=shots)
    results = results_future.result().to_dict()
    logger.info("run simulator complete")

    for r in results["results"]:
        if "data" in r and "counts" in r["data"]:
            counts = r["data"]["counts"]
            reversed_counts = {}  # type: Dict[str, List[str]]
            for key, value in counts.items():
                reversed_counts.setdefault(value, []).append(key)
            r["data"]["counts"] = reversed_counts

    return results


class OracleGenerator:
    """wrapper for generate_oracle. Parse eqs and call generate_oracle.
    """
    def __init__(self, eqs, n, level, arrange) -> None:
        self._eqs = eqs
        self._n = n
        self._level = level
        self._arrange = arrange

    def __call__(self, eqs_index) -> QuantumCircuit:
        return generate_oracle(
            eqs=[self._eqs[j] for j in eqs_index],
            n=self._n, level=self._level, arrange=self._arrange)


def compute_result_from_qc(n, seed: int, shots: int,
                           split: float, level: int,
                           num_sols: int,
                           arrange=True,
                           multiprocess=True,
                           use_ancilla=None,
                           solutions=None,
                           eqs=None,
                           threshold=0.999,
                           iterations=None,
                           dry_run=False
                           ):
    """Entrance wrapper. Warp and check parameters.


    Args:
        n (int): num of variables
        seed (int): random seed
        shots (int): num of shots when measuring qc
        split (float): how many parts to split the equations
        level (int): recursive level \ell
        num_sols (int): intended num of solutions
        arrange (bool): whether to use aggresive algorithm to
            rearrange the gates to reduce the depth
        multiprocess (bool): whether to use multiprocessing to
            speed up brute force search for equations solutions
        use_ancilla (Optional[int]): num of ancilla qubits,
            leave `None` to deduce from other parameters
        solutions (Optional[EQs]): brute force solutions,
            if both this and eqs are not provided, will generate new eqs
        threshold (float):
            threshold used in determine
            the number of iterations of Grover's algorithm
        iterations (Optional[int]):
            if set, will force using this value as the number of iterations.
        dry_run (bool):
            whether to really run the circuit or not

    """

    np.random.seed(seed)

    st_bfs = time.time()
    if eqs is None or solutions is None:
        print("generating new eqs...")
        from generate_eq import generate_simple_eqs
        brute_force_result, eqs = generate_simple_eqs(
            n, num_sol_min=num_sols,
            multiprocess=multiprocess)
    else:
        brute_force_result = solutions
    ed_bfs = time.time()

    st_exc = time.time()
    qc, iterations = grover_circuit(
        eqs=eqs, n=n, threshold=threshold,
        level=level, shots=shots, split=split, iterations=iterations,
        num_ancilla=use_ancilla, num_sols=num_sols, arrange=arrange
    )
    result = run_circuit(qc, shots, dry_run=dry_run)
    ed_exc = time.time()
    return post_process_result(
        qc, result, arrange=arrange, n=n, multiprocess=multiprocess,
        split=split, level=level, eqs=eqs, exc_time=ed_exc-st_exc,
        bfs_time=ed_bfs-st_bfs, bfs_result=brute_force_result,
        iterations=iterations, dry_run=dry_run
    )

def post_process_result(
    qc, result, arrange, n, multiprocess, eqs, iterations, split, level,
    exc_time=None, bfs_time=None, bfs_result=None, dry_run=False
    ):
    counts = result["results"][0]["data"]["counts"]
    top_measurement = sorted(
        counts.items(), key=lambda x: x[0], reverse=True)[0][1][0]
    top_measurement = np.binary_repr(int(top_measurement, 16), n)
    success = is_good_state_reverse(eqs=eqs, x=top_measurement, n=n)

    if not success and not dry_run:
        logger.warning(f"Grover failed! n={n}, split={split}, level={level}, "
                        f"iterations={iterations}")
    if dry_run:
        logger.warning(f"Grover dry run mode! n={n}, split={split}, "
                        f"level={level}, iterations={iterations}")

    dag = circuit_to_dag(qc)
    unroller = Unroller(simulator.configuration().basis_gates)
    unrolled_circ = unroller.run(dag)
    res = {
        "Result": result,
        "num of iterating G": iterations,
        "success": success,
        "top measurement": top_measurement[::-1],
        "brute force result":
            bfs_result and [np.binary_repr(rr, n) for rr in bfs_result],
        "brute force time": bfs_time,
        "depth of qc": unrolled_circ.depth(),
        "gate count of qc": unrolled_circ.count_ops(),
        "total execute time": exc_time,
        "num of variables": n,
        "num of ancillas": qc.num_ancillas,
        "arrange": arrange,
        "multiprocess": multiprocess,
        "split": split,
        "level": level,
    }
    return res


if __name__ == "__main__":
    try:
        from pprint import pprint
    except ImportError:
        pprint = lambda *args, **kwargs: print(*args)

    printer_options = dict(compact=True, width=150)

    import pickle
    data = pickle.load(open("output/eqs.pkl", "rb"))
    dd = data[data.num_var==15][data.num_sol==1].iloc[0]

    num_vars = 15
    level, use_ancilla, split = 2, None, 1.3
    pprint(f"⏰: {time.strftime('%H:%M:%S')} now "
           f"level: {level}, use_ancilla: {use_ancilla}, split: {split}")
    r = compute_result_from_qc(n=num_vars, seed=42, shots=100,
                                split=split,
                                level=level,
                                arrange=True,
                                use_ancilla=None,
                                num_sols=1,
                                solutions=dd.sol,
                                eqs=dd.eqs
                                )
    pprint(f"⏰: {time.strftime('%H:%M:%S')}")
    pprint([num_vars, r["level"], use_ancilla, r["num of iterating G"],
            r["depth of qc"], r])

