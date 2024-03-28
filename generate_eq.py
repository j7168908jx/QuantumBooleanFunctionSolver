import numpy as np
from loguru import logger
import time
from typing import List, Tuple, Iterable

from utils import eqs_to_str, sols_to_str, evaluate


EQ = np.ndarray
EQs = List[np.ndarray]
print = lambda *a, **b: None

class f:
    """A wrapper class so we can use multiprocessing to speed up brute force.
    """
    def __init__(self, eqs, n, cond):
        self.eqs = eqs
        self.n = n
        self.cond = cond

    def __call__(self, x):
        return evaluate(self.eqs, x, self.n, self.cond)


def find_solution_brute_force(
        eqs: EQs, n: int, solution_list: Iterable[int] = None,
        pool: "mp.Pool" = None
    ) -> List[int]:
    logger.info("Brute force searching...")

    if solution_list is None:
        solution_list = range(2**n)

    if pool is not None:
        eval_results = list(pool.imap(f(eqs, n, False), solution_list, 1024))
    else:
        logger.warning("multiprocessing not enabled "
                        "for brute force finding solution")
        eval_results = list(map(f(eqs, n, False), solution_list))

    solutions = [sol for sol, eval_result
                 in zip(solution_list, eval_results)
                 if eval_result]
    return solutions


def generate_eq(n: int, r: int) -> EQs:
    """Randomly choose equations from all monials.

    With `n` variables, there is a total of `1 + n + n(n-1)/2` monomials.

    Randomly choose `r` equations from all monomials, via Poisson distribution.

    """
    usable_monomial = [0] + [2 ** i for i in range(n)] + \
                      [2 ** i + 2 ** j for j in range(n) for i in range(j)]
    monomial_num = np.random.poisson(n * (n+1) / 4, size=(r, ))
    monomial_num[monomial_num >= n*(n+1)/2] = n * (n+1) / 2
    monomial_num[monomial_num == 0] = 1
    eqs = [np.random.choice(usable_monomial, size=i, replace=False)
           for i in monomial_num]
    return eqs


def generate_simple_eqs(
        n: int, num_sol_min=1, num_sol_max=None, multiprocess=True
    ) -> Tuple[List[int], EQs]:
    """Randomly generate simple equations so that there exist
        [num_sol_min, num_sol_max] solutions.

    Use multiprocessing to parallelize the brute force search if set
    `multiprocess=True`

    """
    if num_sol_max is None:
        num_sol_max = num_sol_min

    import multiprocessing as mp
    with mp.Pool(mp.cpu_count() if multiprocess else 1) as pool:
        print("="*20 + "Start of Part 1 - Generating Equations" + "=" * 20)
        eqs = generate_eq(n, n)
        print("Initially Generate", eqs_to_str(eqs, n))
        sol = find_solution_brute_force(eqs, n, pool=pool)

        while len(sol) < num_sol_min or len(sol) > num_sol_max:
            if len(sol) > num_sol_max:
                # logger.debug("Too many solutions.")
                new_eq = generate_eq(n, 1)
                eqs.insert(0, new_eq[0])

            elif len(sol) < num_sol_min:
                # logger.debug("Too few solutions. Analyze...\n\n")

                max_possible_sol_len = 0
                drop_eq_index = 0
                for i in range(len(eqs)):
                    possible_sol = find_solution_brute_force(
                        eqs[:i] + eqs[i+1:], n, pool=pool)

                    if num_sol_min <= len(possible_sol) <= num_sol_max:
                        drop_eq_index = i
                        break
                    elif len(possible_sol) > max_possible_sol_len:
                        drop_eq_index = i

                eqs.pop(drop_eq_index)

            sol = find_solution_brute_force(eqs, n, pool=pool)

    # logger.debug("With", sols_to_str(sol, n))
    print("="*20 + "End of Part 1 - Generating Equations" + "=" * 20)
    return sol, eqs


if __name__ == "__main__":
    import time
    n = 15
    st = time.time()
    for _ in range(1):
        generate_simple_eqs(n)
    print(f"n = {n} time cost: {(time.time()-st)/10:.4f}")


    # eqs = generate_eq(20, 1000)
    # result = []
    # with mp.Pool(mp.cpu_count()) as pool:
    #     for _ in trange(25):
    #         idx = np.random.choice(range(1000), 8, replace=False)
    #         part = [eqs[i] for i in idx]
    #         sol = find_solution_brute_force(part, 16, pool)
    #         result.append(sol)
    # t = [len(i) for i in result]
    # plt.hist(t)
    # plt.show()
