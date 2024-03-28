import numpy as np
from loguru import logger


def determine_iterations_full(n: int, M=1):
    N = 2**n
    theta = np.arcsin(2 * (M*(N-M)) ** .5 / N)
    full_iterations = int(np.round(np.arccos((M/N)**.5) / theta))
    return full_iterations

def determine_minimum_iterations_random(n: int, r: int, part_size: int,
                                  threshold: float,
                                  M,
                                  shots,
                                  max_iter):
    logger.info(f"determining minimum iterations...")
    N = 2 ** n
    tilde_M = M * 2 ** (r - part_size)

    vec = np.ones(shape=(N,)) / (N**.5)
    best_it = max_iter
    for it in range(1, max_iter):
        v = np.random.choice(
            [-1, 1], size=N-M, p=[(tilde_M-M)/(N-M), (N-tilde_M)/(N-M)]
        )
        v = np.concatenate([(-1) * np.ones(shape=(M,)), v])

        vec *= v
        vec = 2 * np.mean(vec, keepdims=True) - vec
        prob = vec ** 2

        correct_prob = prob[:M].sum()
        fail_prob = (1 - correct_prob) ** shots
        logger.debug(f"iteration: {it}, correct prob: {correct_prob:.4e}, "
                        f"fail_prob : {fail_prob:.4e}")

        if 1-fail_prob > threshold:
            best_it = it
            logger.info(f"determined minimum iteration num: {best_it}")
            break

    return best_it


def determine_minimum_iterations_approx(n: int, r: int, part_size: int,
                                  threshold: float,
                                  M,
                                  shots,
                                  max_iter):
    logger.info(f"determining minimum iterations approx...")
    logger.info(f"n={n}, r={r}, m={M}, threshold={threshold}, "
                f"shots={shots}, part_size={part_size}")
    N = 2 ** n
    tilde_M = M * 2 ** (r - part_size)

    G = np.array([
        [-M, N+M-2*tilde_M],
        [-M, N+M-2*tilde_M]
    ]) * 2 / N

    G[0, 0] += 1
    G[1, 1] -= (N+M-2*tilde_M) / (N-M)

    v = np.ones(shape=(2, 1)) / (N**.5)

    best_it = max_iter
    for it in range(1, max_iter):
        v = G @ v
        correct_prob = (v[0] ** 2) * M
        fail_prob = (1 - correct_prob) ** shots
        if 1-fail_prob > threshold:
            best_it = it
            logger.info(f"determined minimum iteration num: {best_it}")
            logger.info(f"fail_prob: {fail_prob}")
            break

    return best_it

def determine_minimum_iterations_expectation(n: int, r: int, part_size: int,
                                  threshold: float,
                                  M,
                                  shots,
                                  max_iter):
    logger.info(f"determining minimum iterations...")
    N = 2 ** n
    tilde_M = M * 2 ** (r - part_size)

    vec = np.ones(shape=(N,)) / (N**.5)
    best_it = max_iter

    O = np.concatenate([np.ones((M,)) * (-1),
                        np.ones((N-M,)) * (N+M-2*tilde_M) / (N-M)])

    for it in range(1, max_iter):
        vec = O * vec
        vec = 2 * np.mean(vec, keepdims=True) - vec
        # vec /= np.linalg.norm(vec)
        prob = vec ** 2

        correct_prob = prob[:M].sum()
        fail_prob = (1 - correct_prob) ** shots
        logger.debug(f"iteration: {it}, correct prob: {correct_prob:.4e}, "
                        f"fail_prob : {fail_prob:.4e}")

        if 1-fail_prob > threshold:
            best_it = it
            logger.info(f"determined minimum iteration num: {best_it}")
            break

    return best_it

if __name__ == "__main__":
    kwargs = dict(n=20, r=19, part_size=12,
                  threshold=0.99, M=18, shots=100, max_iter=10000)

    best_it = determine_minimum_iterations_expectation(**kwargs)
    print(best_it)

    best_it = determine_minimum_iterations_random(**kwargs)
    print(best_it)

    best_it = determine_minimum_iterations_approx(**kwargs)
    print(best_it)
