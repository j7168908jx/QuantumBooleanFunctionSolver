import sys
import time
import pickle as pk
import os
import numpy as np
import pandas as pd

sys.path.insert(0, "../")

from generate_eq import generate_simple_eqs

cols = ["num_var", "num_sol", "seed", "sol", "eqs"]
out_data = pd.DataFrame(columns=cols)
filename = "../output/eqs.pkl"
if os.path.exists(filename):
    out_data = pk.load(open(filename, "rb"))

num_vars = [10, 12, 15, 18, 20, 22, 25]
num_sols = [1, 2, 4, 8, 16, 32, 64]
seeds = iter(np.random.randint(0, 100000, 10000))

for num_var in num_vars:
    for num_sol in num_sols:
        already = out_data[(out_data["num_var"] == num_var) &
                           (out_data["num_sol"] >= num_sol) &
                           (out_data["num_sol"] <= num_sol*1.4)]

        print(f"found {len(already)} eqs for n = {num_var}, sol = {num_sol}")

        for iter_times in range(30-len(already)):
            print(f"now = {time.strftime('%H:%M:%S')}, n = {num_var}, "
                  f"sol_min = {num_sol}, sol_max = {int(num_sol * 1.4)}")
            seed = next(seeds)
            np.random.seed(seed)

            sol, eqs = generate_simple_eqs(num_var, num_sol_min=num_sol,
                                           num_sol_max=int(num_sol*1.4))

            # this line is replaced from dataframe.append since api changes
            out_data = pd.concat(
                [
                    out_data,
                    pd.DataFrame([[num_var, len(sol), seed, sol, eqs]],
                                columns=cols)
                ],
                ignore_index=True, axis=0
            )
            pk.dump(out_data, open(filename, "wb+"))
