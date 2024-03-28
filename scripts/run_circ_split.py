import sys
import os
import numpy as np
import time
import pickle as pk
import pandas as pd

sys.path.insert(0, "../")

from generate_qc import compute_result_from_qc
from utils import determine_oracle_ancilla_usage

# num_var, num_sol, seed, sols, eqs
eqs_df = pk.load(open("../output/eqs.pkl", "rb"))  # type: pd.DataFrame

# save_file = "run_circ_split_clean.pkl"
save_file = "run_circ_split_dry.pkl"
cols = ["num_var", "num_sol", "seed", "level",
        "num_ancilla", "shots", "result"]

result = pd.DataFrame(columns=cols)
if os.path.exists(save_file):
    result = pk.load(open(save_file, "rb"))

eqs_df = eqs_df[eqs_df.num_var > 15]
level = 2
max_split = 2
shots = 100
dry_run = True

for idx, df in eqs_df.groupby("num_var"):
    for i, row in df.iterrows():
        eqs = row["eqs"]
        num_var = row["num_var"]
        num_sol = row["num_sol"]
        seed = row["seed"]
        sol = row["sol"]
        minimum_ancilla = determine_oracle_ancilla_usage(
            int(np.ceil(len(eqs)/max_split)), level=level
        )
        maximum_ancilla = determine_oracle_ancilla_usage(
            len(eqs), level=level
        )

        for num_ancilla in range(minimum_ancilla, maximum_ancilla+1):
            split=-1
            if result[(result["num_var"] == num_var) &
                      (result["num_sol"] == num_sol) &
                      (result["seed"] == seed) &
                      (result["level"] == level) &
                      (result["num_ancilla"] == num_ancilla)].shape[0] > 0:
                print(f"found {num_var, num_sol, seed, level, num_ancilla}")
                continue

            print(f"now = {time.strftime('%H:%M:%S')}, n = {num_var}, "
                  f"sol = {num_sol}, split = {split}")

            res = compute_result_from_qc(
                num_var, seed, shots, split, level, num_sol,
                use_ancilla=num_ancilla, solutions=sol, eqs=eqs,
                dry_run=dry_run)

            print(f"now = {time.strftime('%H:%M:%S')}, n = {num_var}, "
                  f"sol = {num_sol}, split = {split} complete")
            print(res)
            res["Result"]["results"][0]["data"].pop("statevector")

            # this line is replaced from dataframe.append since api changes
            raw = [[num_var, num_sol, seed, level, num_ancilla, shots, res]]
            result = pd.concat([
                    result,
                    pd.DataFrame(raw, columns=cols)
                ], ignore_index=True, axis=0)
            pk.dump(result, open(save_file, "wb+"))
