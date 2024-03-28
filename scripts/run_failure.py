import sys
import os
import time
import pickle as pk
import pandas as pd

sys.path.insert(0, "../")

from generate_qc import compute_result_from_qc

if __name__ == "__main__":
    # num_var, num_sol, seed, sols, eqs
    eqs_df = pk.load(open("../output/eqs.pkl", "rb"))  # type: pd.DataFrame

    threshold = 0.8

    # note: change this value to compute each of the three groups
    grp = 1
    print(f"running group {grp}")

    save_file = f"../output/run_failure_group{grp}.pkl"
    cols = ["num_var", "num_sol", "seed", "level", "shots", "result"]
    result = pd.DataFrame(columns=cols)
    if os.path.exists(save_file):
        result = pk.load(open(save_file, "rb"))

    eqs_df = eqs_df[eqs_df.num_var <= 18]
    # separate the data into groups having num_sol in 1-5, 5-20, 20-90
    eqs_df["g1"] = eqs_df["num_sol"].apply(lambda x: 1 if x < 5 else 0)
    eqs_df["g2"] = eqs_df["num_sol"].apply(lambda x: 1 if 5 <= x < 20 else 0)
    eqs_df["g3"] = eqs_df["num_sol"].apply(lambda x: 1 if 20 <= x < 90 else 0)
    eqs_df["group"] = eqs_df["g1"] + eqs_df["g2"] * 2 + eqs_df["g3"] * 3
    eqs_df.drop(columns=["g1", "g2", "g3"], inplace=True)

    # from each group, keep at most 15 instances
    shrink_df = pd.DataFrame(columns=eqs_df.columns)
    for (num_var, group), df in eqs_df.groupby(["num_var", "group"]):
        # this line is replaced from dataframe.append since api changes
        shrink_df = pd.concat([shrink_df, df], axis=0)


    # subselect group in each run
    df = shrink_df[shrink_df.group == grp]
    df.sort_values(by=["num_var", "num_sol", "seed"], inplace=True)

    all_shots = [4, 16, 64, 256]

    for i, row in df.iterrows():
        for shots in all_shots:
            eqs = row["eqs"]
            num_var = row["num_var"]
            num_sol = row["num_sol"]
            seed = row["seed"]
            sol = row["sol"]
            split = 2
            level = 2
            curr_result_count = result.groupby(["num_var", "shots"]).count()
            if (num_var, shots) in curr_result_count.index:
                n_res = curr_result_count.loc[num_var, shots].result
                if n_res >= 15:
                    print(f"check {num_var} {shots} having {n_res} results")
                    continue

            if result[(result["num_var"] == num_var) &
                      (result["num_sol"] == num_sol) &
                      (result["seed"] == seed) &
                      (result["shots"] == shots)].shape[0] > 0:
                print(f"found {num_var, num_sol, seed, shots}")
                continue
            print(f"now = {time.strftime('%H:%M:%S')}, "
                  f"n = {num_var}, sol = {num_sol}")

            try:
                res = compute_result_from_qc(
                    num_var, seed, shots, split, level, num_sol,
                    solutions=sol, eqs=eqs, threshold=threshold, dry_run=False
                )
                res["Result"]["results"][0]["data"].pop("statevector")
            except TimeoutError:
                print(f"determine iteartions failed for "
                      f"{num_var, num_sol, seed, shots}")
                res = []
            print(f"now = {time.strftime('%H:%M:%S')}, "
                  f"n = {num_var}, sol = {num_sol} complete")
            print(res)

            raw = [[num_var, num_sol, seed, level, shots, res]]
            result = pd.concat([
                result, pd.DataFrame(raw, columns=cols)
            ], ignore_index=True, axis=0)

            pk.dump(result, open(save_file, "wb+"))
