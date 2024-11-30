# Code for Paper "Resource Efficient Boolean Function Solver on Quantum Computer"

See [arXiv 2310.05013](https://arxiv.org/abs/2310.05013)


## Code Structure

- `.ipynb` files are the Jupyter notebooks for the figures.
- `.py` in the root directory are the main functions and classes.
- `.py` in the `scripts` folder stores the scripts for running the experiments and saving the result to the `output` folder.
  - temporary outputs are stored in `pickle` format.


## Example Equations

An example set of equations are provided in the `output` folder.
The equations are stored in the `csv` format for cross-version compatibility.

Convert to `pickle` format that suits your Python/pickle version:

```python
import pandas as pd
import numpy as np
import pickle

eqs_df = pd.read_csv("output/eqs.csv", index_col=0)
eqs_df.eqs = eqs_df.eqs.apply(lambda x: [np.array(i) for i in eval(x)])
eqs_df.sol = eqs_df.sol.apply(lambda x: eval(x))
with open("output/eqs.pkl", "wb") as f:
    pickle.dump(eqs_df, f)
```

