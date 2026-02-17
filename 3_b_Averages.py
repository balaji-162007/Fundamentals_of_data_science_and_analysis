import numpy as np
import pandas as pd
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mean_pandas = pd.Series(data).mean()
print(f"Mean (Pandas): {mean_pandas}")
mean_numpy = np.mean(data)
print(f"Mean (NumPy): {mean_numpy}")
median = pd.Series(data).median()
print(f"Median: {median}")
mode = pd.Series(data).mode()
print(f"Mode: {mode}")
