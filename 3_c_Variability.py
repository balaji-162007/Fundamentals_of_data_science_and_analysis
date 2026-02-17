import numpy as np
import pandas as pd
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
range_value = max(data) - min(data)
print(f"Range: {range_value}")
variance_pandas = pd.Series(data).var()
print(f"Variance (Pandas): {variance_pandas}")
std_dev_pandas = pd.Series(data).std()
print(f"Standard Deviation (Pandas): {std_dev_pandas}")
variance_numpy = np.var(data)
print(f"Variance (NumPy): {variance_numpy}")
std_dev_numpy = np.std(data)
print(f"Standard Deviation (NumPy): {std_dev_numpy}")
