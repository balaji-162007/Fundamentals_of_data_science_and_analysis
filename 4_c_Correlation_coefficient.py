import numpy as np
import pandas as pd
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)
correlation_numpy = np.corrcoef(x, y)[0, 1]
print(f"Correlation Coefficient (NumPy): {correlation_numpy}")
df = pd.DataFrame({'x': x, 'y': y})
correlation_pandas = df.corr().loc['x', 'y']
print(f"Correlation Coefficient (Pandas): {correlation_pandas}")
