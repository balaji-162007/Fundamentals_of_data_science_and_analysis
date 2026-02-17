import pandas as pd
data = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7, 7]
series = pd.Series(data)
frequency = series.value_counts()
print(frequency)
