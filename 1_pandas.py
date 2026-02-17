import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)

print(df.head())

filtered_df = df[df['Age'] > 30]
print(filtered_df)

df['Senior'] = df['Age'] > 30
print(df)

grouped_df = df.groupby('City')['Age'].mean()
print(grouped_df)
