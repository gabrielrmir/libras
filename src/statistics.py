import pandas as pd

df = pd.read_csv('data/capture.csv', header=None)
print(df[0].value_counts())
