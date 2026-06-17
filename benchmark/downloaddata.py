import openml
import pandas as pd

dataset = openml.datasets.get_dataset(43257)

X, y, _, _ = dataset.get_data()

df = pd.concat([X, y], axis=1)
df.to_csv("dataset.csv", index=False)