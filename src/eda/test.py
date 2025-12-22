from src.eda.runner import run_eda
import pandas as pd

df = pd.DataFrame({
    "a": [1, 2, 3],
    "b": ["x", "y", "z"]
})

print(run_eda(df, target_col="a"))
