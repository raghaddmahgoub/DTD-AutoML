import pandas as pd
from agents.static.eda_agent import EDAAgent
import os
from benchmark import AnalysisEvaluator


filename = 'assets/data/Classification Datasets/Titanic-Dataset.csv'
df_name = os.path.splitext(os.path.basename(filename))[0] 
df = pd.read_csv(filename)

eda = EDAAgent(
    df=df,
    target_column="Survived",
    df_name=df_name
)

report = eda.run(run_type="raw")
eda.generate_preprocessing_context()



benchmark = AnalysisEvaluator(df, report, target="Survived")
metrics = benchmark.evaluate()

import json
print(json.dumps(metrics, indent=2))