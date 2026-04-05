import pandas as pd
from agents.eda_agent.eda_agent import TargetSuggestionAgent

# Load your dataset
df = pd.read_csv('R:/GP/assets/data/Datasets/Classification Datasets/Titanic-Dataset.csv')

# Run suggestion agent
tsa = TargetSuggestionAgent(df)
result = tsa.run()

# Print nicely
print("\n🔍 TARGET SUGGESTIONS:\n")

for s in result["suggestions"]:
    print(f"Column: {s['column']}")
    print(f"Priority: {s['priority']}")
    print(f"Task: {s['task']}")

    print("Evidence:")
    for e in s["evidence"]:
        print(f"  + {e}")

    if s["exclusions"]:
        print("Exclusions:")
        for ex in s["exclusions"]:
            print(f"  - {ex}")

    print("-" * 40)

print("\nNote:", result["note"])

path = tsa.generate_target_json()
print("Saved to:", path)