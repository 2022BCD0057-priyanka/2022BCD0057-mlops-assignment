from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

df = pd.read_csv(DATA_DIR / "housing.csv")

# Version 1 (small)
df.sample(500, random_state=42).to_csv(DATA_DIR / "data_v1.csv", index=False)

# Version 2 (full)
df.to_csv(DATA_DIR / "data_v2.csv", index=False)

print("Created data_v1.csv and data_v2.csv in the data folder.")