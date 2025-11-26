import pandas as pd
import glob
import os

# CHANGE THIS to your folder path
DATA_FOLDER = r"./"

# Load one CSV (the first one found)
files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
if not files:
    raise FileNotFoundError("No CSV files found in folder!")

print(f"Found {len(files)} CSV files. Previewing: {files[0]}\n")

df = pd.read_csv(files[0])

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== COLUMN NAMES =====")
print(list(df.columns))

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== CHECK LABEL-LIKE COLUMNS =====")
label_candidates = [c for c in df.columns if "label" in c.lower() or "class" in c.lower() or "target" in c.lower()]
print(label_candidates)

print("\n===== SAMPLE UNIQUE VALUES (for small columns) =====")
for col in df.columns:
    if df[col].dtype == "object" or df[col].nunique() < 20:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())
