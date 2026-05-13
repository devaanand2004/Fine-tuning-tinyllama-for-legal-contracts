# inspect_master_clauses.py
import pandas as pd
from collections import Counter

df = pd.read_csv("CUAD_v1/master_clauses.csv")
print(f"Total contracts : {len(df)}")
print(f"Total columns   : {len(df.columns)}")

# Find all answer columns
answer_cols = [c for c in df.columns if "Answer" in c]
print(f"Answer columns  : {len(answer_cols)}")

# Count valid records per clause type
print("\nPer clause type — valid record counts:")
total = 0
for col in answer_cols:
    clause_type = col.replace("-Answer", "").replace("- Answer", "").strip()
    valid = df[col].dropna()
    valid = valid[~valid.astype(str).str.lower().isin(["no", "nan", "none", ""])]
    valid = valid[valid.astype(str).str.len() >= 30]
    count = len(valid)
    total += count
    print(f"  {clause_type:<45} {count}")

print(f"\nTotal valid records: {total}")

# Show a sample answer so we know what the text looks like
print("\nSample clause texts:")
for col in answer_cols[:3]:
    clause_type = col.replace("-Answer", "").replace("- Answer", "").strip()
    valid = df[col].dropna()
    valid = valid[~valid.astype(str).str.lower().isin(["no", "nan", "none", ""])]
    valid = valid[valid.astype(str).str.len() >= 30]
    if len(valid) > 0:
        print(f"\n  [{clause_type}]")
        print(f"  {str(valid.iloc[0])[:300]}")