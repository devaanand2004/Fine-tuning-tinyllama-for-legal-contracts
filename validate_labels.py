# validate_labels.py
import pandas as pd

df = pd.read_csv("cuad_risk_labels.csv")

print("=== Basic stats ===")
print(f"Total records : {len(df)}")
print(f"Null rows     : {df.isnull().sum().sum()}")
print(f"\nRisk distribution:")
print(df["risk_level"].value_counts())
print(f"\nClass balance (%):")
print((df["risk_level"].value_counts(normalize=True) * 100).round(1))

print("\n=== Per clause type breakdown ===")
print(df.groupby(["clause_type", "risk_level"]).size().unstack(fill_value=0))

print("\n=== Sample HIGH risk entries ===")
high = df[df["risk_level"] == "HIGH"].sample(min(3, len(df[df["risk_level"] == "HIGH"])), random_state=42)
for _, row in high.iterrows():
    print(f"\n  Clause type : {row['clause_type']}")
    print(f"  Clause text : {str(row['clause_text'])[:200]}")
    print(f"  Reason      : {row['reason']}")

print("\n=== Sample LOW risk entries ===")
low = df[df["risk_level"] == "LOW"].sample(min(3, len(df[df["risk_level"] == "LOW"])), random_state=42)
for _, row in low.iterrows():
    print(f"\n  Clause type : {row['clause_type']}")
    print(f"  Clause text : {str(row['clause_text'])[:200]}")
    print(f"  Reason      : {row['reason']}")

print("\n=== Potential issues ===")
breakdown = df.groupby(["clause_type", "risk_level"]).size().unstack(fill_value=0)
for clause in breakdown.index:
    row   = breakdown.loc[clause]
    total = row.sum()
    dominant = row.max()
    if total >= 10 and dominant / total >= 0.98:
        print(f"  WARNING: '{clause}' is {row.idxmax()} {dominant/total*100:.0f}% of the time ({dominant}/{total})")

print("\nIf distribution looks reasonable, proceed to training.")
