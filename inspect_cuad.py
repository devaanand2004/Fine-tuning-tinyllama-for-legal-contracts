# inspect_cuad.py
import json
import re
from pathlib import Path
from collections import Counter

RAW_PATH = Path("CUAD_v1/CUAD_v1.json")

print("Loading CUAD_v1.json...")
with open(RAW_PATH, encoding="utf-8") as f:
    cuad_raw = json.load(f)

data = cuad_raw["data"]
print(f"Total contracts : {len(data)}")

# Flatten all QA pairs
records = []
for contract in data:
    for paragraph in contract["paragraphs"]:
        for qa in paragraph["qas"]:
            if qa.get("is_impossible", False):
                continue
            answers = qa.get("answers", [])
            if not answers:
                continue
            clause_text = answers[0]["text"].strip()
            if len(clause_text) < 30:
                continue
            match = re.search(r'"([^"]+)"', qa["question"])
            clause_type = match.group(1) if match else qa["question"][:50]
            records.append({
                "clause_type": clause_type,
                "clause_text": clause_text,
            })

print(f"Total valid records: {len(records)}")

print("\nPer clause type counts:")
counts = Counter(r["clause_type"] for r in records)
for ct, n in sorted(counts.items()):
    print(f"  {ct:<45} {n}")

print("\nSample texts:")
for ct in list(counts.keys())[:3]:
    sample = next(r for r in records if r["clause_type"] == ct)
    print(f"\n  [{ct}]")
    print(f"  {sample['clause_text'][:300]}")