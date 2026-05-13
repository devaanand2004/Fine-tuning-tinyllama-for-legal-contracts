# parse_cuad.py — LegallyBound Phase 1: Explore CUAD dataset
# Run: python parse_cuad.py

import json
from pathlib import Path

DATASET_PATH = Path("E:/your_project/CUAD_v1/CUAD_v1.json")

print("=" * 60)
print("CUAD v1 — DATASET EXPLORATION")
print("=" * 60)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    cuad = json.load(f)

# ── Top-level keys ─────────────────────────────────────────────
print(f"\nTop-level keys : {list(cuad.keys())}")

contracts = cuad["data"]
print(f"Total contracts (top-level entries) : {len(contracts)}")

# ── Count paragraphs ───────────────────────────────────────────
total_paragraphs = sum(len(c["paragraphs"]) for c in contracts)
print(f"Total paragraphs                    : {total_paragraphs}")

# ── Unique clause / question types ────────────────────────────
question_set = set()
for contract in contracts:
    for para in contract["paragraphs"]:
        for qa in para["qas"]:
            question_set.add(qa["question"])

print(f"\nUnique clause/question types : {len(question_set)}")
print("\nAll clause types:")
for i, q in enumerate(sorted(question_set), 1):
    print(f"  {i:>3}. {q}")

# ── Sample entry ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE ENTRY")
print("=" * 60)
sample_contract  = contracts[0]
sample_paragraph = sample_contract["paragraphs"][0]
sample_qa        = sample_paragraph["qas"][0]

print(f"Contract title  : {sample_contract['title']}")
print(f"Context (first 300 chars):\n  {sample_paragraph['context'][:300]}...")
print(f"\nQuestion        : {sample_qa['question']}")

if sample_qa["answers"]:
    ans = sample_qa["answers"][0]
    print(f"Answer text     : {ans['text'][:200]}")
    print(f"Answer start idx: {ans['answer_start']}")
else:
    print("Answer          : (empty — unanswerable)")

print("\nDone — no model loaded in this script.")