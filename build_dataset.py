# build_dataset.py — LegallyBound Phase 1: Build train/eval JSONL
# Run: python build_dataset.py

import json
import random
from pathlib import Path

DATASET_PATH = Path("E:/your_project/CUAD_v1/CUAD_v1.json")
DATA_DIR     = Path("data")
CONTEXT_LIMIT = 1800   # chars ≈ 450 tokens

random.seed(42)

# ── Load ───────────────────────────────────────────────────────
print("Loading CUAD_v1.json …")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    cuad = json.load(f)

contracts = cuad["data"]

# ── Extract samples ────────────────────────────────────────────
samples = []
for contract in contracts:
    contract_name = contract["title"]
    for para in contract["paragraphs"]:
        context_chunk = para["context"][:CONTEXT_LIMIT]
        for qa in para["qas"]:
            # Skip if no answer or answer is whitespace
            if not qa["answers"]:
                continue
            answer_text = qa["answers"][0]["text"].strip()
            if not answer_text:
                continue

            clause_type = qa["question"]

            formatted = (
                "<|system|>\n"
                "You are a legal contract analyst. Extract and "
                "summarize the requested clause clearly and "
                "concisely.\n</s>\n"
                "<|user|>\n"
                f"Contract excerpt:\n{context_chunk}\n\n"
                f"Question: {clause_type}\n</s>\n"
                "<|assistant|>\n"
                f"{answer_text}\n</s>"
            )

            samples.append({
                "text":          formatted,
                "contract_name": contract_name,
                "clause_type":   clause_type,
                "context":       context_chunk,
                "answer":        answer_text,
            })

# ── Shuffle & split ────────────────────────────────────────────
random.shuffle(samples)

split_idx   = int(len(samples) * 0.9)
train_data  = samples[:split_idx]
eval_data   = samples[split_idx:]

unique_clauses = len({s["clause_type"] for s in samples})
avg_len = sum(len(s["text"]) for s in samples) / len(samples)

# ── Save ───────────────────────────────────────────────────────
DATA_DIR.mkdir(exist_ok=True)

train_path = DATA_DIR / "train.jsonl"
eval_path  = DATA_DIR / "eval.jsonl"

with open(train_path, "w", encoding="utf-8") as f:
    for s in train_data:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open(eval_path, "w", encoding="utf-8") as f:
    for s in eval_data:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# ── Report ─────────────────────────────────────────────────────
print(f"Total valid samples           : {len(samples)}")
print(f"Train samples                 : {len(train_data)}")
print(f"Eval  samples                 : {len(eval_data)}")
print(f"Unique clause types           : {unique_clauses}")
print(f"Avg formatted length (chars)  : {avg_len:.0f}")
print(f"\nSaved → {train_path}")
print(f"Saved → {eval_path}")