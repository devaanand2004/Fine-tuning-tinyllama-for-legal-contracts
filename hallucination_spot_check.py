# hallucination_spot_check.py — LegallyBound Phase 2: Spot-check
# Run: python hallucination_spot_check.py

import json
import re
from pathlib import Path

RESULTS_FILE  = Path("results/zero_shot_results.json")
SPOT_CHECK_FILE = Path("results/zero_shot_spot_check.txt")

# ── Load ───────────────────────────────────────────────────────
with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"][:5]


def extract_clause_type(prompt: str) -> str:
    """Pull the clause type out of the prompt string."""
    match = re.search(r"Question:\s*(.+?)\s*</s>", prompt, re.DOTALL)
    return match.group(1).strip() if match else "Unknown"


lines_out = []

for i, sample in enumerate(samples, 1):
    clause_type  = extract_clause_type(sample["prompt"])
    ground_truth = sample["answer"]
    generated    = sample["generated"]

    block = (
        f"\n── Sample {i} ──────────────────────────────────────────\n"
        f"CLAUSE TYPE : {clause_type}\n\n"
        f"GROUND TRUTH:\n{ground_truth}\n\n"
        f"GENERATED   :\n{generated}\n\n"
        "FLAG (human review): Does the generated text cite any\n"
        "statute, section number, or legal clause name that does\n"
        "NOT appear verbatim in the contract excerpt above?\n"
        "──────────────────────────────────────────────────────────\n"
    )

    print(block)
    lines_out.append(block)

# ── Write to file ──────────────────────────────────────────────
SPOT_CHECK_FILE.parent.mkdir(exist_ok=True)
with open(SPOT_CHECK_FILE, "w", encoding="utf-8") as f:
    f.write("LegallyBound — Zero-shot Hallucination Spot-Check\n")
    f.write("=" * 60 + "\n")
    f.writelines(lines_out)

print(f"\nSpot-check written → {SPOT_CHECK_FILE}")