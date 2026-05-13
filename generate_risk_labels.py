# generate_risk_labels.py
# Uses Groq (free tier) to label each CUAD clause with HIGH / MEDIUM / LOW risk
# Get your free API key at: https://console.groq.com

import os
# Set your Groq API key as an environment variable before running:
#   export GROQ_API_KEY="your-key-here"  (Linux/Mac)
#   set GROQ_API_KEY=your-key-here       (Windows)
# Get your free API key at: https://console.groq.com
assert os.environ.get("GROQ_API_KEY"), "Set GROQ_API_KEY environment variable first"

import json
import re
import time
import pandas as pd
from pathlib import Path
from collections import Counter
from groq import Groq
from tqdm import tqdm

client = Groq(api_key=os.environ["GROQ_API_KEY"])

RAW_PATH   = Path("CUAD_v1/CUAD_v1.json")
SAVE_PATH  = Path("cuad_risk_labels.csv")
SAVE_EVERY = 50    # checkpoint to CSV every N labelled records
REQUEST_DELAY = 2  # seconds between requests — Groq free tier: 14,400 req/day
                   # 2s delay = ~1,800 req/hour, well within limits

# ── Risk rubric — gives the model a consistent baseline ──────
RISK_RUBRIC = """
Use this rubric as your baseline, but override it if the actual
clause text justifies a different rating.

HIGH risk (significant legal or financial exposure):
  Uncapped Liability, Liquidated Damages, Non-Compete,
  Irrevocable Or Perpetual License, Covenant Not To Sue,
  Ip Ownership Assignment, Joint Ip Ownership,
  Unlimited/All-You-Can-Eat-License, Exclusivity,
  Source Code Escrow

MEDIUM risk (requires attention, manageable exposure):
  Termination For Convenience, Anti-Assignment, Audit Rights,
  Change Of Control, Revenue/Profit Sharing, Post-Termination Services,
  Price Restrictions, Rofr/Rofo/Rofn, Cap On Liability,
  Non-Transferable License, Affiliate License-Licensor,
  Affiliate License-Licensee, License Grant, Minimum Commitment,
  Volume Restriction, Insurance, No-Solicit Of Customers,
  No-Solicit Of Employees, Non-Disparagement,
  Competitive Restriction Exception, Most Favored Nation,
  Third Party Beneficiary, Warranty Duration

LOW risk (standard boilerplate, minimal exposure):
  Governing Law, Agreement Date, Effective Date, Expiration Date,
  Document Name, Parties, Renewal Term,
  Notice Period To Terminate Renewal
"""


def get_risk_label(clause_type: str, clause_text: str, retries: int = 3) -> dict | None:
    """
    Call Groq (llama-3.1-8b-instant) to generate a risk label for one clause.
    Returns {"risk_level": "HIGH"|"MEDIUM"|"LOW", "reason": str}
    or None if all retries fail.
    """
    prompt = (
        f"{RISK_RUBRIC}\n\n"
        f"Clause type: {clause_type}\n"
        f"Clause text: {clause_text[:600]}\n\n"
        "Based on the rubric AND the actual clause text, rate the risk.\n"
        "Output ONLY a JSON object — no text outside the JSON:\n"
        '{"risk_level": "HIGH" or "MEDIUM" or "LOW", '
        '"reason": "one sentence explaining based on the actual text"}'
    )

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1,   # low temp for consistent JSON output
            )

            raw   = response.choices[0].message.content.strip()
            start = raw.find("{")
            end   = raw.rfind("}") + 1

            if start < 0 or end <= start:
                raise ValueError("No JSON found in response")

            parsed = json.loads(raw[start:end])
            level  = parsed.get("risk_level", "").upper()
            reason = parsed.get("reason", "").strip()

            if level not in ("HIGH", "MEDIUM", "LOW"):
                raise ValueError(f"Invalid risk_level: {level}")
            if not reason:
                raise ValueError("Empty reason")

            return {"risk_level": level, "reason": reason}

        except Exception as e:
            err_str = str(e)

            # Rate limit hit — wait longer before retrying
            if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                wait = 60
                print(f"\n  Rate limited — waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            # Non-retryable errors — fail fast
            if "401" in err_str or "authentication" in err_str.lower() or "api_key" in err_str.lower():
                print(f"\n  FATAL auth error: {err_str}")
                print("  → Check your GROQ_API_KEY at https://console.groq.com")
                raise SystemExit(1)

            # Other errors — exponential backoff
            wait = 2 ** attempt   # 1s, 2s, 4s
            print(f"\n  Attempt {attempt+1} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)

    return None  # all retries exhausted


def main():
    # ── Load CUAD ─────────────────────────────────────────────
    print("Loading CUAD_v1.json...")
    with open(RAW_PATH, encoding="utf-8") as f:
        cuad_raw = json.load(f)

    # ── Flatten into (clause_type, clause_text) pairs ─────────
    records = []
    for contract in cuad_raw["data"]:
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

    print(f"Total valid records to label: {len(records)}")
    print(f"Distribution: {dict(Counter(r['clause_type'] for r in records))}\n")

    # ── Resume from checkpoint if it exists ───────────────────
    already_done = 0
    results = []

    if SAVE_PATH.exists():
        existing = pd.read_csv(SAVE_PATH)
        results = existing.to_dict("records")
        already_done = len(results)
        print(f"Resuming from checkpoint — {already_done} already labelled\n")

    # ── Generate labels ───────────────────────────────────────
    skipped = 0

    for i, rec in enumerate(tqdm(records[already_done:], desc="Labelling",
                                 initial=already_done, total=len(records))):
        label = get_risk_label(rec["clause_type"], rec["clause_text"])

        if label is None:
            skipped += 1
            tqdm.write(f"  SKIPPED ({rec['clause_type']}) after all retries")
        else:
            results.append({
                "clause_type": rec["clause_type"],
                "clause_text": rec["clause_text"],
                "risk_level":  label["risk_level"],
                "reason":      label["reason"],
            })

        # Checkpoint save every SAVE_EVERY records
        if len(results) % SAVE_EVERY == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(SAVE_PATH, index=False)
            tqdm.write(f"  Checkpoint saved — {len(results)} labelled, {skipped} skipped")

        # Respect Groq rate limits
        time.sleep(REQUEST_DELAY)

    # ── Final save ────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"Done.")
    print(f"Total labelled : {len(results)}")
    print(f"Total skipped  : {skipped}")
    print(f"Saved to       : {SAVE_PATH}")
    print(f"\nRisk distribution:")
    print(df["risk_level"].value_counts())
    print(f"\nPer clause type breakdown:")
    print(df.groupby(["clause_type", "risk_level"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
