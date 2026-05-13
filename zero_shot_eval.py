# zero_shot_eval.py — LegallyBound Phase 2: Zero-shot baseline
# Run: python zero_shot_eval.py

import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH   = Path("E:/your_project/TinyLlama-1.1B-Chat-v1.0")
EVAL_JSONL   = Path("data/eval.jsonl")
RESULTS_DIR  = Path("results")
RESULTS_FILE = RESULTS_DIR / "zero_shot_results.json"

N_SAMPLES    = 50
random.seed(42)

# ── Load eval data ─────────────────────────────────────────────
print("Loading eval.jsonl …")
with open(EVAL_JSONL, "r", encoding="utf-8") as f:
    eval_data = [json.loads(line) for line in f]

sample_pool = random.sample(eval_data, min(N_SAMPLES, len(eval_data)))
print(f"Evaluating on {len(sample_pool)} samples")

# ── Load tokenizer & model ─────────────────────────────────────
print("Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (4-bit) …")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
)
model.eval()
print("Model ready.")

# ── Inference loop ─────────────────────────────────────────────
records = []
for i, sample in enumerate(sample_pool):
    context     = sample["context"]
    clause_type = sample["clause_type"]
    ground_truth = sample["answer"]

    prompt = (
        "<|system|>\n"
        "You are a legal contract analyst. Extract and "
        "summarize the requested clause clearly and "
        "concisely.\n</s>\n"
        "<|user|>\n"
        f"Contract excerpt:\n{context}\n\n"
        f"Question: {clause_type}\n</s>\n"
        "<|assistant|>\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    t0 = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - t0
    new_tokens = output_ids[0][input_len:]
    generated  = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    records.append({
        "prompt":       prompt,
        "generated":    generated,
        "answer":       ground_truth,
        "clause_type":  clause_type,
        "time_s":       elapsed,
    })

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(sample_pool)} …")

print("Inference complete. Computing metrics …")

# ── ROUGE-L ────────────────────────────────────────────────────
from rouge_score import rouge_scorer as rs_module

scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
rouge_scores = [
    scorer.score(r["answer"], r["generated"])["rougeL"].fmeasure
    for r in records
]
avg_rouge = sum(rouge_scores) / len(rouge_scores)

# ── BERTScore — uses online distilbert model ───────────────────
from bert_score import score as bert_score_fn

predictions = [r["generated"] for r in records]
references  = [r["answer"]    for r in records]

_, _, bert_f1 = bert_score_fn(
    predictions,
    references,
    lang="en",
    model_type="distilbert-base-uncased",
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
)
avg_bertscore = bert_f1.mean().item()

# ── Timing ─────────────────────────────────────────────────────
avg_time = sum(r["time_s"] for r in records) / len(records)

# ── Save ───────────────────────────────────────────────────────
RESULTS_DIR.mkdir(exist_ok=True)
result_payload = {
    "rouge_l":       round(avg_rouge, 4),
    "bertscore_f1":  round(avg_bertscore, 4),
    "avg_time_s":    round(avg_time, 3),
    "samples": [
        {
            "prompt":      r["prompt"],
            "generated":   r["generated"],
            "answer":      r["answer"],
            "clause_type": r["clause_type"],
        }
        for r in records
    ],
}
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(result_payload, f, indent=2, ensure_ascii=False)

print(f"\nResults saved → {RESULTS_FILE}")

# ── Print table ────────────────────────────────────────────────
print("""
============================================
ZERO-SHOT BASELINE RESULTS (50 samples)
============================================
Metric            | Score
------------------|-------""")
print(f"ROUGE-L           | {avg_rouge:.3f}")
print(f"BERTScore F1      | {avg_bertscore:.3f}")
print(f"Avg time/sample(s)| {avg_time:.2f}")
print("============================================")

# ── Cleanup ────────────────────────────────────────────────────
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()
print("GPU memory cleared.")