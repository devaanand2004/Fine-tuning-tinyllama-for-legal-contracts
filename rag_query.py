# rag_query.py — LegallyBound Phase 5: RAG + Legal-BERT Risk Classification

import gc
import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import duckdb

warnings.filterwarnings("ignore", message="You passed `quantization_config`", category=UserWarning)

EMBEDDER_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
MODELS_DIR          = Path("models")
DB_PATH             = Path("legallybound.duckdb")
RISK_CLASSIFIER_DIR = Path("models/risk_classifier")

ID2LABEL = {0: "HIGH", 1: "MEDIUM", 2: "LOW"}

RISK_REASONS = {
    "HIGH"  : "Clause poses significant legal or financial exposure — review with counsel.",
    "MEDIUM": "Clause requires attention but risk is manageable with standard review.",
    "LOW"   : "Standard boilerplate clause with minimal legal risk.",
}


# ── Helpers ───────────────────────────────────────────────────
def _extract_label(clause_type_raw: str) -> str:
    match = re.search(r'"([^"]+)"', clause_type_raw)
    return match.group(1) if match else clause_type_raw


# ── Embedder ──────────────────────────────────────────────────
def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBEDDER_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


# ── TinyLlama ─────────────────────────────────────────────────
def _load_model(variant: str):
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, GenerationConfig,
    )
    model_path = MODELS_DIR / f"legallybound_{variant}"
    tokenizer  = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), quantization_config=bnb_config,
        device_map="auto", local_files_only=True,
    )
    model.eval()
    model.generation_config = GenerationConfig(
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    inputs    = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


# ── Legal-BERT risk classifier (cached) ───────────────────────
_risk_model     = None
_risk_tokenizer = None

def _load_risk_classifier():
    global _risk_model, _risk_tokenizer
    if _risk_model is not None:
        return _risk_model, _risk_tokenizer

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("Loading Legal-BERT risk classifier...")
    _risk_tokenizer = AutoTokenizer.from_pretrained(str(RISK_CLASSIFIER_DIR), local_files_only=True)
    _risk_model     = AutoModelForSequenceClassification.from_pretrained(
        str(RISK_CLASSIFIER_DIR), local_files_only=True
    )
    _risk_model.eval()
    return _risk_model, _risk_tokenizer


def _risk_label_classifier(clause_type: str, clause_answer: str) -> dict:
    model, tokenizer = _load_risk_classifier()

    input_text = f"Clause type: {clause_type} [SEP] {clause_answer[:400]}"
    inputs     = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=1)[0]
    label_id   = torch.argmax(probs).item()
    level      = ID2LABEL[label_id]
    confidence = float(probs[label_id])

    return {
        "risk_level" : level,
        "reason"     : f"{RISK_REASONS[level]} (confidence: {confidence:.0%})",
        "confidence" : confidence,
    }


# ── Utility ───────────────────────────────────────────────────
def list_clause_types() -> list[str]:
    con  = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute("SELECT DISTINCT clause_type FROM contract_chunks ORDER BY clause_type").fetchall()
    con.close()
    return [r[0] for r in rows]


# ── Main RAG pipeline ─────────────────────────────────────────
def rag_answer(
    query: str,
    clause_filter: Optional[str] = None,
    model_variant: str = "B",
    top_k: int = 5,
    include_risk: bool = True,
) -> dict:

    # Step 1 — embed query
    embedder    = _load_embedder()
    query_vec   = embedder.encode([query], convert_to_numpy=True)[0]
    vec_literal = "[" + ", ".join(str(float(x)) for x in query_vec) + "]"

    # Step 2 — retrieve from DuckDB
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute("LOAD vss;")

    sql = f"""
        SELECT id, contract_name, clause_type, chunk_text,
               array_cosine_similarity(embedding, {vec_literal}::FLOAT[384]) AS score
        FROM contract_chunks
        {"WHERE clause_type ILIKE '%" + clause_filter + "%'" if clause_filter else ""}
        ORDER BY score DESC
        LIMIT {top_k}
    """
    rows    = con.execute(sql).fetchall()
    cols    = ["id", "contract_name", "clause_type", "chunk_text", "score"]
    results = [dict(zip(cols, row)) for row in rows]
    con.close()

    if not results:
        hint = (
            f"No chunks found matching clause_filter='{clause_filter}'. "
            "Call list_clause_types() to see available labels."
            if clause_filter else "No relevant contract chunks found."
        )
        return {
            "answer": hint, "risk_level": "UNKNOWN", "risk_reason": "No data to assess.",
            "risk_confidence": 0.0, "source_chunks": [], "source_scores": [],
            "model_variant": model_variant,
        }

    context = "\n\n---\n\n".join(r["chunk_text"][:300] for r in results)

    # Step 3 — clause extraction via TinyLlama
    extraction_prompt = (
        "<|system|>\n"
        "You are a legal contract analyst. Read the excerpts and answer "
        "the question in 2-3 sentences. Do NOT copy raw contract text. "
        "Summarise the key obligation or condition in plain English.\n</s>\n"
        "<|user|>\n"
        f"Contract excerpts:\n{context}\n\n"
        f"Question: {query}\n</s>\n"
        "<|assistant|>\n"
    )

    llm_model, tokenizer = _load_model(model_variant)
    clause_answer = _generate(llm_model, tokenizer, extraction_prompt, max_new_tokens=200)

    # Step 4 — risk classification via Legal-BERT
    risk = {"risk_level": "N/A", "reason": "Risk disabled", "confidence": 0.0}

    if include_risk:
        raw_clause_type = clause_filter or results[0]["clause_type"]
        detected_clause = _extract_label(raw_clause_type)
        risk = _risk_label_classifier(detected_clause, clause_answer)
        
        reason_prompt = (
            "<|system|>\n"
            "You are a legal contract analyst. Explain briefly why the given clause was classified at this risk level. Keep it to 1-2 sentences.\n</s>\n"
            "<|user|>\n"
            f"Clause type: {detected_clause}\n"
            f"Clause summary: {clause_answer}\n"
            f"Assessed Risk Level: {risk['risk_level']}\n\n"
            "Provide a brief, contextual explanation for why this clause has this risk level.\n</s>\n"
            "<|assistant|>\n"
        )
        contextual_reason = _generate(llm_model, tokenizer, reason_prompt, max_new_tokens=100)
        risk["reason"] = f"{contextual_reason} (confidence: {risk['confidence']:.0%})"

    # Free TinyLlama (saves VRAM/RAM)
    del llm_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "answer"          : clause_answer,
        "risk_level"      : risk["risk_level"],
        "risk_reason"     : risk["reason"],
        "risk_confidence" : risk.get("confidence", 0.0),
        "source_chunks"   : [r["chunk_text"] for r in results],
        "source_scores"   : [float(r["score"]) for r in results],
        "model_variant"   : model_variant,
    }


# ── Demo ──────────────────────────────────────────────────────
if __name__ == "__main__":

    demo_queries = [
        {"query": "What are the termination conditions?",     "clause_filter": "Termination For Convenience"},
        {"query": "What are the indemnification obligations?","clause_filter": None},
        {"query": "Are there any non-compete restrictions?",  "clause_filter": "Non-Compete"},
    ]

    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*60}")
        print(f"DEMO QUERY {i}: {demo['query']}")
        print(f"Clause filter : {demo['clause_filter']}")
        print("="*60)

        result = rag_answer(**demo, model_variant="B")

        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nRISK LEVEL      : {result['risk_level']}")
        print(f"RISK CONFIDENCE : {result['risk_confidence']:.0%}")
        print(f"RISK REASON     : {result['risk_reason']}")

        if result["source_scores"]:
            print(f"\nTOP SOURCE CHUNK (score {result['source_scores'][0]:.3f}):")
            print(result["source_chunks"][0][:400] + "...")
        else:
            print("\nNo source chunks returned.")
