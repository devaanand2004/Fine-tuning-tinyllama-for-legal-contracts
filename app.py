# app.py — LegallyBound Web UI Backend (FastAPI)
# Serves the frontend and exposes APIs for upload, risk extraction (SSE), and search.
# Optimisations:
#   1. Merged prompts   — clause-type + summary in a single TinyLlama call (3→2 per chunk)
#   2. Smart filtering  — skip low-relevance boilerplate chunks via embedding similarity
#   3. SSE streaming    — results stream to the UI in real time

import gc
import json
import os
import re
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore", message="You passed `quantization_config`", category=UserWarning)

# ── Config ────────────────────────────────────────────────────
EMBEDDER_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
MODELS_DIR          = Path("models")
DB_PATH             = Path("legallybound.duckdb")
RISK_CLASSIFIER_DIR = Path("models/risk_classifier")
UPLOAD_DIR          = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ID2LABEL = {0: "HIGH", 1: "MEDIUM", 2: "LOW"}
RISK_REASONS = {
    "HIGH"  : "Clause poses significant legal or financial exposure — review with counsel.",
    "MEDIUM": "Clause requires attention but risk is manageable with standard review.",
    "LOW"   : "Standard boilerplate clause with minimal legal risk.",
}

# Anchor phrases for smart filtering — chunks similar to these are "legally interesting"
LEGAL_ANCHORS = [
    "termination and cancellation rights",
    "indemnification and liability obligations",
    "non-compete and non-solicitation restrictions",
    "confidentiality and non-disclosure requirements",
    "limitation of liability and damages cap",
    "governing law and jurisdiction",
    "intellectual property assignment and licensing",
    "warranty and representations",
    "payment terms and penalties",
    "force majeure and excusable delays",
    "dispute resolution and arbitration",
    "data protection and privacy obligations",
    "insurance and risk allocation",
    "assignment and change of control",
]

# Minimum cosine similarity to any anchor for a chunk to be analyzed
RELEVANCE_THRESHOLD = 0.25

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="LegallyBound — Contract Risk Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Helpers ───────────────────────────────────────────────────
def _extract_label(clause_type_raw: str) -> str:
    match = re.search(r'"([^"]+)"', clause_type_raw)
    return match.group(1) if match else clause_type_raw


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        EMBEDDER_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


# ── TinyLlama loader ─────────────────────────────────────────
def _load_model(variant: str = "B"):
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


def _classify_risk(clause_type: str, clause_text: str) -> dict:
    model, tokenizer = _load_risk_classifier()
    input_text = f"Clause type: {clause_type} [SEP] {clause_text[:400]}"
    inputs     = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs      = torch.softmax(logits, dim=1)[0]
    label_id   = torch.argmax(probs).item()
    level      = ID2LABEL[label_id]
    confidence = float(probs[label_id])
    return {
        "risk_level" : level,
        "reason"     : RISK_REASONS[level],
        "confidence" : confidence,
    }


# ── PDF / TXT parsing ────────────────────────────────────────
def _parse_file(file_path: Path) -> str:
    """Extract text content from a PDF or TXT file."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="replace")

    elif suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc  = fitz.open(str(file_path))
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="PyMuPDF (fitz) not installed. Run: pip install PyMuPDF",
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Smart relevance filtering ────────────────────────────────
_anchor_embeddings = None

def _get_anchor_embeddings(embedder):
    """Compute and cache anchor embeddings for smart filtering."""
    global _anchor_embeddings
    if _anchor_embeddings is None:
        _anchor_embeddings = embedder.encode(
            LEGAL_ANCHORS, convert_to_numpy=True, show_progress_bar=False,
        )
    return _anchor_embeddings


def _compute_relevance(chunk_embeddings: np.ndarray, embedder) -> np.ndarray:
    """
    For each chunk, compute max cosine similarity to any legal anchor.
    Returns array of relevance scores (one per chunk).
    """
    anchors = _get_anchor_embeddings(embedder)

    # Normalize for cosine similarity
    chunk_norm  = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
    anchor_norm = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)

    # (num_chunks, num_anchors)
    sim_matrix = chunk_norm @ anchor_norm.T
    return sim_matrix.max(axis=1)


# ── Embed uploaded contract into a session table ──────────────
def _embed_contract(contract_id: str, filename: str, text: str):
    """Chunk, embed, and store a user-uploaded contract in the DB."""
    chunks   = _chunk_text(text)
    embedder = _load_embedder()
    embs     = embedder.encode(chunks, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    # Compute relevance scores
    relevance = _compute_relevance(embs, embedder)

    con = duckdb.connect(str(DB_PATH))
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")

    # Drop and recreate to ensure schema has relevance column
    con.execute("DROP TABLE IF EXISTS user_uploads;")
    con.execute("""
        CREATE TABLE user_uploads (
            id             VARCHAR,
            contract_id    VARCHAR,
            contract_name  VARCHAR,
            clause_type    VARCHAR,
            chunk_text     TEXT,
            embedding      FLOAT[384],
            relevance      FLOAT
        );
    """)

    # Remove previous upload with same contract_id
    con.execute("DELETE FROM user_uploads WHERE contract_id = ?", [contract_id])

    rows = [
        (f"{contract_id}_{i}", contract_id, filename, "uploaded", chunk, emb.tolist(), float(rel))
        for i, (chunk, emb, rel) in enumerate(zip(chunks, embs, relevance))
    ]
    con.executemany("INSERT INTO user_uploads VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()

    relevant_count = int((relevance >= RELEVANCE_THRESHOLD).sum())

    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    return len(chunks), relevant_count


# ══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_contract(file: UploadFile = File(...)):
    """Upload a contract file (PDF or TXT), parse and embed it."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No file provided.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".txt"):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    contract_id = str(uuid.uuid4())[:8]
    save_path   = UPLOAD_DIR / f"{contract_id}{suffix}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text        = _parse_file(save_path)
    num_chunks, relevant_count = _embed_contract(contract_id, file.filename, text)

    return {
        "contract_id"    : contract_id,
        "filename"       : file.filename,
        "num_chunks"     : num_chunks,
        "relevant_chunks": relevant_count,
        "text_length"    : len(text),
    }


@app.get("/api/risks/stream")
async def stream_risks(contract_id: str):
    """
    SSE endpoint — streams risk results one chunk at a time.
    Optimisations:
      - Merged prompt (clause type + summary in one call)
      - Smart filtering (skips low-relevance chunks)
      - Streams results as they are ready
    """

    def _event_stream():
        con = duckdb.connect(str(DB_PATH), read_only=True)

        try:
            rows = con.execute(
                "SELECT id, chunk_text, relevance FROM user_uploads "
                "WHERE contract_id = ? ORDER BY id",
                [contract_id],
            ).fetchall()
        except duckdb.CatalogException:
            con.close()
            yield f"data: {json.dumps({'type': 'error', 'message': 'No uploaded contracts found.'})}\n\n"
            return

        con.close()

        if not rows:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Contract {contract_id} not found.'})}\n\n"
            return

        total_chunks   = len(rows)
        relevant_rows  = [(cid, txt, rel) for cid, txt, rel in rows if rel >= RELEVANCE_THRESHOLD]
        skipped_count  = total_chunks - len(relevant_rows)

        # Send initial metadata
        yield f"data: {json.dumps({'type': 'meta', 'total': total_chunks, 'relevant': len(relevant_rows), 'skipped': skipped_count})}\n\n"

        if not relevant_rows:
            yield f"data: {json.dumps({'type': 'done', 'message': 'No legally relevant chunks found.'})}\n\n"
            return

        # Load TinyLlama once
        llm_model, tokenizer = _load_model("B")
        start_time = time.time()

        for idx, (chunk_id, chunk_text, relevance_score) in enumerate(relevant_rows):

            t0 = time.time()

            # ── OPTIMISATION 1: Merged prompt ─────────────────
            # Single TinyLlama call for BOTH clause type + summary
            merged_prompt = (
                "<|system|>\n"
                "You are a legal contract analyst. Given a contract excerpt:\n"
                "1. Identify the clause type in 2-5 words.\n"
                "2. Summarise the key obligation or condition in 1-2 sentences.\n\n"
                "Format your response exactly as:\n"
                "CLAUSE TYPE: <type>\n"
                "SUMMARY: <summary>\n</s>\n"
                "<|user|>\n"
                f"Contract excerpt:\n{chunk_text[:500]}\n</s>\n"
                "<|assistant|>\n"
            )
            merged_output = _generate(llm_model, tokenizer, merged_prompt, max_new_tokens=150)

            # Parse the merged output
            clause_type = "General"
            summary     = merged_output

            ct_match = re.search(r"CLAUSE\s*TYPE:\s*(.+?)(?:\n|SUMMARY)", merged_output, re.IGNORECASE)
            if ct_match:
                clause_type = ct_match.group(1).strip().strip('"').strip("'")

            sm_match = re.search(r"SUMMARY:\s*(.+)", merged_output, re.IGNORECASE | re.DOTALL)
            if sm_match:
                summary = sm_match.group(1).strip()

            clause_type = _extract_label(clause_type) if clause_type else "General"

            # ── Step 2: Legal-BERT risk classification ────────
            risk = _classify_risk(clause_type, chunk_text)

            # ── OPTIMISATION 1 continued: contextual reason ───
            reason_prompt = (
                "<|system|>\n"
                "You are a legal contract analyst. Explain briefly why the given clause "
                "was classified at this risk level. Keep it to 1-2 sentences.\n</s>\n"
                "<|user|>\n"
                f"Clause type: {clause_type}\n"
                f"Clause summary: {summary}\n"
                f"Assessed Risk Level: {risk['risk_level']}\n\n"
                "Provide a brief, contextual explanation for this risk level.\n</s>\n"
                "<|assistant|>\n"
            )
            contextual_reason = _generate(llm_model, tokenizer, reason_prompt, max_new_tokens=100)

            elapsed = time.time() - t0

            result = {
                "type"         : "risk",
                "index"        : idx,
                "total"        : len(relevant_rows),
                "chunk_id"     : chunk_id,
                "clause_type"  : clause_type,
                "summary"      : summary,
                "risk_level"   : risk["risk_level"],
                "risk_reason"  : f"{contextual_reason} (confidence: {risk['confidence']:.0%})",
                "confidence"   : risk["confidence"],
                "relevance"    : round(relevance_score, 3),
                "excerpt"      : chunk_text[:300],
                "time_seconds" : round(elapsed, 1),
            }

            # ── OPTIMISATION 3: Stream the result immediately ─
            yield f"data: {json.dumps(result)}\n\n"

        # Cleanup
        del llm_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        total_time = time.time() - start_time
        yield f"data: {json.dumps({'type': 'done', 'total_time': round(total_time, 1), 'analyzed': len(relevant_rows), 'skipped': skipped_count})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Keep the old endpoint for backwards-compat but redirect to stream
@app.get("/api/risks")
async def extract_risks(contract_id: str):
    """Non-streaming fallback (collects all results then returns)."""
    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        rows = con.execute(
            "SELECT id, chunk_text, relevance FROM user_uploads "
            "WHERE contract_id = ? AND relevance >= ? ORDER BY id",
            [contract_id, RELEVANCE_THRESHOLD],
        ).fetchall()
    except duckdb.CatalogException:
        con.close()
        raise HTTPException(status_code=404, detail="No uploaded contracts found. Upload one first.")

    con.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"Contract '{contract_id}' not found or no relevant chunks.")

    llm_model, tokenizer = _load_model("B")
    results = []

    for chunk_id, chunk_text, rel in rows:
        # Merged prompt
        merged_prompt = (
            "<|system|>\n"
            "You are a legal contract analyst. Given a contract excerpt:\n"
            "1. Identify the clause type in 2-5 words.\n"
            "2. Summarise the key obligation or condition in 1-2 sentences.\n\n"
            "Format your response exactly as:\n"
            "CLAUSE TYPE: <type>\n"
            "SUMMARY: <summary>\n</s>\n"
            "<|user|>\n"
            f"Contract excerpt:\n{chunk_text[:500]}\n</s>\n"
            "<|assistant|>\n"
        )
        merged_output = _generate(llm_model, tokenizer, merged_prompt, max_new_tokens=150)

        clause_type = "General"
        summary     = merged_output

        ct_match = re.search(r"CLAUSE\s*TYPE:\s*(.+?)(?:\n|SUMMARY)", merged_output, re.IGNORECASE)
        if ct_match:
            clause_type = ct_match.group(1).strip().strip('"').strip("'")

        sm_match = re.search(r"SUMMARY:\s*(.+)", merged_output, re.IGNORECASE | re.DOTALL)
        if sm_match:
            summary = sm_match.group(1).strip()

        clause_type = _extract_label(clause_type) if clause_type else "General"

        risk = _classify_risk(clause_type, chunk_text)

        reason_prompt = (
            "<|system|>\n"
            "You are a legal contract analyst. Explain briefly why the given clause "
            "was classified at this risk level. Keep it to 1-2 sentences.\n</s>\n"
            "<|user|>\n"
            f"Clause type: {clause_type}\n"
            f"Clause summary: {summary}\n"
            f"Assessed Risk Level: {risk['risk_level']}\n\n"
            "Provide a brief, contextual explanation for this risk level.\n</s>\n"
            "<|assistant|>\n"
        )
        contextual_reason = _generate(llm_model, tokenizer, reason_prompt, max_new_tokens=100)

        results.append({
            "chunk_id"     : chunk_id,
            "clause_type"  : clause_type,
            "summary"      : summary,
            "risk_level"   : risk["risk_level"],
            "risk_reason"  : f"{contextual_reason} (confidence: {risk['confidence']:.0%})",
            "confidence"   : risk["confidence"],
            "excerpt"      : chunk_text[:300],
        })

    del llm_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"contract_id": contract_id, "total_chunks": len(results), "risks": results}


@app.get("/api/search")
async def semantic_search(
    contract_id: str,
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=20),
):
    """
    Semantic search within an uploaded contract.
    Uses embedder for similarity + TinyLlama for clause extraction + Legal-BERT for risk.
    """
    embedder  = _load_embedder()
    query_vec = embedder.encode([q], convert_to_numpy=True)[0]
    vec_lit   = "[" + ", ".join(str(float(x)) for x in query_vec) + "]"

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        con.execute("LOAD vss;")
        sql = f"""
            SELECT id, chunk_text,
                   array_cosine_similarity(embedding, {vec_lit}::FLOAT[384]) AS score
            FROM user_uploads
            WHERE contract_id = ?
            ORDER BY score DESC
            LIMIT {top_k}
        """
        rows = con.execute(sql, [contract_id]).fetchall()
    except duckdb.CatalogException:
        con.close()
        raise HTTPException(status_code=404, detail="No uploaded contracts found.")

    con.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No results for contract '{contract_id}'.")

    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    # Build context from top results
    context = "\n\n---\n\n".join(row[1][:300] for row in rows)

    # TinyLlama answer generation
    llm_model, tokenizer = _load_model("B")
    extraction_prompt = (
        "<|system|>\n"
        "You are a legal contract analyst. Read the excerpts and answer "
        "the question in 2-3 sentences. Do NOT copy raw contract text. "
        "Summarise the key obligation or condition in plain English.\n</s>\n"
        "<|user|>\n"
        f"Contract excerpts:\n{context}\n\n"
        f"Question: {q}\n</s>\n"
        "<|assistant|>\n"
    )
    answer = _generate(llm_model, tokenizer, extraction_prompt, max_new_tokens=200)

    # Risk classification on the answer
    risk = _classify_risk("General", answer)

    # Contextual reason
    reason_prompt = (
        "<|system|>\n"
        "You are a legal contract analyst. Explain briefly why the given clause "
        "was classified at this risk level. Keep it to 1-2 sentences.\n</s>\n"
        "<|user|>\n"
        f"Clause summary: {answer}\n"
        f"Assessed Risk Level: {risk['risk_level']}\n\n"
        "Provide a brief, contextual explanation.\n</s>\n"
        "<|assistant|>\n"
    )
    contextual_reason = _generate(llm_model, tokenizer, reason_prompt, max_new_tokens=100)

    del llm_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "query"           : q,
        "answer"          : answer,
        "risk_level"      : risk["risk_level"],
        "risk_reason"     : f"{contextual_reason} (confidence: {risk['confidence']:.0%})",
        "risk_confidence" : risk["confidence"],
        "source_chunks"   : [
            {"id": r[0], "text": r[1][:400], "score": round(float(r[2]), 4)}
            for r in rows
        ],
    }


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
