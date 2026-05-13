# embed_and_store.py — LegallyBound Phase 5: Embed chunks into DuckDB
# Run: python embed_and_store.py

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import duckdb

# ✅ Use model name instead of local path
EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DB_PATH  = Path("legallybound.duckdb")
DATA_DIR = Path("data")

# ── Load all samples ───────────────────────────────────────────
print("Loading data/train.jsonl + data/eval.jsonl ...")
samples = []
for fpath in [DATA_DIR / "train.jsonl", DATA_DIR / "eval.jsonl"]:
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

print(f"Total samples to embed: {len(samples)}")

texts         = [s["context"]       for s in samples]
contracts     = [s["contract_name"] for s in samples]
clause_types  = [s["clause_type"]   for s in samples]

# ── Embed ──────────────────────────────────────────────────────
print(f"Loading embedder: {EMBEDDER_NAME} ...")

# ✅ This will auto-download and cache
embedder = SentenceTransformer(EMBEDDER_NAME)

print("Generating embeddings (batch_size=64) ...")
embeddings = embedder.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print(f"Embedding shape: {embeddings.shape}")

# ── DuckDB ─────────────────────────────────────────────────────
print(f"\nConnecting to {DB_PATH} ...")
con = duckdb.connect(str(DB_PATH))

con.execute("INSTALL vss;")
con.execute("LOAD vss;")

con.execute("""
    CREATE TABLE IF NOT EXISTS contract_chunks (
        id             INTEGER PRIMARY KEY,
        contract_name  VARCHAR,
        clause_type    VARCHAR,
        chunk_text     TEXT,
        embedding      FLOAT[384]
    );
""")

# Clear existing rows for clean re-run
con.execute("DELETE FROM contract_chunks;")

print("Inserting rows ...")
rows = [
    (i, contracts[i], clause_types[i], texts[i], embeddings[i].tolist())
    for i in range(len(samples))
]

con.executemany(
    "INSERT INTO contract_chunks VALUES (?, ?, ?, ?, ?)",
    rows,
)

print("Creating HNSW index ...")

# ✅ Enable persistence for HNSW
con.execute("SET hnsw_enable_experimental_persistence = true;")

con.execute("DROP INDEX IF EXISTS emb_idx;")

con.execute("""
    CREATE INDEX emb_idx
    ON contract_chunks
    USING HNSW (embedding)
    WITH (metric = 'cosine');
""")

row_count = con.execute("SELECT COUNT(*) FROM contract_chunks").fetchone()[0]
con.close()

db_size_mb = DB_PATH.stat().st_size / 1024**2

print(f"\nRows inserted  : {row_count}")
print(f"DB file size   : {db_size_mb:.1f} MB")
print(f"Saved          -> {DB_PATH}")
print("\nDone -- DuckDB vector store ready.")