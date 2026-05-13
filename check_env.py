# check_env.py — LegallyBound Phase 0: Environment Verification
# Run: python check_env.py

import gc
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
MODEL_PATH = Path("E:/your_project/TinyLlama-1.1B-Chat-v1.0")

# ── Library versions ───────────────────────────────────────────
print("=" * 55)
print("LIBRARY VERSIONS")
print("=" * 55)

import transformers
import peft
import bitsandbytes
import duckdb
import sentence_transformers

print(f"torch                : {torch.__version__}")
print(f"transformers         : {transformers.__version__}")
print(f"peft                 : {peft.__version__}")
print(f"bitsandbytes         : {bitsandbytes.__version__}")
print(f"duckdb               : {duckdb.__version__}")
print(f"sentence_transformers: {sentence_transformers.__version__}")

# ── CUDA info ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("CUDA / GPU INFO")
print("=" * 55)
print(f"CUDA available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_vram  = (torch.cuda.get_device_properties(0).total_memory
                  - torch.cuda.memory_allocated(0)) / 1024**3
    print(f"GPU name       : {gpu_name}")
    print(f"Total VRAM (GB): {total_vram:.2f}")
    print(f"Free  VRAM (GB): {free_vram:.2f}")
else:
    print("WARNING: No CUDA GPU detected — model will run on CPU (very slow)")

# ── Load tokenizer ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("TOKENIZER LOAD TEST")
print("=" * 55)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)
print(f"Tokenizer loaded from : {MODEL_PATH}")
print(f"Vocab size            : {tokenizer.vocab_size}")

# ── Load model in 4-bit ────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL LOAD TEST (4-bit BnB)")
print("=" * 55)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
print(f"Model loaded from : {MODEL_PATH}")

# ── Quick inference test ───────────────────────────────────────
print("\n" + "=" * 55)
print("INFERENCE TEST")
print("=" * 55)
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
    )
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Input  : Hello")
print(f"Output : {decoded}")

# ── Memory after load ──────────────────────────────────────────
if torch.cuda.is_available():
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"\nGPU memory used after model load: {mem_used:.2f} GB")

# ── Cleanup ────────────────────────────────────────────────────
del model
del tokenizer
gc.collect()
torch.cuda.empty_cache()

print("\n✅  Environment OK")