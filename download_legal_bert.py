# download_legal_bert.py
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os
os.environ["TRANSFORMERS_NO_VISION"] = "1"   # ← stops torchvision from loading

SAVE_DIR = Path("models/legal-bert-base-uncased")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
tokenizer.save_pretrained(str(SAVE_DIR))

print("Downloading model weights...")
model = AutoModel.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    use_safetensors=True,
)
model.save_pretrained(str(SAVE_DIR), safe_serialization=True)

print(f"\nDone. Files in {SAVE_DIR}:")
for f in sorted(SAVE_DIR.iterdir()):
    print(f"  {f.name}")