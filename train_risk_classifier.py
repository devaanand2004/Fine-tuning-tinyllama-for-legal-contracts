# train_risk_classifier.py
# Fine-tunes Legal-BERT on cuad_risk_labels.csv → models/risk_classifier/

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BASE_MODEL_DIR = Path("models/legal-bert-base-uncased")
OUTPUT_DIR     = Path("models/risk_classifier")
DATA_PATH      = "cuad_risk_labels.csv"
MAX_LENGTH     = 256
BATCH_SIZE     = 16
EPOCHS         = 4
SEED           = 42

LABEL2ID = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
ID2LABEL = {0: "HIGH", 1: "MEDIUM", 2: "LOW"}


class RiskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds  = np.argmax(logits, axis=1)
    report = classification_report(
        labels, preds,
        target_names=["HIGH", "MEDIUM", "LOW"],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy" : report["accuracy"],
        "f1_high"  : report["HIGH"]["f1-score"],
        "f1_medium": report["MEDIUM"]["f1-score"],
        "f1_low"   : report["LOW"]["f1-score"],
        "f1_macro" : report["macro avg"]["f1-score"],
    }


def main():
    print("Loading cuad_risk_labels.csv...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["clause_text", "risk_level"])
    df = df[df["risk_level"].isin(["HIGH", "MEDIUM", "LOW"])]

    df["input_text"] = (
        "Clause type: " + df["clause_type"].astype(str)
        + " [SEP] "
        + df["clause_text"].astype(str)
    )
    df["label_id"] = df["risk_level"].map(LABEL2ID)

    print(f"Total samples : {len(df)}")
    print(f"Distribution  :\n{df['risk_level'].value_counts()}\n")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=SEED
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), local_files_only=True)

    train_dataset = RiskDataset(train_df["input_text"].tolist(), train_df["label_id"].tolist(), tokenizer)
    val_dataset   = RiskDataset(val_df["input_text"].tolist(),   val_df["label_id"].tolist(),   tokenizer)
    test_dataset  = RiskDataset(test_df["input_text"].tolist(),  test_df["label_id"].tolist(),  tokenizer)

    print("Loading Legal-BERT base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(BASE_MODEL_DIR),
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    print("\n=== Test set evaluation ===")
    test_results = trainer.predict(test_dataset)
    preds = np.argmax(test_results.predictions, axis=1)
    print(classification_report(
        test_dataset.labels.numpy(), preds,
        target_names=["HIGH", "MEDIUM", "LOW"], zero_division=0,
    ))

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nSaved to {OUTPUT_DIR}")
    print("Done. Update rag_query.py and run the demo.")


if __name__ == "__main__":
    main()
