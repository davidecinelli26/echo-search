"""
Echo-Search | Fase 3 — Training dello Stance Detector (DeBERTa-v3-base)
=======================================================================
Full fine-tuning di `microsoft/deberta-v3-base` (3 classi: AGAINST/FAVOR/NEUTRAL)
su VAST+SemEval (data/stance/train.jsonl), con HF Trainer, fp16, early stopping.

SCELTE ARCHITETTURALI:
----------------------
- FULL FINE-TUNING (no LoRA): il modello è piccolo (184M) e sta nei 6 GB in fp16;
  il full FT massimizza il Macro-F1 sulla classe minoritaria (NEUTRAL) rispetto
  ad adapter a basso rango (Sun et al. 2019).
- INPUT A COPPIA (target, text): realizza [CLS] target [SEP] documento [SEP] in
  modo nativo (Devlin et al. 2019), non per concatenazione manuale.
- EARLY STOPPING su Macro-F1 di validation: seleziona il checkpoint migliore
  sulla metrica che ci interessa (classi sbilanciate), non sulla loss.

Runtime GPU (Task 3) — comando verificato:
    uv run --no-sync --with "torch==2.2.2" --with "accelerate>=0.30" \\
      --with "datasets>=2.19" --index "https://download.pytorch.org/whl/cu121" \\
      python scripts/train_stance.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.evaluation.metrics import compute_macro_f1
from src.nlp.stance_data import (
    ID2LABEL,
    LABEL2ID,
    load_stance_jsonl,
    stratified_split,
)

logger = logging.getLogger(__name__)


def build_hf_dataset(rows: list[dict], tokenizer, max_len: int) -> Dataset:
    """Tokenizza le coppie (target, text) e mappa le label agli id canonici."""
    targets = [r["target"] for r in rows]
    texts = [r["text"] for r in rows]
    enc = tokenizer(targets, texts, truncation=True, max_length=max_len)
    enc["labels"] = [LABEL2ID[r["label"]] for r in rows]
    return Dataset.from_dict(enc)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    y_true = [ID2LABEL[int(i)] for i in labels]
    y_pred = [ID2LABEL[int(i)] for i in preds]
    return {"macro_f1": compute_macro_f1(y_true, y_pred)["macro_f1"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning stance detector (DeBERTa-v3-base)")
    parser.add_argument("--train-path", default="data/stance/train.jsonl")
    parser.add_argument("--out-dir", default="data/models/stance-deberta-v3-base")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Dry-run: 20 esempi, 1 step.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    set_seed(args.seed)

    rows = load_stance_jsonl(args.train_path)
    train_rows, val_rows = stratified_split(rows, val_frac=0.1, seed=args.seed)
    if args.smoke:
        train_rows, val_rows = train_rows[:20], val_rows[:20]

    # use_fast=False: il fast tokenizer di DeBERTa-v3 richiede `protobuf` per convertire
    # il modello SentencePiece; lo slow tokenizer usa solo `sentencepiece` (già dipendenza)
    # e coerente con src/nlp/stance_detector.py. Salva spm.model, caricabile a runtime.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    train_ds = build_hf_dataset(train_rows, tokenizer, args.max_len)
    val_ds = build_hf_dataset(val_rows, tokenizer, args.max_len)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=args.out_dir + "/checkpoints",
        num_train_epochs=1 if args.smoke else args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        max_steps=1 if args.smoke else -1,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    logger.info(f"Modello salvato in {args.out_dir}")


if __name__ == "__main__":
    main()
