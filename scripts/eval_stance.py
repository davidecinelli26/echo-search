"""
Echo-Search | Fase 3 — Valutazione dello Stance Detector
========================================================
Macro-F1 sul test held-out (VAST+SemEval) con breakdown per source e per classe,
confronto FT vs baseline zero-shot, hook per il gold set (data/gold/) e per i
benchmark esterni citabili. Riusa src/evaluation/metrics.py::compute_macro_f1.

SCELTE ARCHITETTURALI:
----------------------
- VAST-test è zero-shot (topic disgiunti dal train): il suo Macro-F1 è la prova
  di generalizzazione a temi mai visti → riportato come sotto-breakdown.
- Il confronto FT vs baseline zero-shot quantifica il guadagno del fine-tuning
  (anticipa l'ablation 7.2) usando lo STESSO protocollo sullo stesso test.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluation.metrics import compute_macro_f1
from src.nlp.stance_data import ID2LABEL, load_stance_jsonl

logger = logging.getLogger(__name__)


def evaluate_by_source(rows: list[dict], y_pred: list[str]) -> dict:
    """Macro-F1 complessivo e per source, dato y_true nelle rows e y_pred."""
    if len(rows) != len(y_pred):
        raise ValueError(f"len(rows)={len(rows)} != len(y_pred)={len(y_pred)}")
    y_true = [r["label"] for r in rows]
    report = {"overall": compute_macro_f1(y_true, y_pred), "by_source": {}}
    for src in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        report["by_source"][src] = compute_macro_f1(
            [y_true[i] for i in idx], [y_pred[i] for i in idx]
        )
    return report


def predict_ft(model_dir: str, rows: list[dict], max_len: int, batch_size: int) -> list[str]:
    """Predizioni del modello fine-tuned (coppie target/text)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use_fast=False: il tokenizer FT è salvato come SentencePiece (spm.model); il fast
    # richiederebbe protobuf. Coerente con train_stance.py e stance_detector.py.
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    preds: list[str] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        enc = tok(
            [r["target"] for r in batch], [r["text"] for r in batch],
            truncation=True, max_length=max_len, padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(ID2LABEL[int(i)] for i in logits.argmax(dim=-1).cpu().numpy())
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Valutazione stance FT vs baseline")
    parser.add_argument("--test-path", default="data/stance/test.jsonl")
    parser.add_argument("--model-dir", default="data/models/stance-deberta-v3-base")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default="data/models/stance-deberta-v3-base/eval_report.json")
    parser.add_argument("--baseline", action="store_true",
                        help="Valuta anche il baseline zero-shot (src/nlp/stance_detector).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    rows = load_stance_jsonl(args.test_path)
    report = {"finetuned": evaluate_by_source(rows, predict_ft(
        args.model_dir, rows, args.max_len, args.batch_size))}

    if args.baseline:
        # Baseline zero-shot: usa il backend attuale, un target per volta.
        from src.nlp.stance_detector import StanceDetector
        det = StanceDetector(model_name="cross-encoder/nli-roberta-base")
        by_target: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            by_target.setdefault(r["target"], []).append(i)
        y_pred = [None] * len(rows)
        for target, idx in by_target.items():
            docs = [rows[i]["text"] for i in idx]
            for j, res in zip(idx, det.classify_batch(docs, target)):
                y_pred[j] = res["label"]
        report["baseline_zeroshot"] = evaluate_by_source(rows, y_pred)

    # Hook gold set (in-domain per tema), se annotato.
    report["gold"] = {}
    for theme in ("nuclear_energy", "climate_change", "vaccines"):
        p = Path(f"data/gold/stance_{theme}.jsonl")
        if p.exists():
            g = load_stance_jsonl(p)
            report["gold"][theme] = evaluate_by_source(g, predict_ft(
                args.model_dir, g, args.max_len, args.batch_size))["overall"]
        else:
            report["gold"][theme] = "N/A (gold set non ancora annotato)"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Report scritto in {args.out}")
    print(json.dumps({k: (v.get("overall") if isinstance(v, dict) and "overall" in v else v)
                      for k, v in report.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
