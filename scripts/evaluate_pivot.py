"""
Echo-Search | Corpus ibrido (Fase 3) — Valutazione del pivot
============================================================
Misura ciò che i dati attuali consentono:
- Macro-F1 del modello FT sul GOLD SOCIAL (per tema, dominio tweet nativo),
  dalle label `stance_gold` dei corpora social.
- Stance Diversity COMPARATA news↔social sulle query canoniche (topics.yaml).

Retrieval-NDCG (mancano qrels) e Macro-F1 news (manca il gold pilota) sono
dichiarati come limiti nel report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.evaluation.metrics import compute_macro_f1
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def gold_from_social_corpus(corpus_path: str | Path, target: str) -> list[dict]:
    """Righe {text, target, label} dal campo stance_gold di un corpus social."""
    gold = []
    with Path(corpus_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                gold.append({"text": d["text"], "target": target, "label": d["stance_gold"]})
    return gold


def mean_diversity_by_ecosystem(per_query: list[dict]) -> dict[str, float]:
    """Media dei diversity_score per ecosistema (gestendo ecosistemi assenti in alcune query)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for q in per_query:
        for eco, score in q.items():
            sums[eco] = sums.get(eco, 0.0) + score
            counts[eco] = counts.get(eco, 0) + 1
    return {eco: round(sums[eco] / counts[eco], 4) for eco in sums}


_SOCIAL_THEMES = ("climate_change", "vaccines")   # nucleare: solo news
_MODEL_DIR = "data/models/stance-deberta-v3-base"


def main() -> None:
    parser = argparse.ArgumentParser(description="Valutazione del pivot (stance social + diversity)")
    parser.add_argument("--model-dir", default=_MODEL_DIR)
    parser.add_argument("--topics", default="configs/topics.yaml")
    parser.add_argument("--out", default=f"{_MODEL_DIR}/pivot_eval_report.json")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    from scripts.eval_stance import predict_ft
    from src.app import EchoSearch

    topics = load_config(args.topics)["topics"]
    report: dict = {
        "stance_macro_f1_social": {},
        "diversity_comparata": {},
        "limiti": {
            "retrieval_ndcg": "N/A (richiede annotazione qrels query->doc)",
            "stance_macro_f1_news": "N/A (richiede gold pilota news annotato)",
        },
    }

    # (A) Macro-F1 sul gold social (temi con ecosistema social)
    for theme in _SOCIAL_THEMES:
        target = topics[theme]["target"]
        gold = gold_from_social_corpus(f"data/corpora/{theme}_social.jsonl", target)
        y_pred = predict_ft(args.model_dir, gold, 256, 32)
        res = compute_macro_f1([g["label"] for g in gold], y_pred)
        report["stance_macro_f1_social"][theme] = {
            "macro_f1": res["macro_f1"], "per_class": res["per_class"], "n": len(gold)}
        logger.info(f"[{theme}] Macro-F1 social = {res['macro_f1']} su {len(gold)} tweet")

    # (B) Diversity comparata news<->social sulle query canoniche
    for theme, spec in topics.items():
        app = EchoSearch(topic=theme)
        app.load()
        per_query = []
        for q in spec["queries"]:
            r = app.search(q, target=spec["target"])
            per_query.append({eco: b["diversity"]["diversity_score"] for eco, b in r["ecosystems"].items()})
        report["diversity_comparata"][theme] = {
            "media": mean_diversity_by_ecosystem(per_query),
            "n_query": len(per_query),
        }
        logger.info(f"[{theme}] diversity media = {report['diversity_comparata'][theme]['media']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Report pivot scritto in {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
