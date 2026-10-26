"""
Echo-Search | Fase 1 (1.5) — Split per il training dello Stance Detector
========================================================================
Assembla i dati di stance in un formato unificato (text, target, label) da due
fonti complementari, così il fine-tuning (Fase 3) impara una nozione di stance
generalizzabile:

- VAST (Allaway & McKeown 2020, da GitHub): migliaia di topic diversi, pensato
  per lo ZERO-SHOT su temi non visti → è ciò che fa generalizzare il modello al
  nucleare e ai vaccini, che nessun dataset copre direttamente.
- tweet_eval / SemEval-2016 Task 6 (da HF): 5 target polarizzati (aborto,
  ateismo, CLIMA, femminismo, Hillary) → robustezza su stance nette; il config
  `stance_climate` tocca direttamente uno dei nostri temi.

Le etichette eterogenee vengono normalizzate a FAVOR / AGAINST / NEUTRAL.
Il gold test in-domain sui NOSTRI temi (nucleare/clima/vaccini) NON è qui: va
annotato a mano in Fase 3 (leave-one-topic-out definito in configs/topics.yaml).

Output (in `data/stance/`):
- `train.jsonl`, `test.jsonl` — {text, target, label, source}
- `MANIFEST.json`             — conteggi, distribuzione label, #target

Uso:
    uv run --extra data --no-sync python scripts/build_stance_splits.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

_VAST_BASE = "https://raw.githubusercontent.com/emilyallaway/zero-shot-stance/master/data/VAST"
_VAST_FILES = {"train": "vast_train.csv", "test": "vast_test.csv"}
# VAST label: 0=con(AGAINST), 1=pro(FAVOR), 2=neutral
_VAST_LABEL = {0: "AGAINST", 1: "FAVOR", 2: "NEUTRAL"}

# tweet_eval/SemEval label: 0=none(NEUTRAL), 1=against, 2=favor
_TE_LABEL = {0: "NEUTRAL", 1: "AGAINST", 2: "FAVOR"}
_TE_TARGET = {
    "stance_abortion": "legalization of abortion",
    "stance_atheism": "atheism",
    "stance_climate": "climate change",
    "stance_feminist": "feminist movement",
    "stance_hillary": "Hillary Clinton",
}


def load_vast(split: str) -> list[dict]:
    url = f"{_VAST_BASE}/{_VAST_FILES[split]}"
    req = urllib.request.Request(url, headers={"User-Agent": "EchoSearch/0.1 (academic)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        content = r.read().decode("utf-8", "replace")

    rows = []
    for row in csv.DictReader(io.StringIO(content)):
        text = (row.get("post") or "").strip()
        target = (row.get("topic_str") or "").strip()
        try:
            label = _VAST_LABEL[int(row["label"])]
        except (KeyError, ValueError):
            continue
        if text and target:
            rows.append({"text": text, "target": target, "label": label, "source": "vast"})
    logger.info(f"VAST/{split}: {len(rows)} esempi")
    return rows


def load_tweeteval(split: str) -> list[dict]:
    rows = []
    for config, target in _TE_TARGET.items():
        ds = load_dataset("tweet_eval", config, split=split)
        for r in ds:
            rows.append({
                "text": r["text"].strip(),
                "target": target,
                "label": _TE_LABEL[r["label"]],
                "source": "semeval",
            })
    logger.info(f"tweet_eval/{split}: {len(rows)} esempi ({len(_TE_TARGET)} target)")
    return rows


def write_split(rows: list[dict], path: Path) -> dict:
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(payload, encoding="utf-8")
    return {
        "count": len(rows),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "by_label": dict(Counter(r["label"] for r in rows)),
        "by_source": dict(Counter(r["source"] for r in rows)),
        "n_targets": len({r["target"] for r in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assembla gli split di stance (VAST + SemEval)")
    parser.add_argument("--out-dir", default="data/stance")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": ["vast", "semeval(tweet_eval)"], "splits": {}}

    for split in ("train", "test"):
        rows = load_vast(split) + load_tweeteval(split)
        stats = write_split(rows, out_dir / f"{split}.jsonl")
        manifest["splits"][split] = stats
        logger.info(f"[{split}] {stats['count']} esempi | label={stats['by_label']} | target unici={stats['n_targets']}")

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Manifest stance scritto in {out_dir / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
