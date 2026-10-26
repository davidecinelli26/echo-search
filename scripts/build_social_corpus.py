"""
Echo-Search | Corpus ibrido (Fase 1) — Ecosistema social
========================================================
Costruisce il corpus `social` (tweet etichettati per stance) per clima e vaccini
da dataset pubblici (raw GitHub, nessuna API Twitter), con schema allineato ai
corpora CC-News + i campi `source_type` e `stance_gold`.

SCELTE ARCHITETTURALI:
- SEPARAZIONE download / logica: le funzioni pure (parse, sample, build) sono
  testabili senza rete; `main()` fa l'I/O.
- CAMPIONAMENTO STRATIFICATO per stance: garantisce la classe AGAINST, quasi
  assente nella cronaca CC-News (misura FT: clima 2.6%, vaccini 2.9%).
- DEDUP per testo: evita i duplicati (RT) — il difetto noto dei corpora news.
- SOCIAL = tier 3 fisso: i dataset non hanno metadati d'autore (registro §5).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import random
import urllib.request
from collections import Counter
from pathlib import Path

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

SOCIAL_SOURCES = {
    "climate_change": {
        "url": "https://raw.githubusercontent.com/edwardcqian/climate_change_sentiment/master/data/sample_data.csv",
        "text_col": "message",
        "label_col": "sentiment",
        "label_map": {"1": "FAVOR", "-1": "AGAINST", "0": "NEUTRAL"},  # "2" (News) scartata
        "target": "climate change",
    },
    "vaccines": {
        "url": "https://raw.githubusercontent.com/sohampoddar26/covid-vax-stance/main/dataset/our_labelled_data.csv",
        "text_col": "tweet",
        "label_col": "majority_label",
        "label_map": {"ProVax": "FAVOR", "AntiVax": "AGAINST", "Neutral": "NEUTRAL"},
        "target": "vaccine safety",
    },
}
_TARGET_PER_CLASS = 333  # tetto per classe (clima abbondante; vaccini limitato dagli AntiVax)


def parse_labeled_rows(csv_text: str, text_col: str, label_col: str, label_map: dict) -> list[dict]:
    """Righe {text, label} mappate; salta label fuori dal map o testo vuoto; dedup per testo."""
    rows, seen = [], set()
    for r in csv.DictReader(io.StringIO(csv_text)):
        raw = (r.get(label_col) or "").strip()
        text = (r.get(text_col) or "").strip()
        if raw not in label_map or not text:
            continue
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "label": label_map[raw]})
    return rows


def stratified_sample(rows: list[dict], n_per_class: int, seed: int = 42) -> list[dict]:
    """Per ogni label campiona min(n_per_class, conteggio). Deterministico a parità di seed."""
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        out.extend(items[: min(n_per_class, len(items))])
    return out


def build_social_docs(rows: list[dict], theme: str, target: str) -> list[dict]:
    """Documenti con schema allineato ai corpora news + source_type/stance_gold."""
    docs = []
    for i, r in enumerate(rows):
        docs.append({
            "doc_id": f"{theme}_social-{i:04d}",
            "topic": theme,
            "title": "",
            "text": r["text"],
            "domain": "twitter.com",
            "url": "",
            "date": "",
            "orientation": "unknown",
            "factual": "unknown",
            "tier": 3,
            "source_type": "social",
            "stance_gold": r["label"],
        })
    return docs


def download_csv(url: str) -> str:
    """Scarica un CSV come testo (UA esplicito, come in build_stance_splits)."""
    req = urllib.request.Request(url, headers={"User-Agent": "EchoSearch/0.1 (academic)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def write_corpus(docs: list[dict], theme: str, out_dir: Path) -> dict:
    path = out_dir / f"{theme}_social.jsonl"
    payload = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
    path.write_text(payload, encoding="utf-8")
    return {
        "count": len(docs),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "by_stance": dict(Counter(d["stance_gold"] for d in docs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Costruisce l'ecosistema social (tweet etichettati)")
    parser.add_argument("--out-dir", default="data/corpora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="30/classe, test rapido")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_target = 30 if args.smoke else _TARGET_PER_CLASS
    manifest = {"sources": {}, "topics": {}}

    for theme, cfg in SOCIAL_SOURCES.items():
        logger.info(f"[{theme}] scarico {cfg['url']}")
        rows = parse_labeled_rows(download_csv(cfg["url"]), cfg["text_col"], cfg["label_col"], cfg["label_map"])
        counts = Counter(r["label"] for r in rows)
        n_per_class = min(n_target, min(counts.values()))
        sampled = stratified_sample(rows, n_per_class, args.seed)
        docs = build_social_docs(sampled, theme, cfg["target"])
        stats = write_corpus(docs, theme, out_dir)
        manifest["sources"][theme] = cfg["url"]
        manifest["topics"][theme] = stats
        logger.info(f"[{theme}] disponibili {dict(counts)} | n/classe={n_per_class} | scritti {stats['count']} | {stats['by_stance']}")

    (out_dir / "MANIFEST_social.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Manifest social scritto in {out_dir / 'MANIFEST_social.json'}")


if __name__ == "__main__":
    main()
