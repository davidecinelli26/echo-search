"""
Echo-Search | Script Offline (Fase 2) — Costruzione degli indici PER TEMA
==========================================================================
Legge i corpora congelati (data/corpora/{tema}.jsonl, prodotti da
build_corpora.py) e costruisce per OGNI tema un Inverted Index BM25 (Sparse)
+ un Vector DB FAISS (Dense, BGE).

Indice per-tema (scelta architetturale Fase 2): l'utente interroga un tema
divisivo specifico, quindi ogni corpus è indicizzato e valutato separatamente.

Output (per ogni tema):
    data/processed/{tema}/bm25_index.pkl
    data/processed/{tema}/faiss_index.bin (+ .meta.pkl)

Uso:
    uv run --extra data --no-sync python scripts/index_corpus.py
    uv run --no-sync python scripts/index_corpus.py --skip-dense   # solo BM25, rapido
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger("index_corpus")


def load_topic_corpus(path: Path) -> tuple[list[str], list[str]]:
    """Carica (testi, doc_id) da un corpus JSONL per-tema."""
    docs: list[str] = []
    ids: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append(d["text"])
            ids.append(d["doc_id"])
    return docs, ids


_ECOSYSTEMS = ("news", "social")


def iter_corpus_specs(topics: dict, corpora_dir: Path) -> list[dict]:
    """Enumera le coppie (tema × ecosistema) con corpus presente su disco.

    news  -> {tema}.jsonl        social -> {tema}_social.jsonl
    Il nucleare (senza file social) produce solo la spec news.
    """
    specs: list[dict] = []
    for topic_id in topics:
        candidates = {
            "news": corpora_dir / f"{topic_id}.jsonl",
            "social": corpora_dir / f"{topic_id}_social.jsonl",
        }
        for eco in _ECOSYSTEMS:
            path = candidates[eco]
            if path.exists():
                specs.append({
                    "topic": topic_id,
                    "ecosystem": eco,
                    "corpus_path": path,
                    "out_name": f"{topic_id}/{eco}",
                })
    return specs


def index_topic(
    label: str,
    corpus_path: Path,
    out: Path,
    config_path: str,
    skip_dense: bool,
) -> None:
    docs, ids = load_topic_corpus(corpus_path)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{label}] {len(docs)} documenti -> {out}")

    sparse = SparseRetriever(config_path=config_path)
    sparse.index(docs, ids)
    sparse.save_index(str(out / "bm25_index.pkl"))

    if skip_dense:
        logger.info(f"[{label}] dense skippato (--skip-dense)")
        return

    dense = DenseRetriever(config_path=config_path)
    dense.index(docs, ids)
    dense.save_index(str(out / "faiss_index.bin"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo-Search — indicizzazione offline per-tema")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--topics", default="configs/topics.yaml")
    parser.add_argument("--corpora-dir", default="data/corpora")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--skip-dense", action="store_true", help="Solo BM25 (rapido, no FAISS)")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    topics = load_config(args.topics)["topics"]
    corpora_dir = Path(args.corpora_dir)
    out_dir = Path(args.out_dir)

    specs = iter_corpus_specs(topics, corpora_dir)
    logger.info(f"Indicizzo {len(specs)} corpora: {[s['out_name'] for s in specs]}")
    for spec in specs:
        index_topic(
            label=spec["out_name"],
            corpus_path=spec["corpus_path"],
            out=out_dir / spec["out_name"],
            config_path=args.config,
            skip_dense=args.skip_dense,
        )

    logger.info("Indicizzazione per (tema × ecosistema) completata.")


if __name__ == "__main__":
    main()
