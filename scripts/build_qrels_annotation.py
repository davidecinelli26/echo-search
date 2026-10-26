"""
Echo-Search | Fase 7 — Fogli di annotazione dei qrels (rilevanza per il retrieval).
=================================================================================
Per ogni query canonica (topics.yaml) costruisce il POOL dei documenti da giudicare
e produce un CSV per annotatore, con la colonna `relevance` da compilare a mano
(0 = non pertinente, 1 = marginale, 2 = pienamente pertinente). I qrels compilati
alimentano NDCG@10 / MAP / Recall@20 (src/evaluation/metrics.py).

SCELTE ARCHITETTURALI:
----------------------
- POOLING MULTI-RUN (Cranfield): il pool e' l'UNIONE dei top-K di TUTTI i run che
  verranno confrontati (BM25, denso, RRF, fusione, pool del reranker). Pescare dal
  solo run di fusione renderebbe non giudicati — e quindi conteggiati come non
  rilevanti — i documenti che solo le baseline recuperano, falsando l'ablazione a
  loro sfavore. E' l'errore di pool bias piu' comune.
- ORDINE RANDOMIZZATO E RANGO NASCOSTO: entro ciascuna query i documenti sono
  mescolati con seed fisso e le colonne rank/score NON compaiono nel foglio. Un
  annotatore che vede "rank 1" e' primato a giudicarlo rilevante (position bias).
  La provenienza per run e' salvata a parte, in {topic}_pool.json.
- UN FOGLIO PER ANNOTATORE, identico: la doppia annotazione richiede indipendenza.
  Due colonne nello stesso file si contaminerebbero a vicenda e il kappa sarebbe
  gonfiato.
- ESTRATTO AL CONFINE DI FRASE: il testo e' troncato su frase intera (riusa
  src/utils/text.split_into_sentences), non a meta' parola.

Uso (serve la GPU per il ramo denso):
    uv run --no-sync --with "torch==2.2.2" --index cu121 python scripts/build_qrels_annotation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.utils.text import split_into_sentences

logger = logging.getLogger(__name__)

# `nota` e' libera: serve in fase di adjudicazione delle divergenze.
_FIELDS = ["query", "doc_id", "title", "text", "relevance", "nota"]
_SEED = 42


# ---------------------------------------------------------------------------
# Logica pura (testabile senza indici ne' modelli)
# ---------------------------------------------------------------------------

def excerpt(text: str, max_chars: int) -> str:
    """
    Estratto del documento troncato al confine di frase entro `max_chars`.

    Gli a-capo sono appiattiti (una cella CSV multi-riga e' illeggibile in Excel).
    Se la prima frase eccede gia' il limite si taglia al confine di parola.
    """
    flat = " ".join((text or "").split())
    if len(flat) <= max_chars:
        return flat

    kept: list[str] = []
    used = 0
    for sentence in split_into_sentences(flat):
        if used + len(sentence) + 1 > max_chars:
            break
        kept.append(sentence)
        used += len(sentence) + 1

    if not kept:
        return flat[:max_chars].rsplit(" ", 1)[0] + " [...]"
    return " ".join(kept) + " [...]"


def union_pool(runs: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Unione dei run: doc_id -> elenco dei run che l'hanno recuperato.

    E' il cuore del pooling: un documento entra nel pool se ALMENO UN run in esame
    lo propone. L'ordine dei run nel valore preserva la provenienza, utile per
    analizzare a posteriori quanti rilevanti fossero esclusivi di un solo run.
    """
    provenance: dict[str, list[str]] = {}
    for run_name, doc_ids in runs.items():
        for doc_id in doc_ids:
            provenance.setdefault(doc_id, [])
            if run_name not in provenance[doc_id]:
                provenance[doc_id].append(run_name)
    return provenance


def shuffled_ids(doc_ids, seed: int = _SEED) -> list[str]:
    """Ordine di presentazione randomizzato ma riproducibile (rango nascosto)."""
    ids = sorted(doc_ids)  # base deterministica: l'insieme puo' arrivare non ordinato
    random.Random(seed).shuffle(ids)
    return ids


def build_rows(provenance: dict[str, list[str]], query: str, doc_meta: dict,
               max_chars: int, seed: int = _SEED) -> list[dict]:
    """Righe del foglio per una query, in ordine randomizzato e senza rango."""
    rows = []
    for doc_id in shuffled_ids(provenance, seed):
        meta = doc_meta.get(doc_id, {})
        rows.append({
            "query": query,
            "doc_id": doc_id,
            "title": " ".join((meta.get("title") or "").split()),
            "text": excerpt(meta.get("text", ""), max_chars),
            "relevance": "",   # <-- da compilare: 0 / 1 / 2
            "nota": "",
        })
    return rows


# ---------------------------------------------------------------------------
# I/O e retrieval
# ---------------------------------------------------------------------------

def load_doc_meta(corpus_path: Path) -> dict[str, dict]:
    """Mappa doc_id -> documento completo (serve titolo e testo integrale)."""
    meta = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                meta[d["doc_id"]] = d
    return meta


def runs_for_query(retriever: HybridRetriever, query: str, depth: int,
                   rerank_pool: int) -> dict[str, list[str]]:
    """I doc_id recuperati da ciascun run in esame per questa query."""
    fused = [r["doc_id"] for r in retriever.search(query, top_k=max(depth, rerank_pool))]
    return {
        "bm25": [r["doc_id"] for r in retriever.sparse.search(query, top_k=depth)],
        "dense": [r["doc_id"] for r in retriever.dense.search(query, top_k=depth)],
        "rrf": [r["doc_id"] for r in retriever.search_rrf_baseline(query, top_k=depth)],
        "fusion": fused[:depth],
        # il run con reranking riordina i primi `rerank_pool` della fusione: il suo
        # top-k finale e' un sottoinsieme di questi, quindi vanno tutti giudicati
        "rerank_pool": fused[:rerank_pool],
    }


def write_sheet(rows: list[dict], path: Path) -> None:
    """CSV in utf-8-sig: senza BOM Excel su Windows sbaglia la codifica."""
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera i fogli qrels da annotare")
    parser.add_argument("--topics", default="configs/topics.yaml")
    parser.add_argument("--corpora-dir", default="data/corpora")
    parser.add_argument("--index-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/gold")
    parser.add_argument("--ecosystem", default="news", help="Ecosistema del corpus da valutare")
    parser.add_argument("--depth", type=int, default=20,
                        help="Profondita' del pool per ciascun run (default 20 -> Recall@20)")
    parser.add_argument("--rerank-pool", type=int, default=20,
                        help="Candidati che il run col reranker riordina (default 20)")
    parser.add_argument("--max-chars", type=int, default=1200,
                        help="Lunghezza massima dell'estratto mostrato all'annotatore")
    parser.add_argument("--annotators", default="davide,giuseppe",
                        help="Nomi degli annotatori, separati da virgola: un foglio identico a testa")
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    topics = load_config(args.topics)["topics"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]

    grand_total = 0
    for topic, spec in topics.items():
        eco_dir = Path(args.index_dir) / topic / args.ecosystem
        if not (eco_dir / "bm25_index.pkl").exists():
            logger.info("[%s] nessun indice '%s' -> salto", topic, args.ecosystem)
            continue

        corpus_path = Path(args.corpora_dir) / f"{topic}.jsonl"
        if not corpus_path.exists():
            logger.warning("[%s] corpus %s assente: niente titoli -> salto", topic, corpus_path)
            continue
        doc_meta = load_doc_meta(corpus_path)

        retr = HybridRetriever.load_from_disk(str(eco_dir), config_path=args.config,
                                              skip_dense=False)

        rows: list[dict] = []
        pool_provenance: dict[str, dict[str, list[str]]] = {}
        for query in spec.get("queries", []):
            runs = runs_for_query(retr, query, args.depth, args.rerank_pool)
            provenance = union_pool(runs)
            pool_provenance[query] = provenance
            rows.extend(build_rows(provenance, query, doc_meta, args.max_chars, args.seed))

        for annotator in annotators:
            write_sheet(rows, out_dir / f"qrels_{topic}_{annotator}.csv")

        (out_dir / f"qrels_{topic}_pool.json").write_text(
            json.dumps({
                "ecosystem": args.ecosystem,
                "depth": args.depth,
                "rerank_pool": args.rerank_pool,
                "seed": args.seed,
                "runs": ["bm25", "dense", "rrf", "fusion", "rerank_pool"],
                "provenance": pool_provenance,
            }, indent=2, ensure_ascii=False), encoding="utf-8")

        grand_total += len(rows)
        exclusive = sum(1 for q in pool_provenance.values()
                        for runs_of in q.values() if len(runs_of) == 1)
        logger.info("[%s] %d coppie (query,doc) su %d query | %d recuperate da un solo run "
                    "-> qrels_%s_{%s}.csv",
                    topic, len(rows), len(pool_provenance), exclusive, topic,
                    ",".join(annotators))

    logger.info("Totale: %d coppie da giudicare, %d fogli per annotatore (%s)",
                grand_total, len(annotators), ", ".join(annotators))


if __name__ == "__main__":
    main()
