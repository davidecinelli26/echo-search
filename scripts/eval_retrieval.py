"""
Echo-Search | Valutazione del retrieval — ablazione a 6 sistemi su qrels annotati.
=================================================================================
Calcola NDCG@10, MAP e Recall@20 per i sei sistemi dell'ablazione, su una o più
serie di qrels, e verifica se l'ORDINAMENTO dei sistemi dipende da chi ha
giudicato.

I sei sistemi:
    1. bm25            solo ramo sparso
    2. dense           solo ramo denso (bge + FAISS)
    3. rrf             Reciprocal Rank Fusion (baseline senza parametri)
    4. fusion_default  fusione pesata con alpha da config.yaml
    5. fusion_tuned    fusione pesata con alpha tarato in leave-one-topic-out
    6. fusion_rerank   fusione (alpha default) + reranking cross-encoder

SCELTE METODOLOGICHE:
- API PUBBLICA, NON REIMPLEMENTAZIONE: la fusione a un dato alpha si ottiene
  impostando retriever.alpha/beta e chiamando search(). Si misura il sistema
  vero, non una copia della sua formula che potrebbe divergere.
- VALUTAZIONE ALLA PROFONDITA' DEL POOL: i run sono troncati a `--depth` (default
  20), che è la profondità a cui il pool è stato giudicato. Andare più in
  profondità aggiungerebbe solo documenti non giudicati, che per la convenzione di
  Cranfield contano come non rilevanti: non cambierebbe le metriche ma darebbe
  l'illusione di un run più completo. Il MAP è quindi un MAP@depth ed è etichettato
  come tale.
- TARATURA IN LEAVE-ONE-TOPIC-OUT: con 11 query, scegliere alpha sullo stesso
  insieme su cui si riporta il risultato sarebbe sovradattamento garantito. Per
  ogni tema, alpha è tarato sulle query degli ALTRI temi e valutato sulle proprie.
  Si riportano gli alpha selezionati per fold e la loro stabilità.
- IL RERANKER RIORDINA I CANDIDATI CHE IL POOL COPRE: il pool è stato costruito
  con i primi `--rerank-pool` della fusione di default (vedi
  build_qrels_annotation.py). Riordinare più candidati di così introdurrebbe
  documenti non giudicati nel top-k del run.
- COPERTURA DEI GIUDIZI RIPORTATA: per ogni sistema si riporta la frazione del
  top-k effettivamente giudicata. Se scende sotto ~0.95 il confronto è viziato dal
  pool e va dichiarato.

Uso:
    uv run --no-sync --with "torch==2.2.2" --index cu121 python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --annotators davide,giuseppe
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src.evaluation.metrics import compute_ndcg, average_precision, compute_recall_at_k
from src.evaluation.qrels import load_qrels_for_topics, relevant_count
from src.utils.config import load_config, load_section
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

SYSTEMS = ["bm25", "dense", "rrf", "fusion_default", "fusion_tuned", "fusion_rerank"]
ALPHA_GRID = [round(0.1 * i, 1) for i in range(1, 10)]


# ---------------------------------------------------------------------------
# Logica pura: metriche su run già calcolati
# ---------------------------------------------------------------------------

def score_run(run: list[str], qrels_q: dict[str, int], k_ndcg: int, k_recall: int) -> dict:
    """Metriche di un singolo run (lista ordinata di doc_id) contro i qrels di una query."""
    as_results = [{"doc_id": d} for d in run]
    judged = sum(1 for d in run[:k_recall] if d in qrels_q)
    return {
        "ndcg": compute_ndcg(as_results, qrels_q, k=k_ndcg),
        "ap": round(average_precision(as_results, qrels_q), 4),
        "recall": compute_recall_at_k(as_results, qrels_q, k=k_recall),
        "judged_frac": round(judged / max(1, len(run[:k_recall])), 4),
    }


def tune_alpha_loto(ndcg_by_alpha: dict[str, dict[float, dict[str, float]]]) -> dict[str, float]:
    """
    Alpha ottimale per ogni tema, tarato sulle query degli ALTRI temi.

    `ndcg_by_alpha` è {tema: {alpha: {query: ndcg}}}. Restituisce {tema: alpha}.
    A parità di NDCG medio vince l'alpha più piccolo (scelta deterministica).
    """
    chosen: dict[str, float] = {}
    for held_out in ndcg_by_alpha:
        best_alpha, best_score = None, -1.0
        for alpha in sorted(ALPHA_GRID):
            scores = [
                ndcg
                for tema, per_alpha in ndcg_by_alpha.items()
                if tema != held_out
                for ndcg in per_alpha[alpha].values()
            ]
            mean = float(np.mean(scores)) if scores else 0.0
            if mean > best_score + 1e-9:
                best_alpha, best_score = alpha, mean
        chosen[held_out] = best_alpha
    return chosen


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Tau-b di Kendall fra due liste di punteggi appaiate (n piccolo: O(n^2) va bene)."""
    n = len(a)
    conc = disc = tie_a = tie_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                tie_a += 1
            elif db == 0:
                tie_b += 1
            elif (da > 0) == (db > 0):
                conc += 1
            else:
                disc += 1
    denom = np.sqrt((conc + disc + tie_a) * (conc + disc + tie_b))
    return round(float((conc - disc) / denom), 4) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Raccolta dei run (richiede indici e modelli)
# ---------------------------------------------------------------------------

def collect_runs(topics: dict, index_dir: Path, ecosystem: str, config_path: str,
                 depth: int, rerank_pool: int, default_alpha: float) -> tuple[dict, dict]:
    """
    Esegue tutti i sistemi su tutte le query, un tema per volta.

    Returns
    -------
    (runs, fusion_by_alpha)
        runs = {sistema: {tema: {query: [doc_id]}}} per i sistemi non parametrici
        fusion_by_alpha = {tema: {alpha: {query: [doc_id]}}}
    """
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker(config_path=config_path)
    runs = {s: {} for s in ("bm25", "dense", "rrf", "fusion_default", "fusion_rerank")}
    fusion_by_alpha: dict[str, dict[float, dict[str, list[str]]]] = {}

    for topic, spec in topics.items():
        eco_dir = index_dir / topic / ecosystem
        if not (eco_dir / "bm25_index.pkl").exists():
            logger.info("[%s] nessun indice '%s' -> salto", topic, ecosystem)
            continue
        queries = spec.get("queries", [])
        retr = HybridRetriever.load_from_disk(str(eco_dir), config_path=config_path)
        for s in runs:
            runs[s][topic] = {}
        fusion_by_alpha[topic] = {a: {} for a in ALPHA_GRID}

        for q in queries:
            runs["bm25"][topic][q] = [r["doc_id"] for r in retr.sparse.search(q, top_k=depth)]
            runs["dense"][topic][q] = [r["doc_id"] for r in retr.dense.search(q, top_k=depth)]
            runs["rrf"][topic][q] = [r["doc_id"] for r in retr.search_rrf_baseline(q, top_k=depth)]

            # fusione di default: serve intera fino a rerank_pool per alimentare il reranker
            retr.alpha, retr.beta = default_alpha, round(1.0 - default_alpha, 4)
            fused = retr.search(q, top_k=max(depth, rerank_pool))
            runs["fusion_default"][topic][q] = [r["doc_id"] for r in fused[:depth]]
            reranked = reranker.rerank(q, fused[:rerank_pool], top_n=rerank_pool)
            runs["fusion_rerank"][topic][q] = [r["doc_id"] for r in reranked[:depth]]

            for alpha in ALPHA_GRID:
                retr.alpha, retr.beta = alpha, round(1.0 - alpha, 4)
                fusion_by_alpha[topic][alpha][q] = [
                    r["doc_id"] for r in retr.search(q, top_k=depth)
                ]

        logger.info("[%s] %d query: run raccolti", topic, len(queries))
        del retr
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return runs, fusion_by_alpha


# ---------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------

def evaluate(runs: dict, fusion_by_alpha: dict, qrels: dict, k_ndcg: int, k_recall: int) -> dict:
    """Metriche aggregate per sistema, più il dettaglio per tema e gli alpha scelti."""
    temi = [t for t in qrels if t in fusion_by_alpha]

    # NDCG per (tema, alpha, query): serve alla taratura LOTO
    ndcg_by_alpha = {
        t: {
            a: {
                q: compute_ndcg([{"doc_id": d} for d in run], qrels[t][q], k=k_ndcg)
                for q, run in per_query.items() if q in qrels[t]
            }
            for a, per_query in fusion_by_alpha[t].items()
        }
        for t in temi
    }
    chosen_alpha = tune_alpha_loto(ndcg_by_alpha)

    all_runs = dict(runs)
    all_runs["fusion_tuned"] = {
        t: fusion_by_alpha[t][chosen_alpha[t]] for t in temi
    }

    # curva NDCG(alpha): stesso filtro dei sistemi (query con almeno un rilevante),
    # altrimenti la curva e i valori in tabella non sarebbero confrontabili.
    alpha_curve = {
        str(a): round(float(np.mean([
            ndcg
            for t in temi
            for q, ndcg in ndcg_by_alpha[t][a].items()
            if relevant_count(qrels[t][q]) > 0
        ])), 4)
        for a in sorted(ALPHA_GRID)
    }

    out = {"per_system": {}, "chosen_alpha": chosen_alpha,
           "alpha_curve": alpha_curve, "per_topic": {}}
    for system in SYSTEMS:
        rows, per_topic = [], {}
        for t in temi:
            topic_rows = [
                score_run(run, qrels[t][q], k_ndcg, k_recall)
                for q, run in all_runs[system].get(t, {}).items()
                if q in qrels[t] and relevant_count(qrels[t][q]) > 0
            ]
            rows.extend(topic_rows)
            if topic_rows:
                per_topic[t] = {
                    m: round(float(np.mean([r[m] for r in topic_rows])), 4)
                    for m in ("ndcg", "ap", "recall")
                }
        out["per_system"][system] = {
            "n_queries": len(rows),
            f"ndcg@{k_ndcg}": round(float(np.mean([r["ndcg"] for r in rows])), 4),
            f"map@{k_recall}": round(float(np.mean([r["ap"] for r in rows])), 4),
            f"recall@{k_recall}": round(float(np.mean([r["recall"] for r in rows])), 4),
            "judged_frac": round(float(np.mean([r["judged_frac"] for r in rows])), 4),
        }
        out["per_topic"][system] = per_topic
    return out


def print_table(name: str, res: dict, k_ndcg: int, k_recall: int) -> None:
    print(f"\n{'='*78}\nQRELS: {name}\n{'='*78}")
    print(f"{'#':<2} {'sistema':<16} {'NDCG@'+str(k_ndcg):>9} {'MAP@'+str(k_recall):>9} "
          f"{'Recall@'+str(k_recall):>11} {'giudicati':>10}")
    print("-" * 78)
    for i, s in enumerate(SYSTEMS, 1):
        m = res["per_system"][s]
        print(f"{i:<2} {s:<16} {m[f'ndcg@{k_ndcg}']:>9.4f} {m[f'map@{k_recall}']:>9.4f} "
              f"{m[f'recall@{k_recall}']:>11.4f} {m['judged_frac']:>10.3f}")
    print(f"\nalpha scelti in LOTO: {res['chosen_alpha']}")
    curva = "  ".join(f"{a}:{v:.4f}" for a, v in res["alpha_curve"].items())
    print(f"curva NDCG@{k_ndcg}(alpha): {curva}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ablazione del retrieval su qrels annotati")
    p.add_argument("--topics", default="configs/topics.yaml")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--index-dir", default="data/processed")
    p.add_argument("--gold-dir", default="data/gold")
    p.add_argument("--ecosystem", default="news")
    p.add_argument("--annotators", default="davide,giuseppe",
                   help="serie di qrels da usare, separate da virgola")
    p.add_argument("--depth", type=int, default=20, help="profondita' dei run = profondita' del pool")
    p.add_argument("--rerank-pool", type=int, default=20)
    p.add_argument("--k-ndcg", type=int, default=10)
    p.add_argument("--out", default="data/gold/retrieval_eval_report.json")
    args = p.parse_args()
    setup_logging(logging.INFO)

    topics = load_config(args.topics)["topics"]
    default_alpha = load_section(args.config, "fusion").get("alpha", 0.45)
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]

    runs, fusion_by_alpha = collect_runs(
        topics, Path(args.index_dir), args.ecosystem, args.config,
        args.depth, args.rerank_pool, default_alpha,
    )

    report = {"depth": args.depth, "rerank_pool": args.rerank_pool,
              "k_ndcg": args.k_ndcg, "default_alpha": default_alpha,
              "ecosystem": args.ecosystem, "results": {}}
    for annotator in annotators:
        qrels = load_qrels_for_topics(topics, annotator, args.gold_dir)
        if not qrels:
            logger.warning("nessun qrels per '%s' -> salto", annotator)
            continue
        res = evaluate(runs, fusion_by_alpha, qrels, args.k_ndcg, args.depth)
        report["results"][annotator] = res
        print_table(annotator, res, args.k_ndcg, args.depth)

    # stabilita' dell'ordinamento fra serie di qrels diverse
    if len(report["results"]) > 1:
        names = list(report["results"])
        key = f"ndcg@{args.k_ndcg}"
        vecs = {n: [report["results"][n]["per_system"][s][key] for s in SYSTEMS] for n in names}
        print(f"\n{'='*78}\nSTABILITA' DELL'ORDINAMENTO FRA ANNOTATORI\n{'='*78}")
        for n in names:
            order = sorted(SYSTEMS, key=lambda s: -report["results"][n]["per_system"][s][key])
            print(f"  {n:<10} {' > '.join(order)}")
        tau = kendall_tau(vecs[names[0]], vecs[names[1]])
        report["kendall_tau_ndcg"] = tau
        print(f"\n  tau-b di Kendall sui {len(SYSTEMS)} sistemi ({key}): {tau}")
        print("  tau = 1.0 -> ordinamento identico: la conclusione non dipende da chi ha giudicato")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport salvato in {args.out}")


if __name__ == "__main__":
    main()
