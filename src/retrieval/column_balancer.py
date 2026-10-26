"""
Echo-Search | Tier 2/3 — Bilanciamento delle colonne (stadio 3, deterministico)
===============================================================================
Realizza l'obiettivo anti-echo-chamber. Da un pool ampio di candidati (già
ordinati per pertinenza dal reranker), classifica le stance e riempie le tre
colonne FAVOR / AGAINST / NEUTRAL garantendo un minimo per fazione: se il Top-K
per pura pertinenza è sbilanciato (es. 18 FAVOR, 2 AGAINST), si pesca più in
profondità nel pool per popolare la colonna minoritaria.

È il "self-correction retrieval" della slide 12 reso DETERMINISTICO: niente
loop agentico né LLM orchestratore (scartati nella slide 13), solo selezione
ordinata. La diversity è misurata sull'INTERO pool classificato — riflette la
vera distribuzione delle prospettive nel corpus per quella query.
"""

import logging

from src.evaluation.metrics import compute_stance_diversity

logger = logging.getLogger(__name__)

STANCES = ("FAVOR", "AGAINST", "NEUTRAL")


def balance_columns(
    pool: list[dict],
    stance_detector,
    target: str,
    per_column: int = 5,
) -> dict:
    """
    Bilancia le colonne stance a partire da un pool ordinato per pertinenza.

    Parametri
    ----------
    pool : list[dict]
        Candidati (tipicamente reranked), ordinati per pertinenza DECRESCENTE.
        Ogni dict deve avere "text".
    stance_detector : StanceDetector
        Classificatore di stance (deve esporre classify_batch).
    target : str
        Topic del dibattito.
    per_column : int
        Documenti massimi per colonna. Default 5.

    Returns
    -------
    dict
        {
            "columns": {"FAVOR": [...], "AGAINST": [...], "NEUTRAL": [...]},
            "diversity": {...},          # calcolata sull'intero pool classificato
            "pool_distribution": {...},  # conteggi stance nel pool
        }
    """
    empty = {s: [] for s in STANCES}
    if not pool:
        return {"columns": empty, "diversity": compute_stance_diversity(empty), "pool_distribution": {}}

    stance_results = stance_detector.classify_batch([d["text"] for d in pool], target)

    full = {s: [] for s in STANCES}       # tutto il pool, per la diversity reale
    columns = {s: [] for s in STANCES}    # colonne bilanciate (top per pertinenza)
    for doc, stance in zip(pool, stance_results):
        label = stance["label"]
        enriched = {
            **doc,
            "stance_label": label,
            "stance_confidence": stance["confidence"],
            "stance_scores": stance["scores"],
        }
        full[label].append(enriched)
        # Il pool è già ordinato per pertinenza: i primi che arrivano sono i
        # più pertinenti → la colonna raccoglie i migliori di quella fazione.
        if len(columns[label]) < per_column:
            columns[label].append(enriched)

    diversity = compute_stance_diversity(full)
    pool_distribution = {s: len(full[s]) for s in STANCES}
    logger.info(
        f"Bilanciamento colonne | pool={pool_distribution} -> "
        f"colonne={ {s: len(columns[s]) for s in STANCES} } | "
        f"diversity={diversity['diversity_score']}"
    )
    return {"columns": columns, "diversity": diversity, "pool_distribution": pool_distribution}
