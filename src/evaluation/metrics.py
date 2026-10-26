"""
Echo-Search | Metriche di valutazione
=====================================
Tutte le metriche del protocollo sperimentale, in un unico modulo di funzioni
PURE: nessun accesso a disco o a rete, nessun modello. È la ragione per cui la
suite di test le copre per intero in millisecondi, ed è anche la ragione per cui
il caricamento dei giudizi di rilevanza vive altrove (`qrels.py`).

Tre famiglie.

RANKING — NDCG@k, average precision, Recall@k. Sono le metriche standard di
Cranfield; l'NDCG è quella riportata come primaria perché è l'unica definita su
giudizi GRADUATI, e la scala di rilevanza del progetto ha tre livelli.

ACCORDO FRA ANNOTATORI — il κ di Cohen, nelle varianti pesate. Su una scala
ordinale la variante quadratica è quella corretta: la non pesata tratterebbe un
disaccordo fra 0 e 1 e uno fra 0 e 2 come lo stesso errore.

STANCE — Macro-F1 (ogni classe pesa uguale, indispensabile con classi
sbilanciate) e la Stance Diversity, che è la metrica originale del progetto:
entropia di Shannon normalizzata su H_max = log2(3), quindi 0 = echo chamber
perfetta e 1 = distribuzione uniforme.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# IR Metrics
# ============================================================

def compute_ndcg(
    results: list[dict],
    relevance_judgments: dict[str, int],
    k: int = 10,
) -> float:
    """
    Calcola NDCG@K per una singola query.

    Parametri
    ----------
    results : list[dict]
        Lista di documenti recuperati ordinati per rank (output del retriever).
        Ogni dict deve avere "doc_id".
    relevance_judgments : dict[str, int]
        {doc_id: relevance_score} — giudizi di rilevanza (tipicamente 0, 1, 2).
    k : int
        Cutoff per il calcolo. Default 10.

    Returns
    -------
    float
        NDCG@K ∈ [0, 1]. 1.0 = ranking perfetto.
    """
    # DCG del ranking restituito
    dcg = 0.0
    for rank, doc in enumerate(results[:k], start=1):
        rel = relevance_judgments.get(doc["doc_id"], 0)
        dcg += rel / math.log2(rank + 1)  # log2(pos + 1), base 2

    # IDCG: ordinamento ideale (relevance decrescente)
    ideal_rels = sorted(relevance_judgments.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
        if rel > 0
    )

    if idcg == 0:
        return 0.0
    return round(dcg / idcg, 4)


def compute_recall_at_k(
    results: list[dict],
    relevance_judgments: dict[str, int],
    k: int = 20,
) -> float:
    """
    Calcola Recall@K: frazione dei documenti rilevanti recuperati nei Top-K.
    """
    total_relevant = sum(1 for rel in relevance_judgments.values() if rel > 0)
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        1 for doc in results[:k]
        if relevance_judgments.get(doc["doc_id"], 0) > 0
    )

    return round(retrieved_relevant / total_relevant, 4)


def average_precision(results: list[dict], relevance: dict[str, int]) -> float:
    """AP per una singola query."""
    precisions = []
    num_relevant = 0

    for rank, doc in enumerate(results, start=1):
        if relevance.get(doc["doc_id"], 0) > 0:
            num_relevant += 1
            precisions.append(num_relevant / rank)

    if not precisions:
        return 0.0
    return sum(precisions) / sum(1 for rel in relevance.values() if rel > 0)


# ============================================================
# Inter-Annotator Agreement
# ============================================================

_KAPPA_WEIGHTS = ("unweighted", "linear", "quadratic")


def compute_cohen_kappa(
    y1: list,
    y2: list,
    labels: Optional[list] = None,
    weights: str = "unweighted",
) -> dict:
    """
    Cohen's kappa fra due annotatori, con pesi opzionali per scale ordinali.

    L'accordo osservato va corretto per l'accordo atteso per caso:
    kappa = (p_o - p_e) / (1 - p_e).

    SCELTA DEI PESI:
    - "unweighted": ogni disaccordo pesa uguale. Corretto per scale NOMINALI.
    - "quadratic":  w_ij = 1 - ((i-j)/(K-1))^2. Corretto per scale ORDINALI, dove
      confondere 0 con 2 è un errore più grave che confondere 0 con 1. Sulla scala
      di rilevanza 0/1/2 dei qrels è questa la variante da riportare: la versione
      non pesata tratta "0 vs 1" e "0 vs 2" come lo stesso errore, che è falso.
    - "linear":     w_ij = 1 - |i-j|/(K-1). Penalizza la distanza linearmente.

    Parametri
    ----------
    y1, y2 : list
        Etichette dei due annotatori, allineate per posizione.
    labels : list, opzionale
        Ordine delle etichette. Per i pesi ordinali l'ordine CONTA: deve essere
        quello della scala. Default: ordinamento delle etichette osservate.
    weights : str
        "unweighted" (default) | "linear" | "quadratic".

    Returns
    -------
    dict
        {"kappa", "observed_agreement", "expected_agreement", "n", "labels",
         "confusion"}. Con accordo atteso pari a 1 (un solo valore usato da
         entrambi) kappa non è definito e vale float("nan").
    """
    if len(y1) != len(y2):
        raise ValueError(f"liste di lunghezza diversa: {len(y1)} vs {len(y2)}")
    if not y1:
        raise ValueError("nessuna osservazione")
    if weights not in _KAPPA_WEIGHTS:
        raise ValueError(f"weights deve essere uno di {_KAPPA_WEIGHTS}, ricevuto {weights!r}")

    if labels is None:
        labels = sorted(set(y1) | set(y2))
    unknown = (set(y1) | set(y2)) - set(labels)
    if unknown:
        raise ValueError(f"etichette non dichiarate in labels: {sorted(unknown)}")

    n = len(y1)
    k = len(labels)
    index = {lab: i for i, lab in enumerate(labels)}

    confusion = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        confusion[index[a], index[b]] += 1

    if k == 1:
        # entrambi hanno usato una sola etichetta: accordo perfetto ma non informativo
        return {"kappa": float("nan"), "observed_agreement": 1.0,
                "expected_agreement": 1.0, "n": n, "labels": list(labels),
                "confusion": confusion.astype(int).tolist()}

    w = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            if weights == "unweighted":
                w[i, j] = 1.0 if i == j else 0.0
            elif weights == "linear":
                w[i, j] = 1.0 - abs(i - j) / (k - 1)
            else:  # quadratic
                w[i, j] = 1.0 - ((i - j) / (k - 1)) ** 2

    p_observed = float((w * confusion).sum() / n)
    # marginali: distribuzione indipendente dei due annotatori
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / (n * n)
    p_expected = float((w * expected).sum())

    kappa = float("nan") if abs(1.0 - p_expected) < 1e-12 else (p_observed - p_expected) / (1.0 - p_expected)

    return {
        "kappa": round(kappa, 4) if kappa == kappa else float("nan"),
        "observed_agreement": round(p_observed, 4),
        "expected_agreement": round(p_expected, 4),
        "n": n,
        "labels": list(labels),
        "confusion": confusion.astype(int).tolist(),
    }


# ============================================================
# Stance Metrics
# ============================================================

def compute_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: Optional[list[str]] = None,
) -> dict:
    """
    Calcola Macro F1-Score per la Stance Detection.

    Il Macro F1 è la metrica corretta per classi sbilanciate:
    calcola F1 per ogni classe indipendentemente e fa la media non pesata.
    Questo garantisce che le classi minoritarie (es. NEUTRAL) abbiano
    lo stesso peso delle classi maggioritarie.

    Returns
    -------
    dict
        {
            "macro_f1": float,
            "per_class": {"FAVOR": {...}, "AGAINST": {...}, "NEUTRAL": {...}}
        }
    """
    if labels is None:
        labels = ["FAVOR", "AGAINST", "NEUTRAL"]

    per_class = {}
    f1_scores = []

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }
        f1_scores.append(f1)

    return {
        "macro_f1": round(float(np.mean(f1_scores)), 4),
        "per_class": per_class,
    }


# ============================================================
# Stance Diversity Score (Metrica Originale del Progetto)
# ============================================================

def compute_stance_diversity(clusters: dict[str, list]) -> dict:
    """
    Calcola la Stance Diversity Score tramite Entropia di Shannon.

    QUESTA È LA METRICA ORIGINALE DEL PROGETTO.
    Misura il successo dell'obiettivo anti-echo-chamber.

    A differenza di NDCG (che misura rilevanza), la Diversity Score misura
    QUANTO L'UTENTE È ESPOSTO A PROSPETTIVE BILANCIATE.

    Formula:
        H = -Σ p_i · log2(p_i)    per i ∈ {FAVOR, AGAINST, NEUTRAL}
        Diversity = H / log2(3)    normalizzazione in [0, 1]

    Interpretazione:
        0.0  → Echo chamber perfetto (tutti i doc della stessa stance)
        0.5  → Sistema parzialmente bilanciato
        1.0  → Distribuzione perfettamente uniforme (anti-echo-chamber ideale)

    Parametri
    ----------
    clusters : dict
        Output di StanceDetector.cluster() — {stance: [doc, ...]}

    Returns
    -------
    dict
        {
            "shannon_entropy": float,      # bit [0, log2(3)]
            "diversity_score": float,      # normalizzato [0, 1]
            "distribution": dict,          # conteggi e proporzioni
            "interpretation": str          # descrizione human-readable
        }
    """
    counts = np.array([len(clusters.get(k, [])) for k in ["FAVOR", "AGAINST", "NEUTRAL"]])
    total = counts.sum()

    if total == 0:
        return {
            "shannon_entropy": 0.0,
            "diversity_score": 0.0,
            "distribution": {},
            "interpretation": "No documents available.",
        }

    proportions = counts / total
    nonzero = proportions[proportions > 0]
    # `+ 0.0` normalizza lo zero negativo: con una sola classe popolata la somma
    # e' -0.0, che stampato diventa "-0.0000" e sembra un bug nei report.
    entropy = float(-np.sum(nonzero * np.log2(nonzero))) + 0.0
    max_entropy = math.log2(3)
    diversity_score = entropy / max_entropy

    # Interpretazione human-readable per il report
    if diversity_score >= 0.85:
        interpretation = "Highly balanced — effective echo chamber mitigation."
    elif diversity_score >= 0.60:
        interpretation = "Moderately balanced — partial perspective exposure."
    elif diversity_score >= 0.30:
        interpretation = "Imbalanced — significant stance bias detected."
    else:
        interpretation = "Severely imbalanced — echo chamber risk high."

    return {
        "shannon_entropy": round(entropy, 4),
        "diversity_score": round(diversity_score, 4),
        "distribution": {
            k: {"count": int(c), "proportion": round(float(p), 3)}
            for k, c, p in zip(["FAVOR", "AGAINST", "NEUTRAL"], counts, proportions)
        },
        "interpretation": interpretation,
    }
