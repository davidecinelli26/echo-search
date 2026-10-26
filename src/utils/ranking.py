"""
Echo-Search | Utils — Ranking Top-K vettoriale
===============================================
Pattern ricorrente in Sparse/Dense/Hybrid Retriever e nel FactChecker
(evidence ranking): dato un array di score sull'intero corpus, servono
gli indici dei K elementi migliori in ordine decrescente.

PERCHÉ argpartition E NON argsort:
- np.argsort completo è O(n log n); np.argpartition è O(n).
- Con corpus da 10.000 documenti e K=20, il full sort ordina il 99.8%
  di elementi che verranno scartati.
- La selezione dei K vincitori viene poi ordinata con un argsort O(K log K):
  costo totale O(n + K log K).
"""

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Indici dei k score più alti, ordinati per score decrescente.

    Parametri
    ----------
    scores : np.ndarray
        Array 1-D di score (float). Valori -inf ammessi (documenti mai visti).
    k : int
        Numero di elementi da selezionare. Con k >= len(scores) restituisce
        il ranking completo.

    Returns
    -------
    np.ndarray
        Array di indici (dtype intero), lunghezza min(k, n).
        L'ordinamento interno usa kind="stable": a parità di score vince
        l'indice più basso → ranking riproducibile (requisito del protocollo
        di valutazione).
    """
    n = scores.shape[0]
    if k <= 0 or n == 0:
        return np.empty(0, dtype=np.intp)
    if k >= n:
        return np.argsort(-scores, kind="stable")

    partition = np.argpartition(-scores, k)[:k]
    return partition[np.argsort(-scores[partition], kind="stable")]
