"""
Echo-Search | Tier 2 — Hybrid Retriever (Score Fusion)
=======================================================
Componente centrale del Retrieval Engine. Fonde BM25 (Sparse) e Sentence-BERT
(Dense) tramite Weighted Score Fusion con normalizzazione Min-Max.

SCELTE ARCHITETTURALI — α e β:
------------------------------------------------------------
La scelta di α e β NON è arbitraria. Il sistema segue questo protocollo:

1. BASELINE: Reciprocal Rank Fusion (RRF, Cormack et al. 2009)
   Score_RRF = Σ 1/(k + rank_i) — non richiede tuning, utile come baseline.

2. GRID SEARCH su validation set:
   α ∈ {0.1, 0.2, ..., 0.9}, β = 1 - α
   Metrica di ottimizzazione: NDCG@10 sul validation set annotato.
   Con 11 query la curva di NDCG@10 e' piatta e la griglia seleziona rumore
   (scripts/eval_retrieval.py): il default (α=0.45, β=0.55) resta quello scelto
   per ragioni di dominio, come spiegato al punto 3.

3. MOTIVAZIONE α < β (Dense pesa leggermente di più):
   Il dominio argomentativo richiede comprensione semantica per sinonimi
   ("meltdown" ≈ "nuclear accident") e parafrasi tipiche del dibattito.
   BM25 rimane fondamentale per termini tecnici specifici del dominio.

Riferimento: Kuzi et al. (2020) "Query Expansion Using Word Embeddings"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.config import load_section
from src.utils.ranking import top_k_indices

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid Retrieval Engine: BM25 + Sentence-BERT con Weighted Score Fusion.

    Formula: Score_finale(d) = α · norm(BM25(d)) + β · norm(Dense(d))
    con α + β = 1.0 e normalizzazione Min-Max preventiva.

    Parametri
    ----------
    alpha : float
        Peso del retriever sparse (BM25). Default 0.45.
    beta : float
        Peso del retriever denso (Sentence-BERT). Default 0.55.
        NOTA: alpha + beta deve essere 1.0.
    normalization : str
        Strategia di normalizzazione degli score prima della fusione.
        "min_max" (default) o "z_score".
    top_k : int
        Numero di documenti finali da restituire al Tier 3.
    config_path : str, opzionale
        Path a configs/config.yaml.
    """

    def __init__(
        self,
        alpha: float = 0.45,
        beta: float = 0.55,
        normalization: str = "min_max",
        top_k: int = 20,
        config_path: Optional[str] = None,
    ):
        fusion_cfg = load_section(config_path, "fusion")
        alpha = fusion_cfg.get("alpha", alpha)
        beta = fusion_cfg.get("beta", beta)
        normalization = fusion_cfg.get("normalization", normalization)
        top_k = fusion_cfg.get("top_k", top_k)

        # Validazione dei pesi
        if not np.isclose(alpha + beta, 1.0, atol=1e-6):
            raise ValueError(
                f"α + β deve essere 1.0. Ricevuto: α={alpha}, β={beta}, somma={alpha+beta:.4f}"
            )

        self.alpha = alpha
        self.beta = beta
        self.normalization = normalization
        self.top_k = top_k

        # Inizializzazione retriever
        self.sparse = SparseRetriever(config_path=config_path)
        self.dense = DenseRetriever(config_path=config_path)

        logger.info(
            f"HybridRetriever | α(BM25)={self.alpha}, β(Dense)={self.beta} | "
            f"norm={self.normalization} | top_k={self.top_k}"
        )

    # ------------------------------------------------------------------
    # Factory method da config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "HybridRetriever":
        """Crea un HybridRetriever interamente configurato da YAML."""
        return cls(config_path=config_path)

    @classmethod
    def load_from_disk(
        cls,
        index_dir: str,
        config_path: str = "configs/config.yaml",
        skip_dense: bool = False,
    ) -> "HybridRetriever":
        """Carica indici BM25 e FAISS pre-costruiti da disco."""
        index_dir = Path(index_dir)
        retriever = cls.from_config(config_path)

        bm25_path = index_dir / "bm25_index.pkl"
        faiss_path = index_dir / "faiss_index.bin"

        retriever.sparse.load_index(str(bm25_path))

        if skip_dense:
            logger.warning("Caricamento dense skippato (--skip-dense)")
        elif faiss_path.exists():
            retriever.dense.load_index(str(faiss_path))
        else:
            logger.warning("FAISS index non trovato in %s — solo BM25 attivo", faiss_path)

        return retriever

    # ------------------------------------------------------------------
    # Fase Offline: Indicizzazione
    # ------------------------------------------------------------------

    def index(self, documents: list[str], doc_ids: Optional[list[str]] = None) -> None:
        """
        Costruisce sia l'Inverted Index BM25 che il Vector DB FAISS.
        Operazione OFFLINE da eseguire una volta sola (scripts/index_corpus.py).
        """
        logger.info("Avvio indicizzazione ibrida (BM25 + FAISS)...")
        self.sparse.index(documents, doc_ids)
        self.dense.index(documents, doc_ids)
        logger.info("Indicizzazione ibrida completata.")

    # ------------------------------------------------------------------
    # Fase Online: Ricerca
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Esegue la ricerca ibrida e restituisce i Top-K documenti fusi.

        Algoritmo:
        1. BM25 → array di score per tutti i documenti
        2. Sentence-BERT → array di score (cosine similarity) per tutti i documenti
        3. Normalizzazione Min-Max di entrambi gli array
        4. Fusione: score_finale = α·sparse_norm + β·dense_norm
        5. Ranking e selezione Top-K

        Parametri
        ----------
        query : str
            Query in testo libero.
        top_k : int, opzionale
            Sovrascrive il top_k di default.

        Returns
        -------
        list[dict]
            Documenti rankeati per score ibrido. Ogni dict include gli score
            parziali per trasparenza e debugging.
        """
        k = top_k or self.top_k

        # Step 1-3: score, normalizzazione e fusione.
        # Se il dense non è caricato (--skip-dense o FAISS assente) si degrada
        # a solo-BM25 invece di fallire: il sistema resta comunque usabile.
        sparse_norm = self._normalize(self.sparse.get_scores_array(query))

        if self.dense.is_indexed:
            dense_norm = self._normalize(self.dense.get_scores_array(query))
            fused_scores = self.alpha * sparse_norm + self.beta * dense_norm
        else:
            logger.warning("Dense non disponibile — fusione degradata a solo-BM25")
            dense_norm = np.zeros_like(sparse_norm)
            fused_scores = sparse_norm

        # Step 4: Ranking Top-K vettoriale (argpartition)
        ranked = top_k_indices(fused_scores, min(k, len(fused_scores)))

        results = []
        for rank, idx in enumerate(ranked.tolist(), start=1):
            if fused_scores[idx] <= 0:
                break
            results.append({
                "doc_id": self.sparse.doc_ids[idx],
                "rank": rank,
                "hybrid_score": round(float(fused_scores[idx]), 6),
                "bm25_score_norm": round(float(sparse_norm[idx]), 6),
                "dense_score_norm": round(float(dense_norm[idx]), 6),
                "text": self.sparse.corpus[idx],
            })

        logger.info(
            f"Hybrid search: {len(results)} risultati | "
            f"top score: {results[0]['hybrid_score']:.4f}" if results else "nessun risultato"
        )
        return results

    def search_rrf_baseline(self, query: str, k_rrf: int = 60, top_k: Optional[int] = None) -> list[dict]:
        """
        BASELINE: Reciprocal Rank Fusion (Cormack et al., 2009).
        Non richiede tuning di α/β — usato per confronto sperimentale.

        Score_RRF(d) = Σ_r 1/(k_rrf + rank_r(d))

        Parametri
        ----------
        k_rrf : int
            Costante di smoothing (default 60, valore originale del paper).
        """
        k = top_k or self.top_k

        sparse_scores = self.sparse.get_scores_array(query)
        dense_scores = self.dense.get_scores_array(query)

        # Rank per ciascun retriever (1-based, score decrescente)
        sparse_ranks = np.argsort(np.argsort(-sparse_scores)) + 1
        dense_ranks = np.argsort(np.argsort(-dense_scores)) + 1

        rrf_scores = 1.0 / (k_rrf + sparse_ranks) + 1.0 / (k_rrf + dense_ranks)

        ranked = top_k_indices(rrf_scores, min(k, len(rrf_scores)))

        return [
            {
                "doc_id": self.sparse.doc_ids[idx],
                "rank": rank,
                "rrf_score": round(float(rrf_scores[idx]), 6),
                "text": self.sparse.corpus[idx],
            }
            for rank, idx in enumerate(ranked.tolist(), start=1)
        ]

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalizza un array di score nell'intervallo [0, 1].
        Necessario prima della fusione perché BM25 e cosine similarity
        hanno scale completamente diverse.
        """
        if self.normalization == "min_max":
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-9:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)

        elif self.normalization == "z_score":
            std = scores.std()
            if std < 1e-9:
                return np.zeros_like(scores)
            return (scores - scores.mean()) / std

        else:
            raise ValueError(f"Normalizzazione sconosciuta: {self.normalization}")
