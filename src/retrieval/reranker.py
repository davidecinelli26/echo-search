"""
Echo-Search | Tier 2 — Cross-Encoder Reranker (stadio 2 del retrieval)
======================================================================
Riordina i Top-N candidati dello stadio di fusione con un cross-encoder che
valuta la coppia (query, documento) tramite cross-attention completa.

PERCHÉ UN SECONDO STADIO (retrieve-then-rerank):
- Lo stadio 1 (BM25 + Dense fusi) è un bi-encoder/lessicale: veloce su tutto il
  corpus ma approssimato (query e documento codificati separatamente).
- Il cross-encoder è O(N) — troppo costoso su tutto il corpus — ma molto più
  preciso. Applicato SOLO ai Top-N candidati costa poco e alza la qualità del
  ranking. È lo stato dell'arte dei sistemi RAG.
- NON sostituisce la fusione: la fusione RECUPERA i candidati, il reranker li
  RIORDINA. I due stadi sono complementari.

Il reranker è agnostico alla stance: riordina per PERTINENZA. Il bilanciamento
delle prospettive avviene dopo, allo stadio di clustering/colonne (Fase 2.4).
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from src.utils.config import load_section
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Re-ranker basato su cross-encoder (default: BAAI/bge-reranker-base).

    Parametri
    ----------
    model_name : str
        Cross-encoder HuggingFace. Default "BAAI/bge-reranker-base".
    top_n : int
        Numero di candidati (dallo stadio 1) da riordinare. Default 30.
    config_path : str, opzionale
        Path a configs/config.yaml (sezione reranker).
    device : str, opzionale
        "cuda" | "cpu". Default: auto-detect.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int = 30,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        cfg = load_section(config_path, "reranker")
        self.model_name = cfg.get("model_name", model_name)
        self.top_n = cfg.get("top_n", top_n)
        self.device = resolve_device(device)

        logger.info(f"Caricamento reranker: {self.model_name} | device={self.device}")
        # tokenizer slow per compatibilità coi tokenizer.json recenti (come StanceDetector)
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            tokenizer_args={"use_fast": False},
        )

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """
        Riordina i primi `top_n` risultati per pertinenza cross-encoder.

        Parametri
        ----------
        query : str
            Query utente.
        results : list[dict]
            Output dello stadio 1 (ogni dict ha almeno "text"), ordinato.
        top_n : int, opzionale
            Sovrascrive il top_n di default.

        Returns
        -------
        list[dict]
            I candidati riordinati, arricchiti con "rerank_score" e "rank"
            aggiornato. La coda oltre top_n viene scartata (non è rilevante
            per il Top-K finale).
        """
        n = top_n or self.top_n
        candidates = results[:n]
        if not candidates:
            return []

        pairs = [(query, r["text"]) for r in candidates]
        scores = self.model.predict(pairs)

        for r, score in zip(candidates, scores):
            r["rerank_score"] = round(float(score), 6)

        reranked = sorted(candidates, key=lambda r: r["rerank_score"], reverse=True)
        for rank, r in enumerate(reranked, start=1):
            r["rank"] = rank

        logger.info(f"Rerank: {len(candidates)} candidati riordinati (top score {reranked[0]['rerank_score']:.4f})")
        return reranked
