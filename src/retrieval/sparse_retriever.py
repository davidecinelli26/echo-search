"""
Echo-Search | Tier 2 — Sparse Retriever (BM25)
===============================================
Implementa la componente Sparse dell'Hybrid Retrieval Engine.

SCELTE ARCHITETTURALI:
-----------------------
- Modello: Okapi BM25 (Robertson et al., 1994)
- Parametri caricati da config.yaml (nessun magic number hardcoded)
- b ∈ [0.50, 0.65]: riduzione penalizzazione per documenti lunghi.
  Articoli giornalistici di approfondimento (1000–3000 token) sarebbero
  ingiustamente penalizzati dal valore default b=0.75, perdendo le evidenze
  fattuali necessarie al Fact-Checking.
- k1 = 1.2: valore standard per corpora in inglese (Robertson & Zaragoza, 2009).
"""

import logging
import pickle
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.utils.config import load_section
from src.utils.ranking import top_k_indices

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    Retriever BM25 con parametri configurabili e persistenza dell'indice.

    Parametri
    ----------
    k1 : float
        Saturazione della frequenza del termine. Default 1.2.
    b : float
        Penalizzazione per lunghezza documento. Range [0, 1].
        Default 0.60 (valore calibrato per articoli giornalistici lunghi).
    config_path : str, opzionale
        Path al file configs/config.yaml. Se fornito, sovrascrive k1 e b.
    """

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.60,
        config_path: Optional[str] = None,
    ):
        # Carica parametri da config se disponibile
        bm25_cfg = load_section(config_path, "bm25")
        self.k1 = bm25_cfg.get("k1", k1)
        self.b = bm25_cfg.get("b", b)

        self.preprocessor = TextPreprocessor()
        self.corpus: list[str] = []           # Testi originali (per il rendering UI)
        self.doc_ids: list[str] = []          # ID opzionali (es. URL, nome file)
        self.bm25_index: Optional[BM25Okapi] = None

        logger.info(f"SparseRetriever inizializzato | k1={self.k1}, b={self.b}")

    # ------------------------------------------------------------------
    # Fase Offline: Indicizzazione
    # ------------------------------------------------------------------

    def index(
        self,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
    ) -> None:
        """
        Costruisce l'Inverted Index BM25 sul corpus.
        Operazione OFFLINE: da eseguire una volta sola prima del retrieval.

        Complessità: O(N × L) dove N = numero documenti, L = lunghezza media.

        Parametri
        ----------
        documents : list[str]
            Lista di testi grezzi del corpus.
        doc_ids : list[str], opzionale
            Identificatori univoci per ogni documento (es. URL, filename).
            Se None, usa indici numerici.
        """
        if not documents:
            raise ValueError("Il corpus è vuoto. Fornire almeno un documento.")

        logger.info(f"Avvio indicizzazione BM25 su {len(documents)} documenti...")

        self.corpus = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        # Preprocessing batch-ottimizzato (usa spaCy pipe())
        tokenized_corpus = self.preprocessor.process_batch(documents)

        # BM25Okapi calcola internamente TF, IDF e avgdl
        # NOTA: rank_bm25 non espone i parametri k1/b direttamente nel costruttore
        # → usiamo la sottoclasse BM25Okapi e aggiorniamo i parametri post-init
        self.bm25_index = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        logger.info(
            f"Indicizzazione completata. "
            f"Vocabolario: {len(self.bm25_index.idf)} termini unici. "
            f"avgdl: {self.bm25_index.avgdl:.1f} token."
        )

    # ------------------------------------------------------------------
    # Fase Online: Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Esegue la ricerca BM25 su query in testo libero.

        NOTA FONDAMENTALE: la query viene preprocessata con la STESSA pipeline
        del corpus (preprocessing simmetrico). Asimmetrie causano vocabulary
        mismatch artificiale.

        Parametri
        ----------
        query : str
            Query in linguaggio naturale (es. "risks of nuclear power plants").
        top_k : int
            Numero di documenti da restituire.

        Returns
        -------
        list[dict]
            Lista di risultati ordinati per score BM25 decrescente.
            Ogni risultato: {"doc_id", "rank", "bm25_score", "text"}
        """
        self._assert_indexed()

        # Preprocessing simmetrico della query
        tokenized_query = self.preprocessor.process(query)

        if not tokenized_query:
            logger.warning("Query vuota dopo preprocessing. Verifica la query.")
            return []

        logger.debug(f"Query tokenizzata: {tokenized_query}")

        # Calcolo scores BM25 per tutti i documenti (ndarray)
        scores = self.bm25_index.get_scores(tokenized_query)

        # Ranking Top-K vettoriale: argpartition O(n) invece del sort O(n log n)
        ranked_indices = top_k_indices(np.asarray(scores), min(top_k, len(scores)))

        results = []
        for rank, idx in enumerate(ranked_indices.tolist(), start=1):
            if scores[idx] <= 0:
                # Documenti con score 0 non condividono alcun termine con la query
                break
            results.append({
                "doc_id": self.doc_ids[idx],
                "rank": rank,
                "bm25_score": round(float(scores[idx]), 6),
                "text": self.corpus[idx],
            })

        logger.info(
            f"BM25 search completata: {len(results)} risultati rilevanti "
            f"(top score: {results[0]['bm25_score'] if results else 0:.4f})"
        )
        return results

    def get_scores_array(self, query: str) -> np.ndarray:
        """
        Restituisce gli score BM25 grezzi per TUTTI i documenti del corpus.
        Usato dall'HybridRetriever per la Score Fusion.

        Restituisce direttamente l'ndarray di rank_bm25: evita il round-trip
        .tolist() → np.array() che copiava due volte l'intero array.
        """
        self._assert_indexed()
        tokenized_query = self.preprocessor.process(query)
        return np.asarray(self.bm25_index.get_scores(tokenized_query))

    # ------------------------------------------------------------------
    # Persistenza Indice
    # ------------------------------------------------------------------

    def save_index(self, path: str) -> None:
        """Serializza l'indice BM25 su disco (evita re-indicizzazione)."""
        index_data = {
            "bm25_index": self.bm25_index,
            "corpus": self.corpus,
            "doc_ids": self.doc_ids,
            "params": {"k1": self.k1, "b": self.b},
        }
        with open(path, "wb") as f:
            pickle.dump(index_data, f)
        logger.info(f"Indice BM25 salvato in: {path}")

    def load_index(self, path: str) -> None:
        """Carica un indice BM25 precedentemente salvato."""
        with open(path, "rb") as f:
            index_data = pickle.load(f)
        self.bm25_index = index_data["bm25_index"]
        self.corpus = index_data["corpus"]
        self.doc_ids = index_data["doc_ids"]
        self.k1 = index_data["params"]["k1"]
        self.b = index_data["params"]["b"]
        logger.info(
            f"Indice BM25 caricato da: {path} | "
            f"{len(self.corpus)} documenti | k1={self.k1}, b={self.b}"
        )

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _assert_indexed(self) -> None:
        if self.bm25_index is None:
            raise RuntimeError(
                "Indice BM25 non costruito. "
                "Eseguire index() prima di search()."
            )

    @property
    def corpus_size(self) -> int:
        return len(self.corpus)
