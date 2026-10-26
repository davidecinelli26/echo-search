"""
Echo-Search | Tier 2 — Dense Retriever (Sentence-BERT + FAISS)
===============================================================
Implementa la componente Dense (semantica) dell'Hybrid Retrieval Engine.

SCELTE ARCHITETTURALI:
-----------------------
- Encoder: Sentence-BERT "all-MiniLM-L6-v2" (Wang et al., 2020)
  → 6x più veloce di bert-base-uncased a parità di qualità semantica
  → 384 dimensioni (vs 768 di BERT-base): indice FAISS più compatto
- Vector DB: FAISS IndexFlatIP (Meta, Johnson et al., 2017)
  → Inner Product su vettori L2-normalizzati ≡ Cosine Similarity
  → Esatto (non approssimato): garantisce risultati riproducibili per la valutazione
- Chunking: 512 token con overlap di 128 token
  MOTIVAZIONE CHUNK_SIZE=512: limite architetturale dei modelli BERT-based.
  MOTIVAZIONE OVERLAP=128 (25%): valore da Lewis et al. (2020) RAG paper.
  Con overlap=50 si perdono claim che cadono esattamente a cavallo dei boundary.
  128 token garantisce che ogni frase di un articolo standard (≤50 token)
  appaia interamente in almeno un chunk.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.config import load_section
from src.utils.ranking import top_k_indices

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retriever semantico basato su Sentence-BERT e FAISS.

    Gestisce il chunking dei documenti lunghi e aggrega i chunk-score
    per documento via Max-Pooling (come descritto nell'architettura).

    Parametri
    ----------
    model_name : str
        Nome del modello Sentence-BERT su HuggingFace Hub.
    chunk_size : int
        Dimensione massima di ogni chunk in token WordPiece. Default 512.
    chunk_overlap : int
        Overlap in token tra chunk consecutivi. Default 128.
    config_path : str, opzionale
        Path a configs/config.yaml.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        config_path: Optional[str] = None,
    ):
        corpus_cfg = load_section(config_path, "corpus")
        dense_cfg = load_section(config_path, "dense")
        model_name = dense_cfg.get("model_name", model_name)
        # BGE/E5 sono asimmetrici: prefisso-istruzione SOLO sulle query (vedi search()).
        self.query_prefix = dense_cfg.get("query_prefix", "")
        chunk_size = corpus_cfg.get("chunk_size", chunk_size)
        chunk_overlap = corpus_cfg.get("chunk_overlap", chunk_overlap)

        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Caricamento modello Sentence-BERT
        logger.info(f"Caricamento Sentence-BERT: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        # FAISS index (inizializzato durante index())
        self.faiss_index: Optional[faiss.IndexFlatIP] = None

        # Mappa chunk → documento originale
        self.chunk_to_doc_map: list[int] = []    # chunk_id → doc_id (indice)
        self.corpus: list[str] = []
        self.doc_ids: list[str] = []

        logger.info(
            f"DenseRetriever | model={model_name} | "
            f"dim={self.embedding_dim} | "
            f"chunk={chunk_size}tok, overlap={chunk_overlap}tok"
        )

    # ------------------------------------------------------------------
    # Fase Offline: Indicizzazione
    # ------------------------------------------------------------------

    def index(
        self,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Costruisce il Vector Database FAISS.

        Pipeline:
        1. Ogni documento viene spezzato in chunk (con overlap)
        2. Ogni chunk viene encodato da Sentence-BERT → vettore 384-dim
        3. I vettori vengono L2-normalizzati (cosine similarity ≡ inner product)
        4. Tutti i vettori vengono inseriti in FAISS IndexFlatIP

        Parametri
        ----------
        documents : list[str]
            Corpus di documenti grezzi.
        doc_ids : list[str], opzionale
            ID per ogni documento.
        batch_size : int
            Dimensione del batch per l'encoding (GPU/CPU).
        """
        self.corpus = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        logger.info(f"Avvio chunking e encoding di {len(documents)} documenti...")

        # Step 1: Chunking di tutti i documenti
        all_chunks = []
        self.chunk_to_doc_map = []

        for doc_idx, doc_text in enumerate(documents):
            chunks = self._chunk_text(doc_text)
            all_chunks.extend(chunks)
            self.chunk_to_doc_map.extend([doc_idx] * len(chunks))

        logger.info(
            f"Chunking completato: {len(all_chunks)} chunk da {len(documents)} documenti "
            f"(media {len(all_chunks)/len(documents):.1f} chunk/doc)"
        )

        # Step 2: Encoding batch
        chunk_embeddings = self.encoder.encode(
            all_chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize → IP ≡ cosine similarity
        )

        # Step 3: Costruzione indice FAISS
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(chunk_embeddings.astype(np.float32))

        logger.info(
            f"FAISS index costruito: {self.faiss_index.ntotal} vettori | "
            f"dim={self.embedding_dim}"
        )

    # ------------------------------------------------------------------
    # Fase Online: Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Ricerca semantica con aggregazione Max-Pooling per documento.

        Strategia Max-Pooling (motivazione):
        Un documento è rilevante se ALMENO UN CHUNK è semanticamente vicino
        alla query. Il Max-Pooling seleziona lo score del chunk migliore per
        ogni documento, evitando di penalizzare documenti lunghi i cui chunk
        iniziali (meno rilevanti) abbasserebbero la media.
        """
        self._assert_indexed()

        # Encoding della query (stesso spazio vettoriale dei documenti)
        query_embedding = self.encoder.encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Ricerca sui chunk — prendiamo più chunk del necessario per il Max-Pooling
        n_chunks_to_search = min(self.faiss_index.ntotal, top_k * 5)
        chunk_scores, chunk_indices = self.faiss_index.search(
            query_embedding, n_chunks_to_search
        )

        # Aggregazione Max-Pooling vettoriale (scatter-op, nessun loop Python)
        doc_scores = self._max_pool_doc_scores(chunk_scores[0], chunk_indices[0], fill=-np.inf)

        # Ranking finale per documento (argpartition Top-K)
        ranked = top_k_indices(doc_scores, min(top_k, len(doc_scores)))

        return [
            {
                "doc_id": self.doc_ids[doc_idx],
                "rank": rank,
                "dense_score": round(float(doc_scores[doc_idx]), 6),
                "text": self.corpus[doc_idx],
            }
            for rank, doc_idx in enumerate(ranked.tolist(), start=1)
            if doc_scores[doc_idx] > 0
        ]

    def get_scores_array(self, query: str) -> np.ndarray:
        """
        Restituisce gli score densi per TUTTI i documenti del corpus.
        Usato dall'HybridRetriever per la Score Fusion.
        Usa Max-Pooling vettoriale per aggregare i chunk di ogni documento;
        restituisce direttamente l'ndarray (niente round-trip .tolist()).
        """
        self._assert_indexed()

        query_embedding = self.encoder.encode(
            [self.query_prefix + query], normalize_embeddings=True
        ).astype(np.float32)

        n_chunks = self.faiss_index.ntotal
        chunk_scores, chunk_indices = self.faiss_index.search(
            query_embedding, n_chunks
        )

        # fill=0.0 ≡ clamp a 0 della versione precedente: i documenti senza
        # chunk (o con soli score coseno negativi) non contribuiscono alla fusione
        return self._max_pool_doc_scores(chunk_scores[0], chunk_indices[0], fill=0.0)

    def _max_pool_doc_scores(
        self,
        chunk_scores: np.ndarray,
        chunk_indices: np.ndarray,
        fill: float,
    ) -> np.ndarray:
        """
        Aggregazione Max-Pooling chunk → documento, interamente vettoriale.

        np.maximum.at è una scatter-op unbuffered: applica
        doc_scores[m[i]] = max(doc_scores[m[i]], s[i]) per tutti i chunk
        in una sola chiamata nativa, gestendo correttamente i documenti
        con più chunk nel batch (a differenza del fancy-indexing normale).
        Sostituisce il precedente loop Python O(n_chunk) per-elemento.
        """
        doc_scores = np.full(len(self.corpus), fill, dtype=np.float64)
        valid = chunk_indices >= 0  # FAISS restituisce -1 per gli slot vuoti
        doc_map = np.asarray(self.chunk_to_doc_map, dtype=np.intp)
        np.maximum.at(
            doc_scores,
            doc_map[chunk_indices[valid]],
            chunk_scores[valid].astype(np.float64),
        )
        return doc_scores

    # ------------------------------------------------------------------
    # Persistenza Indice
    # ------------------------------------------------------------------

    def save_index(self, faiss_path: str) -> None:
        """Serializza l'indice FAISS e i metadati associati su disco."""
        self._assert_indexed()
        faiss_path = Path(faiss_path)
        meta_path = faiss_path.with_suffix(".meta.pkl")

        faiss.write_index(self.faiss_index, str(faiss_path))
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "corpus": self.corpus,
                    "doc_ids": self.doc_ids,
                    "chunk_to_doc_map": self.chunk_to_doc_map,
                    "model_name": self.model_name,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                },
                f,
            )
        logger.info("Indice FAISS salvato in: %s (meta: %s)", faiss_path, meta_path)

    def load_index(self, faiss_path: str) -> None:
        """Carica un indice FAISS precedentemente salvato."""
        faiss_path = Path(faiss_path)
        meta_path = faiss_path.with_suffix(".meta.pkl")

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadati dense mancanti: {meta_path}. "
                "Riesegui scripts/index_corpus.py per rigenerare gli indici."
            )

        self.faiss_index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.corpus = meta["corpus"]
        self.doc_ids = meta["doc_ids"]
        self.chunk_to_doc_map = meta["chunk_to_doc_map"]
        self.model_name = meta.get("model_name", self.model_name)
        self.chunk_size = meta.get("chunk_size", self.chunk_size)
        self.chunk_overlap = meta.get("chunk_overlap", self.chunk_overlap)

        logger.info(
            "Indice FAISS caricato da: %s | %d vettori | %d documenti",
            faiss_path,
            self.faiss_index.ntotal,
            len(self.corpus),
        )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """
        Divide il testo in chunk di dimensione fissa con overlap.

        Strategia: chunking a livello di parola (non di token WordPiece)
        per evitare il taglio a metà parola. La conversione in token
        WordPiece avviene internamente in Sentence-BERT.

        NOTA: chunk_size e chunk_overlap sono in token stimati (1 parola ≈ 1.3 token).
        Per un controllo preciso, si userebbe il tokenizer di Sentence-BERT.
        """
        words = text.split()

        # Stima token → parole (1 token WordPiece ≈ 0.75 parole inglesi)
        words_per_chunk = int(self.chunk_size * 0.75)
        words_overlap = int(self.chunk_overlap * 0.75)
        step = max(1, words_per_chunk - words_overlap)

        if len(words) <= words_per_chunk:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start += step

        return chunks

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    @property
    def is_indexed(self) -> bool:
        """True se l'indice FAISS è caricato — usato dall'Hybrid per degradare a solo-BM25."""
        return self.faiss_index is not None

    def _assert_indexed(self) -> None:
        if self.faiss_index is None:
            raise RuntimeError(
                "FAISS index non costruito. Eseguire index() prima di search()."
            )

    @property
    def corpus_size(self) -> int:
        return len(self.corpus)
