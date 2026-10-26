"""
Echo-Search | Fact-checking, fase 2 — Evidenze dalla knowledge base
===================================================================
Data una claim, le frasi della knowledge base Wikipedia che potrebbero
confermarla o smentirla. Fase condivisa dai due motori di verdetto: e' proprio
perche' entrambi ricevono le stesse evidenze che il confronto fra NLI e LLM
misura il motore e non il recupero.

DUE SCELTE PORTANO IL PESO DEL MODULO.

Il recupero e' SEMANTICO (bge) e non lessicale: una claim e la frase di
Wikipedia che la verifica raramente condividono le parole, e BM25 su una base
di poche migliaia di frasi degenera (l'IDF di un termine presente ovunque
tende a zero).

Il CANCELLO DI PERTINENZA e' la parte che conta davvero. Su una knowledge base
tematica il retrieval restituisce SEMPRE qualcosa: se la base parla di clima e
la claim parla di clima, una frase somigliante si trova comunque. Un modello
NLI messo davanti a un'evidenza che parla d'altro pur restando in tema legge
"diverso" come "contraddittorio" e produce falsi REFUTED, che e' l'errore
peggiore per un sistema che mostra verdetti a un lettore. Sotto la soglia di
somiglianza non si interroga alcun motore: il verdetto e' NOT_ENOUGH_EVIDENCE,
che e' onesto.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.nlp.kb import load_kb
from src.utils.config import load_section
from src.utils.text import split_into_sentences

logger = logging.getLogger(__name__)

_BGE = "BAAI/bge-base-en-v1.5"
# bge-v1.5 e' asimmetrico: l'istruzione va SOLO sulla query (qui la claim),
# mai sui passaggi indicizzati.
_QUERY_INSTR = "Represent this sentence for searching relevant passages: "
# Frasi piu' corte di cosi' sono artefatti di segmentazione, non evidenze.
_MIN_TOKENS = 5


class KbEvidence:
    """
    Indice semantico delle frasi della KB di un tema, con cancello di pertinenza.

    L'indice viene costruito al primo uso e tenuto in memoria: la KB di un tema
    e' di poche migliaia di frasi, quindi il prodotto matrice-vettore per la
    similarita' costa meno del caricamento di un indice su disco.
    """

    def __init__(self, topic: str, kb_dir: str = "data/kb", top_k: int = 5,
                 config_path: Optional[str] = None):
        cfg = load_section(config_path, "fact_checking")
        self.topic = topic
        self.kb_dir = kb_dir
        self.top_k = cfg.get("evidence_top_k", top_k)
        self.relevance_threshold = float(cfg.get("relevance_threshold", 0.62))
        self._encoder = None
        self._frasi: Optional[list[str]] = None
        self._fonti: Optional[list[dict]] = None
        self._emb = None

    @property
    def encoder(self):
        """bge caricato al primo uso, su GPU se disponibile."""
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._encoder = SentenceTransformer(_BGE, device=device)
        return self._encoder

    def _build_index(self) -> None:
        """Spezza la KB in frasi e le codifica una volta sola, con la mappa alle fonti."""
        if self._frasi is not None:
            return
        frasi, fonti = [], []
        try:
            kb_docs = load_kb(self.topic, self.kb_dir)
        except FileNotFoundError:
            logger.warning("KB assente per '%s': ogni claim dara' NOT_ENOUGH_EVIDENCE", self.topic)
            kb_docs = []
        for doc in kb_docs:
            for frase in split_into_sentences(doc["text"]):
                if len(frase.split()) >= _MIN_TOKENS:
                    frasi.append(frase)
                    fonti.append({"title": doc.get("title", ""), "url": doc.get("url", "")})
        self._frasi, self._fonti = frasi, fonti
        self._emb = (
            self.encoder.encode(frasi, normalize_embeddings=True, batch_size=64,
                                convert_to_numpy=True, show_progress_bar=False)
            if frasi else None
        )
        logger.info("KB '%s': %d frasi indicizzate", self.topic, len(frasi))

    def retrieve(self, claim: str) -> list[dict]:
        """
        Le evidenze pertinenti per la claim, o lista vuota se il cancello le blocca.

        Returns
        -------
        list[dict]
            [{"text", "title", "url"}, ...] ordinate per somiglianza decrescente.
            Vuota se la migliore evidenza non raggiunge `relevance_threshold`:
            il chiamante deve leggerlo come NOT_ENOUGH_EVIDENCE.
        """
        self._build_index()
        if self._emb is None:
            return []
        # embedding normalizzati -> il prodotto scalare E' la cosine similarity
        q = self.encoder.encode([_QUERY_INSTR + claim], normalize_embeddings=True,
                                convert_to_numpy=True)[0]
        sims = self._emb @ q
        top = np.argsort(-sims)[: self.top_k]
        if sims[top[0]] < self.relevance_threshold:   # cancello di pertinenza
            return []
        return [
            {"text": self._frasi[i], **self._fonti[i]}
            for i in top.tolist()
        ]


def unique_sources(evidences: list[dict], limit: int = 3) -> list[dict]:
    """Fonti distinte delle evidenze, per URL, nell'ordine di pertinenza."""
    viste, fonti = set(), []
    for ev in evidences:
        url = ev.get("url")
        if url and url not in viste:
            viste.add(url)
            fonti.append({"title": ev.get("title", ""), "url": url})
    return fonti[:limit]
