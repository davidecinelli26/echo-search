"""
Echo-Search | Fact-checking, fase 1 — Estrazione delle claim
============================================================
Dal testo di un documento alle affermazioni che vale la pena verificare.

E' la prima delle tre fasi dell'architettura FEVER (Thorne et al. 2018), che
separa il "cosa verificare" dal "dove cercare" (fase 2, kb_evidence) e dal "chi
giudica" (fase 3, nli_verdict / llm_verdict). La separazione non e' formale: e'
cio' che permette di sostituire il motore di verdetto lasciando identiche le
prime due fasi, ed e' l'unico modo di attribuire al motore la differenza di
prestazioni misurata.

Non tutte le frasi dichiarative meritano una verifica. Il filtro e' euristico,
nello spirito di ClaimBuster (Hassan et al. 2017): un'affermazione verificabile
cita fatti concreti, quindi entita' nominate e valori numerici sono il segnale.
Le entita' pesano il doppio dei numeri perche' un numero senza soggetto ("e'
cresciuto del 12%") non e' verificabile da solo.

spaCy viene caricato al primo uso: chi non chiede il fact-checking non paga il
caricamento del modello.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.utils.config import load_section

logger = logging.getLogger(__name__)

# Le entita' nominate segnalano un fatto verificabile piu' dei soli numeri.
_ENTITY_WEIGHT = 2.0
_NUMBER_WEIGHT = 1.0


class ClaimExtractor:
    """
    Estrae le affermazioni piu' verificabili da un testo.

    Parametri
    ----------
    spacy_model : str
        Modello spaCy con parser (serve doc.sents) e NER (serve la
        check-worthiness). Default dal config, sezione `fact_checking`.
    min_tokens : int
        Lunghezza minima di una claim, in token non di punteggiatura. Sotto
        questa soglia la frase e' quasi sempre un frammento di titolo o una
        didascalia.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm", min_tokens: int = 6,
                 config_path: Optional[str] = None):
        cfg = load_section(config_path, "fact_checking")
        self.spacy_model = cfg.get("spacy_model", spacy_model)
        self.min_tokens = min_tokens
        self._nlp = None

    @property
    def nlp(self):
        """spaCy caricato al primo uso (vedi docstring del modulo)."""
        if self._nlp is None:
            import spacy
            logger.info("Caricamento spaCy: %s", self.spacy_model)
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                logger.error("Modello spaCy '%s' assente. Eseguire: "
                             "python -m spacy download %s", self.spacy_model, self.spacy_model)
                raise
        return self._nlp

    def extract(self, text: str, max_claims: int = 1) -> list[str]:
        """
        Le `max_claims` frasi piu' check-worthy del testo, in ordine di punteggio.

        A parita' di punteggio vince l'ordine di apparizione nel documento: il
        sort e' stabile, quindi lo stesso documento produce sempre le stesse
        claim (requisito di riproducibilita' della demo precalcolata).
        """
        frasi, punteggi = [], []
        for sent in self.nlp(text).sents:
            if self._is_declarative(sent):
                frasi.append(sent.text.strip())
                punteggi.append(_check_worthiness(sent))
        if not frasi:
            return []
        ordine = np.argsort(-np.asarray(punteggi), kind="stable")[:max_claims]
        return [frasi[i] for i in ordine.tolist()]

    def _is_declarative(self, sent) -> bool:
        """Frase abbastanza lunga, non interrogativa e con un predicato verbale."""
        n_tokens = sum(1 for t in sent if not (t.is_punct or t.is_space))
        if n_tokens < self.min_tokens or sent.text.rstrip().endswith("?"):
            return False
        return any(t.pos_ in ("VERB", "AUX") for t in sent)


def _check_worthiness(sent) -> float:
    """Punteggio euristico: entita' nominate e numeri segnalano fatti concreti."""
    n_entita = len(sent.ents)
    n_numeri = sum(1 for t in sent if t.like_num)
    return _ENTITY_WEIGHT * n_entita + _NUMBER_WEIGHT * n_numeri
