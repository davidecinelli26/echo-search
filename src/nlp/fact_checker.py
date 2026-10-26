"""
Echo-Search | Fact-checking — composizione delle tre fasi
=========================================================
Mette in fila le tre fasi e le espone alla pipeline con un'unica interfaccia:

    claims.ClaimExtractor   ->  che cosa verificare
    kb_evidence.KbEvidence  ->  su quali evidenze
    nli_verdict | llm_verdict -> chi emette il verdetto

LE PRIME DUE FASI SONO CONDIVISE, LA TERZA E' SOSTITUIBILE. Non e' un dettaglio
implementativo ma la condizione che rende il confronto fra i due motori un
esperimento controllato: stessa estrazione, stessa knowledge base, stesso
cancello di pertinenza, cambia solo chi decide. Qualunque differenza misurata e'
quindi attribuibile al motore.

UNA CLAIM PER DOCUMENTO. Il documento mostrato all'utente porta il verdetto sulla
sua affermazione piu' verificabile, non su tutte: la card della dashboard ha
spazio per un giudizio, e tre verdetti discordi sullo stesso documento
confonderebbero invece di informare.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.nlp.claims import ClaimExtractor
from src.nlp.kb_evidence import KbEvidence, unique_sources
from src.nlp.nli_verdict import VERDICT_NEI

logger = logging.getLogger(__name__)

ENGINES = ("nli", "llm")


class FactChecker:
    """
    Verifica le affermazioni dei documenti contro la knowledge base del tema.

    Parametri
    ----------
    topic : str
        Tema, che seleziona la knowledge base (`data/kb/wikipedia_{topic}.jsonl`).
    engine : {"nli", "llm"}
        Motore di verdetto. "nli" e' veloce (decine di ms per claim), "llm"
        ragiona meglio ma costa secondi: la scelta e' esposta all'utente nella
        dashboard e all'ablazione negli script di valutazione.
    per_column : int, opzionale
        Documenti verificati in testa a ciascuna colonna. `None` = tutti; un
        valore piccolo tiene interattiva la dashboard col motore LLM.
    """

    def __init__(self, topic: str, engine: str = "nli", per_column: Optional[int] = None,
                 kb_dir: str = "data/kb", config_path: Optional[str] = None):
        if engine not in ENGINES:
            raise ValueError(f"engine deve essere uno di {ENGINES}, ricevuto {engine!r}")
        self.engine_name = engine
        self.per_column = per_column
        self.claims = ClaimExtractor(config_path=config_path)
        self.evidence = KbEvidence(topic, kb_dir=kb_dir, config_path=config_path)
        self._engine = None
        self._config_path = config_path

    @property
    def engine(self):
        """Motore di verdetto, costruito al primo uso (l'NLI carica ~700 MB)."""
        if self._engine is None:
            if self.engine_name == "llm":
                from src.nlp.llm_verdict import LlmVerdict
                self._engine = LlmVerdict(config_path=self._config_path)
            else:
                from src.nlp.nli_verdict import NliVerdict
                self._engine = NliVerdict(config_path=self._config_path)
        return self._engine

    def check_claim(self, claim: str) -> dict:
        """
        Le tre fasi su una singola affermazione.

        Returns
        -------
        dict
            {"claim", "verdict", "reasoning", "sources"}. Senza evidenze
            pertinenti il cancello ha gia' deciso: NOT_ENOUGH_EVIDENCE.
        """
        evidenze = self.evidence.retrieve(claim)
        if not evidenze:
            return {"claim": claim, "verdict": VERDICT_NEI, "reasoning": "", "sources": []}
        esito = self.engine.verdict(claim, [e["text"] for e in evidenze])
        return {"claim": claim, **esito, "sources": unique_sources(evidenze)}

    def check_columns(self, columns: dict) -> dict:
        """
        Arricchisce i documenti delle colonne con il verdetto sulla loro claim.

        I documenti senza affermazioni verificabili passano invariati: meglio
        nessun badge che un verdetto su una frase che non asserisce nulla.
        """
        verificate = {}
        n = 0
        for label, docs in columns.items():
            arricchiti = []
            for i, doc in enumerate(docs):
                da_verificare = self.per_column is None or i < self.per_column
                claims = self.claims.extract(doc["text"], max_claims=1) if da_verificare else []
                if claims:
                    fc = self.check_claim(claims[0])
                    arricchiti.append({**doc, "fact_checks": [fc],
                                       "factuality": {fc["verdict"]: 1}})
                    n += 1
                else:
                    arricchiti.append(doc)
            verificate[label] = arricchiti
        logger.info("Fact-check (%s): %d documenti verificati", self.engine_name, n)
        return verificate
