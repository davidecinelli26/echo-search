"""
Echo-Search | Fact-checking, fase 3a — Verdetto per inferenza naturale (NLI)
============================================================================
Motore di verdetto basato su un modello NLI: data una coppia (evidenza, claim)
stima quanto la prima implichi o contraddica la seconda, e la coppia di
probabilita' si aggrega in SUPPORTED / REFUTED / NOT_ENOUGH_EVIDENCE.

E' intercambiabile con llm_verdict: stessa firma `verdict(claim, evidenze)`,
stesso dizionario in uscita. La differenza di prestazioni fra i due, misurata su
ClimateFEVER a parita' di evidenze, e' quindi attribuibile al solo motore.

DUE SOGLIE, NON UNA. Confermare e smentire non sono compiti simmetrici: una
contraddizione dichiarata per errore e' un'accusa di falsita' rivolta a un
documento reale, una conferma mancata e' solo un'occasione persa. Per questo la
soglia di REFUTED e' molto piu' alta di quella di SUPPORTED (0,95 contro 0,75,
in `configs/config.yaml`). Il profilo di errore che ne risulta e' misurato e
discusso nella relazione.

Gli indici delle classi si leggono da `model.config.id2label`, mai per posizione:
il modello NLI e' sostituibile da configurazione e ogni famiglia ordina le classi
a modo suo.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.utils.config import load_section
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_NEI = "NOT_ENOUGH_EVIDENCE"


def aggregate(entail: np.ndarray, contra: np.ndarray,
              support_threshold: float, refute_threshold: float) -> tuple[str, int]:
    """
    Aggrega le probabilita' di piu' evidenze in un verdetto (logica pura).

    Basta UNA evidenza convincente: si guarda il massimo, non la media, perche'
    le evidenze sono alternative fra loro e mediarle diluirebbe quella giusta
    con tre irrilevanti.

    Returns
    -------
    (verdetto, indice dell'evidenza decisiva)
    """
    best_e, best_c = float(entail.max()), float(contra.max())
    if best_e >= support_threshold and best_e >= best_c:
        return VERDICT_SUPPORTED, int(entail.argmax())
    if best_c >= refute_threshold:
        return VERDICT_REFUTED, int(contra.argmax())
    # nessun segnale abbastanza forte: si dichiara l'incertezza, non si tira a indovinare
    return VERDICT_NEI, int(entail.argmax())


class NliVerdict:
    """
    Motore di verdetto NLI. Il modello (~700 MB) e' caricato al primo uso.

    Parametri
    ----------
    nli_model : str
        Modello NLI HuggingFace. Il default del progetto e' addestrato su FEVER,
        cioe' esattamente sul compito di verificare claim contro evidenze.
    batch_size : int
        Coppie per forward pass. Un solo forward per batch, mai uno per coppia.
    """

    def __init__(self, nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 batch_size: int = 16, device: Optional[str] = None,
                 config_path: Optional[str] = None):
        cfg = load_section(config_path, "fact_checking")
        self.model_name = cfg.get("nli_model", nli_model)
        self.support_threshold = float(cfg.get("support_threshold", 0.75))
        self.refute_threshold = float(cfg.get("refute_threshold", 0.95))
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self._model = None
        self._tokenizer = None
        self.entail_idx: Optional[int] = None
        self.contra_idx: Optional[int] = None

    @property
    def model(self):
        """Modello e tokenizer al primo uso; ricava gli indici entailment/contraddizione."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info("Caricamento NLI: %s | device=%s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(self.model_name)
                .to(self.device).eval()
            )
            id2label = {i: lab.lower() for i, lab in self._model.config.id2label.items()}
            self.entail_idx = next(i for i, lab in id2label.items() if "entail" in lab)
            self.contra_idx = next(i for i, lab in id2label.items() if "contradict" in lab)
        return self._model

    def probabilities(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """
        Probabilita' NLI di tutte le coppie, matrice [n_coppie, n_classi].

        `truncation="only_first"` tronca l'evidenza se serve, mai la claim: la
        claim e' cio' che si sta verificando e amputarla cambierebbe la domanda.
        """
        import torch
        model = self.model                        # innesca il caricamento lazy
        probs = np.empty((len(premises), model.config.num_labels), dtype=np.float32)
        for inizio in range(0, len(premises), self.batch_size):
            fine = inizio + self.batch_size
            encoded = self._tokenizer(
                premises[inizio:fine], hypotheses[inizio:fine],
                padding=True, truncation="only_first", return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = model(**encoded).logits
            probs[inizio:fine] = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def verdict(self, claim: str, evidences: list[str]) -> dict:
        """Verdetto sulla claim date le evidenze. Firma condivisa con LlmVerdict."""
        if not evidences:
            return {"verdict": VERDICT_NEI, "reasoning": ""}
        probs = self.probabilities(evidences, [claim] * len(evidences))
        verdetto, decisiva = aggregate(
            probs[:, self.entail_idx], probs[:, self.contra_idx],
            self.support_threshold, self.refute_threshold,
        )
        # la motivazione e' l'evidenza che ha determinato il verdetto: e' cio' che
        # rende il giudizio ispezionabile dall'utente nella dashboard
        return {"verdict": verdetto, "reasoning": evidences[decisiva][:200]}
