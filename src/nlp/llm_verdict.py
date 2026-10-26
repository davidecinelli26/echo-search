"""
Echo-Search | Fact-checking, fase 3b — Verdetto per ragionamento (LLM locale)
=============================================================================
Motore di verdetto alternativo all'NLI: un LLM legge claim ed evidenze e decide,
motivando. Intercambiabile con nli_verdict — stessa firma `verdict(claim,
evidenze)`, stesso dizionario in uscita.

GROUNDING PURO. Il prompt impone di giudicare SOLO sulle evidenze fornite, mai
sulla conoscenza acquisita in addestramento. Senza questo vincolo una eventuale
superiorita' dell'LLM sarebbe ambigua: non si saprebbe se ragiona meglio o se
semplicemente ricorda la risposta, e il confronto con l'NLI perderebbe senso.

TUTTO IN LOCALE. Il modello gira via Ollama su localhost; si parla col server in
`urllib`, senza aggiungere dipendenze. Nessuna API remota, nessuna chiave,
nessuna quota: il sistema resta eseguibile e riproducibile offline.

DEGRADAZIONE ANZICHE' ERRORE. Se Ollama non risponde il verdetto e'
NOT_ENOUGH_EVIDENCE: il fact-checking e' un arricchimento dei risultati, non deve
poter far cadere una ricerca.

`canon_verdict` e `parse_verdict` sono pure: la parte fragile — interpretare
l'output di un modello generativo — e' testabile senza rete.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Optional

from src.utils.config import load_section

logger = logging.getLogger(__name__)

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_NEI = "NOT_ENOUGH_EVIDENCE"
_VALID = {VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI}

# Sinonimi che i modelli producono spontaneamente -> etichette del progetto.
_ALIASES = {
    "SUPPORT": VERDICT_SUPPORTED, "SUPPORTS": VERDICT_SUPPORTED, "TRUE": VERDICT_SUPPORTED,
    "ENTAILMENT": VERDICT_SUPPORTED, "ENTAILED": VERDICT_SUPPORTED,
    "REFUTE": VERDICT_REFUTED, "REFUTES": VERDICT_REFUTED, "FALSE": VERDICT_REFUTED,
    "CONTRADICTION": VERDICT_REFUTED, "CONTRADICTED": VERDICT_REFUTED,
    "NOT ENOUGH EVIDENCE": VERDICT_NEI, "NEI": VERDICT_NEI, "INSUFFICIENT": VERDICT_NEI,
    "NOT ENOUGH INFO": VERDICT_NEI, "NEUTRAL": VERDICT_NEI, "UNKNOWN": VERDICT_NEI,
    "UNVERIFIABLE": VERDICT_NEI,
}

_SYSTEM = (
    "You are a rigorous fact-checking assistant. You receive a CLAIM and a numbered list "
    "of EVIDENCE passages. Using ONLY the evidence (never prior knowledge), decide:\n"
    "- SUPPORTED: the evidence confirms/entails the claim;\n"
    "- REFUTED: the evidence contradicts the claim;\n"
    "- NOT_ENOUGH_EVIDENCE: the evidence is insufficient, unrelated, or ambiguous.\n"
    'Reply ONLY with JSON: {"verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_EVIDENCE", '
    '"reasoning": "<one concise sentence citing the evidence used>"}.'
)


def canon_verdict(v: str) -> str:
    """Normalizza un'etichetta a una delle tre canoniche; ignoto -> NEI."""
    v = (v or "").strip().upper().replace("-", "_")
    if v in _VALID:
        return v
    return _ALIASES.get(v, _ALIASES.get(v.replace("_", " "), VERDICT_NEI))


def parse_verdict(raw: str) -> dict:
    """
    Estrae {verdict, reasoning} dalla risposta del modello.

    Prova il JSON e, se il modello lo ha sporcato, ripiega sulla ricerca di
    un'etichetta nel testo: e' l'unica parte del sistema che dipende dal
    comportamento di un generativo, quindi non deve poter sollevare eccezioni.
    """
    if not raw or not raw.strip():
        return {"verdict": VERDICT_NEI, "reasoning": ""}
    try:
        obj = json.loads(raw)
        return {"verdict": canon_verdict(str(obj.get("verdict", ""))),
                "reasoning": str(obj.get("reasoning", ""))[:300]}
    except (json.JSONDecodeError, AttributeError):
        up = raw.upper()
        return {"verdict": next((c for c in _VALID if c in up), VERDICT_NEI),
                "reasoning": raw.strip()[:200]}


def _format_prompt(claim: str, evidences: list[str]) -> str:
    """Evidenze numerate, cosi' la motivazione puo' citarle per indice."""
    righe = "\n".join(f"[{i + 1}] {e.strip()}" for i, e in enumerate(evidences))
    return f"CLAIM: {claim}\n\nEVIDENCE:\n{righe}"


class LlmVerdict:
    """Motore di verdetto via LLM locale (Ollama), a temperatura 0 per riproducibilita'."""

    def __init__(self, config_path: Optional[str] = None):
        cfg = load_section(config_path, "llm_fact_checking")
        self.model = cfg.get("model", "qwen2.5:7b")
        self.host = cfg.get("ollama_host", "http://localhost:11434")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout_s = int(cfg.get("timeout_s", 180))
        self.max_evidence = int(cfg.get("evidence_top_k", 5))
        logger.info("LlmVerdict | model=%s | host=%s", self.model, self.host)

    def verdict(self, claim: str, evidences: list[str]) -> dict:
        """Verdetto sulla claim date le evidenze. Firma condivisa con NliVerdict."""
        evidences = [e for e in evidences if e and e.strip()][: self.max_evidence]
        if not evidences:
            return {"verdict": VERDICT_NEI, "reasoning": ""}
        return parse_verdict(self._chat(_format_prompt(claim, evidences)))

    def _chat(self, user: str) -> str:
        """Una chiamata a Ollama; qualunque problema di rete -> stringa vuota -> NEI."""
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",                       # il server vincola l'output a JSON valido
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            self.host + "/api/chat", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.load(r)["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            logger.warning("Ollama non raggiungibile (%s): verdetto -> NEI", e)
            return ""
