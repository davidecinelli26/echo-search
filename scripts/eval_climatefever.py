"""
Echo-Search | Fase 4 — Valutazione del fact-checker su ClimateFEVER (offline, no API).
====================================================================================
Due metriche, con l'NLI scelto via --nli-model (default: DeBERTa-v3-base-mnli-fever-anli):
- PER-EVIDENZA: l'NLI predice il label di OGNI coppia (claim, evidenza). È la
  misura pulita del modello (task NLI puro), senza aggregazione.
- CLAIM-LEVEL: verdetto per claim = argmax della probabilità NLI media sulle
  evidenze; confronto col label del claim (DISPUTED escluso). È il verdetto
  end-to-end del sistema (piu' difficile).
Metriche: Accuracy + Macro-F1 a 3 classi. Riusa i motori di verdetto di src/nlp/.

Uso: uv run --no-sync --with "torch==2.2.2" --with "datasets==5.0.0" --index cu121 python scripts/eval_climatefever.py
     (opzione: --nli-model <hf_id> per confrontare modelli NLI diversi)
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.evaluation.metrics import compute_macro_f1
from src.utils.logger import setup_logging
# I due motori di verdetto sono importati lazy nei rispettivi runner:
# valutare l'LLM non deve caricare torch, e viceversa.

logger = logging.getLogger(__name__)

# claim_label: 0=SUPPORTS 1=REFUTES 2=NOT_ENOUGH_INFO 3=DISPUTED (escluso claim-level)
_CLAIM = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT_ENOUGH_EVIDENCE"}
# evidence_label: 0=SUPPORTS 1=REFUTES 2=NOT_ENOUGH_INFO
_EVID = {0: "SUPPORTED", 1: "REFUTED", 2: "NOT_ENOUGH_EVIDENCE"}
_LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_EVIDENCE"]


def verdict(probs: np.ndarray, entail_idx: int, contra_idx: int) -> str:
    """argmax della probabilità NLI media (per 1 riga = quella riga)."""
    top = int(probs.mean(axis=0).argmax())
    if top == entail_idx:
        return "SUPPORTED"
    if top == contra_idx:
        return "REFUTED"
    return "NOT_ENOUGH_EVIDENCE"


def _report(name: str, y_true: list, y_pred: list) -> None:
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    res = compute_macro_f1(y_true, y_pred, labels=_LABELS)
    per = {k: v["f1"] for k, v in res["per_class"].items()}
    print(f"RESULT {name} n={len(y_true)} accuracy={round(acc, 4)} macro_f1={res['macro_f1']} per_class={per}")


def run_nli(ds, args) -> None:
    """Engine NLI (DeBERTa-fever): per-evidenza + claim-level via entailment/contraddizione."""
    from src.nlp.nli_verdict import NliVerdict  # torch/transformers, lazy)

    # config_path=None -> si usa ESATTAMENTE --nli-model, senza override dal config
    engine = NliVerdict(nli_model=args.nli_model)
    _ = engine.model                            # carica il modello e imposta gli indici
    logger.info("NLI valutato: %s | entail_idx=%d contra_idx=%d",
                engine.model_name, engine.entail_idx, engine.contra_idx)

    ev_true, ev_pred, cl_true, cl_pred = [], [], [], []
    for i, row in enumerate(ds):
        if args.max_claims and i >= args.max_claims:
            break
        evs = [e for e in row["evidences"] if e.get("evidence")]
        if not evs:
            continue
        probs = engine.probabilities([e["evidence"] for e in evs], [row["claim"]] * len(evs))
        for e, p in zip(evs, probs):            # per-evidenza
            if e["evidence_label"] in _EVID:
                ev_true.append(_EVID[e["evidence_label"]])
                ev_pred.append(verdict(p[None, :], engine.entail_idx, engine.contra_idx))
        if row["claim_label"] != 3:             # claim-level (no DISPUTED)
            cl_true.append(_CLAIM[row["claim_label"]])
            cl_pred.append(verdict(probs, engine.entail_idx, engine.contra_idx))

    print(f"RESULT model {engine.model_name} (nli)")
    _report("per_evidence", ev_true, ev_pred)
    _report("claim_level", cl_true, cl_pred)


def run_llm(ds, args) -> None:
    """Engine LLM (Ollama): l'LLM ragiona sulle evidenze e dà il verdetto.

    Claim-level su TUTTE le evidenze (il confronto significativo). Per-evidenza
    solo con --per-evidence (1 chiamata per evidenza -> ORE su ClimateFEVER intero).
    """
    from src.nlp.llm_verdict import LlmVerdict  # solo Ollama, no torch)

    engine = LlmVerdict(config_path="configs/config.yaml")
    if args.llm_model:
        engine.model = args.llm_model
    logger.info("LLM valutato: %s (host %s)", engine.model, engine.host)

    ev_true, ev_pred, cl_true, cl_pred = [], [], [], []
    t0 = time.time()
    seen = 0
    for i, row in enumerate(ds):
        if args.max_claims and i >= args.max_claims:
            break
        evs = [e for e in row["evidences"] if e.get("evidence")]
        if not evs:
            continue
        if row["claim_label"] != 3:             # claim-level (no DISPUTED)
            out = engine.verdict(row["claim"], [e["evidence"] for e in evs])
            cl_true.append(_CLAIM[row["claim_label"]])
            cl_pred.append(out["verdict"])
        if args.per_evidence:                   # LENTO: 1 chiamata LLM per evidenza
            for e in evs:
                if e["evidence_label"] in _EVID:
                    o = engine.verdict(row["claim"], [e["evidence"]])
                    ev_true.append(_EVID[e["evidence_label"]])
                    ev_pred.append(o["verdict"])
        seen += 1
        if seen % 50 == 0:
            rate = (time.time() - t0) / seen
            logger.info("LLM: %d claim (%.1fs/claim, ~%.0f min stimati per il resto)",
                        seen, rate, rate * (len(ds) - i) / 60)

    print(f"RESULT model {engine.model} (llm)")
    if ev_true:
        _report("per_evidence", ev_true, ev_pred)
    _report("claim_level", cl_true, cl_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta il fact-checker su ClimateFEVER")
    parser.add_argument("--engine", choices=["nli", "llm"], default="nli",
                        help="motore di verdetto: nli (DeBERTa-fever) o llm (Ollama)")
    parser.add_argument("--max-claims", type=int, default=0, help="0 = tutte")
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help="(engine nli) ID HuggingFace del modello NLI (default: DeBERTa-v3-base-fever)",
    )
    parser.add_argument("--llm-model", default=None,
                        help="(engine llm) override del modello Ollama (default: da config)")
    parser.add_argument("--per-evidence", action="store_true",
                        help="(engine llm) valuta anche per-evidenza (LENTO)")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    from datasets import load_dataset
    ds = load_dataset("tdiggelm/climate_fever", split="test")
    logger.info("ClimateFEVER: %d claim | engine=%s", len(ds), args.engine)

    if args.engine == "nli":
        run_nli(ds, args)
    else:
        run_llm(ds, args)


if __name__ == "__main__":
    main()
