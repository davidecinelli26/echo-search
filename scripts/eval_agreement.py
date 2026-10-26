"""
Echo-Search | Accordo fra annotatori sui qrels.
===============================================
Calcola il kappa di Cohen fra DUE serie di giudizi di rilevanza prodotte
indipendentemente da due annotatori sulle stesse coppie (query, documento),
per tema e sul complessivo.

SCELTE METODOLOGICHE:
- SOLO L'INTERSEZIONE: si confrontano le coppie giudicate da ENTRAMBI. Una coppia
  giudicata da uno solo non e' un disaccordo, e' un dato mancante; trattarla come
  zero gonfierebbe o affosserebbe il kappa a seconda dei marginali.
- KAPPA QUADRATICO COME VALORE PRINCIPALE: la scala 0/1/2 e' ORDINALE. Il kappa
  non pesato tratta "0 contro 1" e "0 contro 2" come lo stesso errore, che e'
  falso: il primo e' un confine sfumato, il secondo una contraddizione. Si
  riportano comunque tutte e tre le varianti, perche' la non pesata e' quella
  che la letteratura cita piu' spesso e ometterla renderebbe il numero non
  confrontabile.
- ETICHETTE DICHIARATE, NON DEDOTTE: labels=["0","1","2"] fisso. Se un tema non
  usa mai il grado 2, dedurre le etichette dai dati cambierebbe la matrice di
  pesi (K=2 invece di 3) e i kappa dei temi non sarebbero piu' confrontabili fra
  loro.
- CONFLITTI A DUE GRADINI CONTATI A PARTE: sono la patologia che conta davvero
  (un giudice dice "non pertinente", l'altro "pienamente pertinente"); l'accordo
  medio puo' essere buono e nascondere comunque un pugno di casi cosi'.
- MARGINALI RIPORTATI: se un annotatore e' sistematicamente piu' generoso, le
  metriche di ranking calcolate sulle sue qrels sono piu' alte in valore
  assoluto anche a ordinamento invariato. Il dato serve a leggere l'ablazione.

Uso:
    uv run --no-sync python scripts/eval_agreement.py
    uv run --no-sync python scripts/eval_agreement.py --annotators davide,giuseppe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.evaluation.metrics import compute_cohen_kappa
from src.evaluation.qrels import load_qrels_for_topics
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

LABELS = ["0", "1", "2"]
WEIGHTS = ("unweighted", "linear", "quadratic")


# ---------------------------------------------------------------------------
# Logica pura
# ---------------------------------------------------------------------------

def align(qrels_a: dict[str, dict[str, int]],
          qrels_b: dict[str, dict[str, int]]) -> tuple[list[str], list[str], dict]:
    """
    Appaia i giudizi dei due annotatori sulle coppie (query, doc_id) comuni.

    Returns
    -------
    (y_a, y_b, coverage)
        Etichette come stringhe (la scala e' categoriale ordinata, non numerica),
        allineate per posizione e in ordine deterministico. `coverage` riporta
        quante coppie ha giudicato ciascuno e quante sono in comune: se il divario
        e' grande, il kappa e' calcolato su un sottoinsieme e va detto.
    """
    pairs_a = {(q, d) for q, docs in qrels_a.items() for d in docs}
    pairs_b = {(q, d) for q, docs in qrels_b.items() for d in docs}
    common = sorted(pairs_a & pairs_b)
    y_a = [str(qrels_a[q][d]) for q, d in common]
    y_b = [str(qrels_b[q][d]) for q, d in common]
    coverage = {
        "judged_a": len(pairs_a),
        "judged_b": len(pairs_b),
        "common": len(common),
        "only_a": len(pairs_a - pairs_b),
        "only_b": len(pairs_b - pairs_a),
    }
    return y_a, y_b, coverage


def agreement_stats(y_a: list[str], y_b: list[str]) -> dict:
    """Tutte le varianti di kappa, piu' marginali e conflitti a due gradini."""
    by_weight = {w: compute_cohen_kappa(y_a, y_b, labels=LABELS, weights=w) for w in WEIGHTS}
    base = by_weight["unweighted"]
    two_step = sum(1 for a, b in zip(y_a, y_b) if abs(int(a) - int(b)) == 2)
    return {
        "n": base["n"],
        "observed_agreement": base["observed_agreement"],
        "kappa": {w: by_weight[w]["kappa"] for w in WEIGHTS},
        "expected_agreement": {w: by_weight[w]["expected_agreement"] for w in WEIGHTS},
        "two_step_conflicts": two_step,
        "confusion": base["confusion"],
        "labels": LABELS,
        "marginals": {
            "a": {lab: y_a.count(lab) for lab in LABELS},
            "b": {lab: y_b.count(lab) for lab in LABELS},
        },
    }


def interpret(kappa: float) -> str:
    """Scala di Landis e Koch (1977), riportata per convenzione e non come verita'."""
    if kappa != kappa:  # nan
        return "non definito"
    for soglia, etichetta in ((0.81, "almost perfect"), (0.61, "substantial"),
                              (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if kappa >= soglia:
            return etichetta
    return "poor"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(report: dict, a: str, b: str) -> None:
    print(f"\n{'=' * 78}\nACCORDO FRA ANNOTATORI: {a} vs {b}\n{'=' * 78}")
    print(f"{'Tema':<24} {'n':>5} {'accordo':>9} {'k non pes.':>11} "
          f"{'k lineare':>10} {'k quad.':>9} {'2 gradini':>10}")
    print("-" * 78)
    for tema, st in list(report["per_topic"].items()) + [("COMPLESSIVO", report["overall"])]:
        if tema == "COMPLESSIVO":
            print("-" * 78)
        print(f"{tema:<24} {st['n']:>5} {st['observed_agreement']:>9.3f} "
              f"{st['kappa']['unweighted']:>11.3f} {st['kappa']['linear']:>10.3f} "
              f"{st['kappa']['quadratic']:>9.3f} {st['two_step_conflicts']:>10}")

    overall = report["overall"]
    print(f"\nInterpretazione (Landis-Koch) del kappa quadratico complessivo: "
          f"{interpret(overall['kappa']['quadratic'])}")

    print(f"\nDistribuzione dei gradi assegnati (su {overall['n']} coppie comuni):")
    for nome, chiave in ((a, "a"), (b, "b")):
        m = overall["marginals"][chiave]
        tot = sum(m.values())
        quota = "  ".join(f"{lab}: {m[lab]:>4} ({m[lab] / tot:.1%})" for lab in LABELS)
        print(f"  {nome:<12} {quota}")

    print(f"\nMatrice di confusione complessiva (righe = {a}, colonne = {b}):")
    print(f"{'':>10}" + "".join(f"{lab:>8}" for lab in LABELS))
    for lab, riga in zip(LABELS, overall["confusion"]):
        print(f"{lab:>10}" + "".join(f"{v:>8}" for v in riga))

    cov = report["coverage"]
    if cov["only_a"] or cov["only_b"]:
        print(f"\nATTENZIONE: coppie giudicate da uno solo — {a}: {cov['only_a']}, "
              f"{b}: {cov['only_b']}. Il kappa e' calcolato sulle {cov['common']} comuni.")


def main() -> None:
    p = argparse.ArgumentParser(description="Kappa di Cohen fra due serie di qrels")
    p.add_argument("--annotators", default="davide,giuseppe",
                   help="i due annotatori da confrontare, separati da virgola")
    p.add_argument("--topics", default="configs/topics.yaml")
    p.add_argument("--gold-dir", default="data/gold")
    p.add_argument("--out", default="data/gold/agreement_report.json")
    args = p.parse_args()
    setup_logging(logging.INFO)

    names = [x.strip() for x in args.annotators.split(",") if x.strip()]
    if len(names) != 2:
        sys.exit(f"servono esattamente due annotatori, ricevuti {names}")
    a, b = names

    topics = list(load_config(args.topics)["topics"])
    qrels_a = load_qrels_for_topics(topics, a, args.gold_dir)
    qrels_b = load_qrels_for_topics(topics, b, args.gold_dir)
    comuni = [t for t in topics if t in qrels_a and t in qrels_b]
    if not comuni:
        sys.exit(f"nessun tema annotato da entrambi ({a}, {b})")

    report = {"annotators": [a, b], "per_topic": {}, "labels": LABELS}
    y_a_all: list[str] = []
    y_b_all: list[str] = []
    cov_all = {"judged_a": 0, "judged_b": 0, "common": 0, "only_a": 0, "only_b": 0}

    for tema in comuni:
        y_a, y_b, cov = align(qrels_a[tema], qrels_b[tema])
        report["per_topic"][tema] = agreement_stats(y_a, y_b)
        y_a_all.extend(y_a)
        y_b_all.extend(y_b)
        for k in cov_all:
            cov_all[k] += cov[k]
        logger.info("[%s] %d coppie comuni (%s: %d, %s: %d)",
                    tema, cov["common"], a, cov["judged_a"], b, cov["judged_b"])

    report["overall"] = agreement_stats(y_a_all, y_b_all)
    report["overall"]["interpretation"] = interpret(report["overall"]["kappa"]["quadratic"])
    report["coverage"] = cov_all

    print_report(report, a, b)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport salvato in {args.out}")


if __name__ == "__main__":
    main()
