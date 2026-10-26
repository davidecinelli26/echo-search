"""
Echo-Search | Evaluation — Caricamento dei giudizi di rilevanza (qrels)
=======================================================================
I fogli annotati in data/gold/ hanno una riga per coppia (query, documento) con
una colonna `relevance` compilata a mano (0 = non pertinente, 1 = marginale,
2 = pienamente pertinente). Qui si trasformano nella struttura che le metriche
di ranking si aspettano: {query: {doc_id: rilevanza}}.

SCELTE:
- I/O SEPARATO DALLE METRICHE: src/evaluation/metrics.py resta puro (nessun
  accesso al filesystem), così le metriche sono testabili senza dati su disco.
- RIGHE NON COMPILATE SCARTATE, non trattate come 0: un foglio parzialmente
  annotato deve produrre qrels parziali, non qrels sbagliati. Il conteggio delle
  righe scartate è restituito al chiamante, che decide se è accettabile.
- CONVENZIONE DI CRANFIELD: i documenti mai entrati nel pool sono assenti dai
  qrels e le metriche li trattano come non rilevanti. È l'assunzione standard e
  vale solo se il pool è l'unione dei run confrontati (vedi
  scripts/build_qrels_annotation.py).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_GRADES = {"0", "1", "2"}


def parse_qrels_rows(rows) -> tuple[dict[str, dict[str, int]], int]:
    """
    Trasforma righe di foglio in {query: {doc_id: rilevanza}} (logica pura).

    Returns
    -------
    (qrels, n_skipped)
        `n_skipped` conta le righe con `relevance` vuota: sono coppie non ancora
        giudicate, non giudizi di non-rilevanza.
    """
    qrels: dict[str, dict[str, int]] = {}
    skipped = 0
    for i, row in enumerate(rows, start=2):  # riga 1 = header
        grade = (row.get("relevance") or "").strip()
        if not grade:
            skipped += 1
            continue
        if grade not in VALID_GRADES:
            raise ValueError(
                f"riga {i}: relevance {grade!r} non valida (attese {sorted(VALID_GRADES)})"
            )
        query = row["query"]
        doc_id = row["doc_id"]
        previous = qrels.setdefault(query, {}).get(doc_id)
        if previous is not None and previous != int(grade):
            raise ValueError(
                f"riga {i}: giudizio contraddittorio per ({query!r}, {doc_id}): "
                f"{previous} poi {grade}"
            )
        qrels[query][doc_id] = int(grade)
    return qrels, skipped


def load_qrels(path: str | Path) -> tuple[dict[str, dict[str, int]], int]:
    """Carica un singolo foglio qrels. I file sono in utf-8-sig (compatibili Excel)."""
    with Path(path).open(encoding="utf-8-sig") as f:
        return parse_qrels_rows(list(csv.DictReader(f)))


def load_qrels_for_topics(
    topics, annotator: str, gold_dir: str | Path = "data/gold",
) -> dict[str, dict[str, dict[str, int]]]:
    """
    Carica i qrels di un annotatore per più temi.

    Returns
    -------
    dict
        {tema: {query: {doc_id: rilevanza}}}. I temi senza foglio sono omessi.
    """
    gold_dir = Path(gold_dir)
    out: dict[str, dict[str, dict[str, int]]] = {}
    for topic in topics:
        path = gold_dir / f"qrels_{topic}_{annotator}.csv"
        if not path.exists():
            logger.warning("[%s] foglio %s assente -> salto", topic, path.name)
            continue
        qrels, skipped = load_qrels(path)
        if skipped:
            logger.warning("[%s] %d coppie non ancora giudicate: escluse dai qrels",
                           topic, skipped)
        n_judged = sum(len(v) for v in qrels.values())
        logger.info("[%s] %d query, %d giudizi (%s)", topic, len(qrels), n_judged, annotator)
        out[topic] = qrels
    return out


def relevant_count(qrels_for_query: dict[str, int]) -> int:
    """Numero di documenti rilevanti (grado > 0) per una query."""
    return sum(1 for grade in qrels_for_query.values() if grade > 0)
