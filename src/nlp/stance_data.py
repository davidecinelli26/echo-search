"""
Echo-Search | Fase 3 — Dati di Stance (utility condivise)
=========================================================
Caricamento degli split di stance e split stratificato train/validation,
riusati da scripts/train_stance.py e scripts/eval_stance.py.

SCELTE ARCHITETTURALI:
----------------------
- MAPPING LABEL UNICO: definito qui una sola volta (LABEL2ID/ID2LABEL) ed
  esportato a training, modulo e valutazione. Evita disallineamenti tra
  gli id del classificatore e le stringhe di stance.
- SPLIT STRATIFICATO DETERMINISTICO: la validation preserva le proporzioni
  di classe (Macro-F1 su classi sbilanciate è sensibile alla composizione)
  ed è riproducibile via seed (requisito di riproducibilità del progetto).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Fonte di verità del mapping label (ordine per id crescente).
LABEL2ID: dict[str, int] = {"AGAINST": 0, "FAVOR": 1, "NEUTRAL": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def load_stance_jsonl(path: str | Path) -> list[dict]:
    """Carica uno split jsonl (una riga JSON per esempio: {text, target, label, source})."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(
    rows: list[dict],
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Divide `rows` in (train, val) mantenendo le proporzioni per label.
    Per ogni classe estrae round(n_classe * val_frac) esempi nella validation.
    Deterministico a parità di seed.
    """
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for label in sorted(by_label):  # ordine stabile tra le classi
        items = by_label[label][:]
        rng.shuffle(items)
        n_val = round(len(items) * val_frac)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val
