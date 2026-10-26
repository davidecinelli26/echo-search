"""
Echo-Search | Fase 3 — Infrastruttura del Gold Set di Stance
============================================================
Campiona ~200-300 documenti per tema dai corpora CC-News (data/corpora/) per
l'annotazione manuale del gold set, con un sottoinsieme in OVERLAP per il calcolo
di Cohen's kappa. `--load` ricarica i fogli annotati in data/gold/.

Solo infrastruttura: l'annotazione (label FAVOR/AGAINST/NEUTRAL) è lavoro umano.

SCELTE ARCHITETTURALI:
----------------------
- CAMPIONAMENTO DETERMINISTICO (seed): il gold set è riproducibile e citabile.
- OVERLAP ESPLICITO: una frazione dei documenti va annotata da entrambi gli
  annotatori → misura di affidabilità (Cohen's kappa), requisito del piano.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

_THEMES = ("nuclear_energy", "climate_change", "vaccines")


def sample_for_annotation(
    docs: list[dict], n_sample: int, overlap_frac: float, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Estrae n_sample documenti (cap sul disponibile) + il sottoinsieme in overlap."""
    rng = random.Random(seed)
    pool = docs[:]
    rng.shuffle(pool)
    sheet = pool[: min(n_sample, len(pool))]
    n_overlap = round(len(sheet) * overlap_frac)
    return sheet, sheet[:n_overlap]


def _write_sheet(sheet: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "title", "text", "label"])  # label da riempire a mano
        for d in sheet:
            w.writerow([d["doc_id"], d.get("title", ""), d["text"], ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera/ricarica i fogli del gold set")
    parser.add_argument("--corpora-dir", default="data/corpora")
    parser.add_argument("--out-dir", default="data/gold")
    parser.add_argument("--n-sample", type=int, default=250)
    parser.add_argument("--overlap-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load", action="store_true",
                        help="Ricarica i CSV annotati (data/gold/*_annotated.csv) in stance_{tema}.jsonl")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.load:
        for theme in _THEMES:
            csv_path = out / f"{theme}_annotated.csv"
            if not csv_path.exists():
                logger.warning(f"Manca {csv_path}, salto.")
                continue
            rows = []
            with csv_path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("label", "").strip():
                        rows.append({"text": r["text"], "target": _target_of(theme),
                                     "label": r["label"].strip().upper(), "source": "gold"})
            (out / f"stance_{theme}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            logger.info(f"{theme}: {len(rows)} esempi annotati → stance_{theme}.jsonl")
        return

    for theme in _THEMES:
        docs = [json.loads(l) for l in
                (Path(args.corpora_dir) / f"{theme}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        sheet, overlap = sample_for_annotation(docs, args.n_sample, args.overlap_frac, args.seed)
        _write_sheet(sheet, out / f"{theme}_to_annotate.csv")
        (out / f"{theme}_overlap_ids.json").write_text(
            json.dumps([d["doc_id"] for d in overlap], indent=2), encoding="utf-8")
        logger.info(f"{theme}: {len(sheet)} da annotare ({len(overlap)} in overlap) → {theme}_to_annotate.csv")


def _target_of(theme: str) -> str:
    return {"nuclear_energy": "nuclear energy", "climate_change": "climate change",
            "vaccines": "vaccine safety"}[theme]


if __name__ == "__main__":
    main()
