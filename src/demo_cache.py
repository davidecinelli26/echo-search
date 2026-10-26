"""
Echo-Search | Cache offline dei risultati della demo.
=====================================================
Corpus e query canoniche sono FISSI e i modelli DETERMINISTICI: i risultati non
cambiano mai. Li precalcoliamo una volta (scripts/precompute_demo.py) e la GUI
li rilegge da JSON in millisecondi (nessun modello, nessuna GPU).

Chiave = (topic, query, fact_check, summarize, engine): la stessa firma di
`run_search` nella dashboard, così cache e live combaciano esattamente.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("data/demo_cache")


def _key(query: str, fact_check: bool, summarize: bool, engine: str) -> str:
    raw = f"{query.strip()}|fc={int(fact_check)}|sum={int(summarize)}|eng={engine}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def cache_path(topic: str, query: str, fact_check: bool, summarize: bool, engine: str) -> Path:
    return CACHE_DIR / topic / f"{_key(query, fact_check, summarize, engine)}.json"


def load_cached(topic: str, query: str, fact_check: bool, summarize: bool, engine: str) -> Optional[dict]:
    """Ritorna la response precalcolata se esiste, altrimenti None (fallback live)."""
    p = cache_path(topic, query, fact_check, summarize, engine)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _json_default(o):
    # scalari NumPy (float32/int64) -> scalare Python; resto -> stringa (fail-safe)
    return o.item() if hasattr(o, "item") else str(o)


def save_cached(topic: str, response: dict, query: str, fact_check: bool,
                summarize: bool, engine: str) -> Path:
    p = cache_path(topic, query, fact_check, summarize, engine)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(response, ensure_ascii=False, default=_json_default),
                 encoding="utf-8")
    return p
