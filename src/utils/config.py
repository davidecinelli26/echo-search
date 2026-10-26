"""
Echo-Search | Utils — Caricamento centralizzato della configurazione YAML
==========================================================================
Elimina il boilerplate ripetuto in ogni modulo:

    if config_path:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        section = cfg.get("sezione", {})

SCELTE:
- Cache per path assoluto (lru_cache): config.yaml viene letto e parsato UNA
  volta per processo, anche se richiesto da N moduli (Sparse, Dense, Hybrid,
  Stance, FactChecker, Summarizer condividono lo stesso file).
- encoding="utf-8" esplicito: il file contiene caratteri non-ASCII (α, β, ≈)
  e su Windows l'encoding di default non è UTF-8.
- load_section accetta config_path=None e restituisce {} → i costruttori
  possono eliminare il ramo `if config_path:` mantenendo i propri default.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml


@lru_cache(maxsize=None)
def _read_yaml(resolved_path: str) -> dict:
    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str) -> dict:
    """Intera configurazione come dict (cache-ata per path assoluto)."""
    return _read_yaml(str(Path(config_path).resolve()))


def load_section(config_path: Optional[str], section: str) -> dict[str, Any]:
    """
    Restituisce una sezione della config come dict (copia superficiale,
    sicura da mutare). Con config_path=None restituisce {} così i default
    dei costruttori restano attivi.
    """
    if not config_path:
        return {}
    return dict(load_config(config_path).get(section) or {})
