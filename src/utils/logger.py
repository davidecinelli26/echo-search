"""
Echo-Search | Utils — logging centralizzato
===========================================
Un formato unico per tutti i punti d'ingresso (CLI, dashboard, script di
valutazione). I moduli di libreria non configurano mai il logging: si limitano a
`logging.getLogger(__name__)`, cosi' chi li importa resta padrone dell'output.
"""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configura il root logger con il formato standard del progetto.
    Idempotente: logging.basicConfig è un no-op se il root logger
    ha già degli handler configurati.
    """
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
