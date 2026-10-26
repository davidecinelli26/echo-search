"""
Echo-Search | Utils — scelta del device Torch
=============================================
Un solo punto in cui si decide fra GPU e CPU, condiviso da tutti i moduli che
caricano un modello. Duplicare questa riga in ogni costruttore significherebbe
poter finire con meta' pipeline su GPU e meta' su CPU senza accorgersene.
"""

from typing import Optional

import torch


def resolve_device(device: Optional[str] = None) -> str:
    """
    Restituisce il device su cui eseguire l'inference.

    Un valore esplicito ("cuda" | "cpu") ha sempre precedenza; con None
    viene fatto l'auto-detect della GPU.
    """
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
