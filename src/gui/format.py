"""
Echo-Search | GUI — decisioni di presentazione, isolate dal rendering
====================================================================
Le poche scelte della dashboard che contengono una decisione (quanto allargare
una colonna, quale badge mostrare, quale fascia di colore) stanno qui come
funzioni PURE: dati in ingresso, valore in uscita, nessuna dipendenza da
Streamlit.

Il motivo e' la verificabilita'. Streamlit non e' testabile in unita', quindi
tutto cio' che si puo' sbagliare in silenzio — per esempio far sparire una
colonna invece di restringerla — va tenuto fuori dal codice che disegna.
"""

_LABELS = ["FAVOR", "NEUTRAL", "AGAINST"]


def column_widths(focus):
    """Larghezze delle 3 colonne: uguali, o 6 alla fazione scelta e 2 alle altre (sempre visibili)."""
    if focus not in _LABELS:
        return [1, 1, 1]
    return [6 if lab == focus else 2 for lab in _LABELS]


def verdict_badge(factuality):
    """Badge del doc dai conteggi di fact-check; priorità REFUTED > SUPPORTED > NEI; None se nessuna claim."""
    f = factuality or {}
    if f.get("REFUTED", 0) > 0:
        return {"icon": "✕", "label": "REFUTED", "tone": "danger"}
    if f.get("SUPPORTED", 0) > 0:
        return {"icon": "✓", "label": "SUPPORTED", "tone": "success"}
    if f.get("NOT_ENOUGH_EVIDENCE", 0) > 0:
        return {"icon": "–", "label": "NOT_ENOUGH_EVIDENCE", "tone": "neutral"}
    return None


def truncate(text, n):
    """Anteprima: taglia a n caratteri con ellissi."""
    text = text or ""
    return text if len(text) <= n else text[:n].rstrip() + "…"






def diversity_tone(score):
    """Fascia colore dello Stance Diversity Score: >=0.6 bilanciato, >=0.3 parziale, altrimenti echo."""
    if score >= 0.6:
        return "success"
    if score >= 0.3:
        return "warning"
    return "danger"
