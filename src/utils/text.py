"""
Echo-Search | Utils — Segmentazione in frasi condivisa
=======================================================
Splitter regex leggero (nessuna dipendenza pesante), condiviso da:
- MultiPerspectiveSummarizer (fase estrattiva)
- FactChecker (costruzione dell'evidence pool)

La SIMMETRIA tra i due moduli è deliberata: se summarizer e fact-checker
segmentassero le stesse frasi in modo diverso, le evidenze citate nei
riassunti non sarebbero allineate con quelle verificate dal fact-checking.

Per produzione: sostituire con lo spaCy sentencizer (più accurato su
abbreviazioni tipo "Dr.", "U.S."), mantenendo questa firma.
"""

import re

# Split su . ! ? seguiti da spazio + maiuscola (euristica robusta su testo
# giornalistico in inglese; evita di spezzare su decimali tipo "3.5")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_into_sentences(text: str) -> list[str]:
    """Segmenta il testo in frasi, scartando frammenti vuoti."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
