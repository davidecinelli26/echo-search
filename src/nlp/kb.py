"""
Echo-Search | Knowledge base — lettura degli articoli Wikipedia
==============================================================
Carica la base di conoscenza di un tema, costruita offline da
`scripts/build_kb.py` e usata come unica fonte di evidenze del fact-checking.

La KB e' una fonte ESTERNA e fidata, non i documenti recuperati. Verificare un
articolo contro gli altri articoli dello stesso corpus misurerebbe il consenso
della stampa, non la verita': su un tema divisivo le due cose divergono, ed e'
esattamente il caso che il progetto studia.
"""

import json
from pathlib import Path


def load_kb(topic: str, kb_dir: str | Path = "data/kb") -> list[dict]:
    """Articoli KB del tema come [{doc_id, text, title, url}] (evidence-pool)."""
    path = Path(kb_dir) / f"wikipedia_{topic}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"KB non trovato per '{topic}': {path}")
    docs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                docs.append({"doc_id": d["doc_id"], "text": d["text"],
                             "title": d.get("title", ""), "url": d.get("url", "")})
    return docs
