"""
Echo-Search | Fase 1 (1.3) — Knowledge Base fidata (Wikipedia mirata)
=====================================================================
Costruisce la fonte di VERITÀ Tier-1 del fact-checker: per ogni tema cerca su
Wikipedia gli articoli rilevanti (usando le query canoniche di topics.yaml) e
ne scarica l'introduzione (riassunto autorevole dell'articolo). È "mirata" —
non il dump da 20GB, ma poche decine di articoli per tema.

SCELTE:
- Wikipedia via API (solo urllib stdlib): copertura verificata dei temi,
  inclusi gli articoli che smentiscono le claim note ("Vaccines and autism").
- FETCH BATCHED: l'API `extracts` restituisce fino a 20 intro per richiesta.
  Passiamo da ~1 richiesta/articolo a ~1 ogni 20 → rispettiamo il rate limit
  di Wikipedia (che ci ha già messo in cooldown con richieste troppo fitte).
- INTRO invece del full text: l'introduzione riassume i fatti salienti ed è
  sufficiente come evidenza; evita risposte enormi e altre richieste. Il full
  text si può aggiungere in seguito, per articolo, se servisse più granularità.
- Evidenze tracciabili all'URL canonico (explainability, slide 9).

Output (in `data/kb/`):
- `wikipedia_{topic}.jsonl` — un articolo per riga (intro)
- `MANIFEST.json`          — conteggi, hash, titoli

Uso:
    uv run --no-sync python scripts/build_kb.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

_API = "https://en.wikipedia.org/w/api.php"
# User-Agent CONFORME alle policy Wikimedia: include un URL di contatto reale.
# Senza di esso il client è "non identificato" e limitato a 10 req/min (invece di 200).
_UA = "EchoSearch/0.1 (https://github.com/davidecinelli26/echo-search) Python-urllib"
_MIN_WORDS = 40          # scarta stub/disambigua
_PAUSE_S = 1.0           # ~60 req/min, ben sotto il limite di 200/min (UA conforme)
_BATCH = 20              # max titoli per richiesta extracts

# Categorie Wikipedia seed per tema: danno copertura AMPIA (centinaia di articoli)
# oltre alle query canoniche. Si esplorano anche le sottocategorie (depth 1).
_SEED_CATEGORIES = {
    "nuclear_energy": ["Nuclear power", "Nuclear safety and security",
                       "Nuclear power accidents", "Nuclear energy policy", "Radioactive waste"],
    "climate_change": ["Climate change", "Global warming", "Effects of climate change",
                       "Climate change policy", "Climate change denial"],
    "vaccines": ["Vaccination", "Vaccines", "Vaccine hesitancy", "Vaccine-preventable diseases"],
}


def category_members(category: str, depth: int = 1, limit: int = 600) -> list[str]:
    """Titoli degli articoli (ns=0) in una categoria Wikipedia + sottocategorie (depth-limitata)."""
    titles: list[str] = []
    subcats: list[str] = []
    cont: dict = {}
    while len(titles) < limit:
        data = _api({"action": "query", "list": "categorymembers",
                     "cmtitle": f"Category:{category}", "cmlimit": 500,
                     "cmtype": "page|subcat", **cont})
        for m in data.get("query", {}).get("categorymembers", []):
            if m["ns"] == 0:
                titles.append(m["title"])
            elif m["ns"] == 14 and depth > 0:
                subcats.append(m["title"].replace("Category:", ""))
        cont = data.get("continue", {})
        if not cont:
            break
        time.sleep(_PAUSE_S)
    if depth > 0:
        for sub in subcats[:15]:            # limita l'esplosione delle sottocategorie
            if len(titles) >= limit:
                break
            titles.extend(category_members(sub, depth - 1, limit=120))
            time.sleep(_PAUSE_S)
    return titles[:limit]


def _api(params: dict, retries: int = 6) -> dict:
    url = _API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429 (rate limit) / 503 (overload): rispetta l'header Retry-After se
            # presente (best practice Wikimedia), altrimenti backoff esponenziale ≥5s.
            if e.code in (429, 503) and attempt < retries - 1:
                header = e.headers.get("Retry-After", "")
                wait = int(header) if header.isdigit() else max(5, 2 ** attempt)
                logger.warning(f"HTTP {e.code} — Retry-After={header or 'n/d'}, attendo {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("unreachable")


def search_titles(query: str, limit: int) -> list[str]:
    data = _api({"action": "query", "list": "search", "srsearch": query, "srlimit": limit})
    return [h["title"] for h in data.get("query", {}).get("search", [])]


def fetch_intros(titles: list[str]) -> list[dict]:
    """Scarica in batch (intro + URL) le pagine dei titoli dati."""
    pages: list[dict] = []
    for i in range(0, len(titles), _BATCH):
        chunk = titles[i:i + _BATCH]
        data = _api({
            "action": "query", "prop": "extracts|info", "titles": "|".join(chunk),
            "explaintext": 1, "exintro": 1, "exlimit": _BATCH, "inprop": "url", "redirects": 1,
        })
        pages.extend(data.get("query", {}).get("pages", {}).values())
        time.sleep(_PAUSE_S)
    return pages


def build_kb(topics: dict, out_dir: Path, per_topic: int, per_query: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source": "wikipedia", "granularity": "intro", "per_topic": per_topic, "topics": {}}

    for topic_id, spec in topics.items():
        # Query seed: target + query canoniche (già mirate al dibattito)
        seeds = [spec["target"], *spec.get("queries", [])]
        titles: list[str] = []
        seen: set = set()
        for q in seeds:
            for t in search_titles(q, per_query):
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
            time.sleep(_PAUSE_S)
        # Copertura ampia: articoli delle categorie Wikipedia del tema (+ sottocategorie)
        for cat in _SEED_CATEGORIES.get(topic_id, []):
            if len(titles) >= per_topic:
                break
            for t in category_members(cat, depth=1, limit=per_topic):
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
                if len(titles) >= per_topic:
                    break
        titles = titles[:per_topic]
        logger.info(f"[{topic_id}] {len(titles)} titoli candidati (query + categorie)")

        docs = []
        for page in fetch_intros(titles):
            text = page.get("extract", "")
            if len(text.split()) < _MIN_WORDS:
                continue
            docs.append({
                "doc_id": f"wiki-{topic_id}-{len(docs):03d}",
                "topic": topic_id,
                "title": page.get("title", ""),
                "url": page.get("fullurl", ""),
                "text": text,
                "source": "wikipedia",
                "tier": 1,
            })

        path = out_dir / f"wikipedia_{topic_id}.jsonl"
        payload = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
        path.write_text(payload, encoding="utf-8")
        manifest["topics"][topic_id] = {
            "articles": len(docs),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "titles": [d["title"] for d in docs],
        }
        total_words = sum(len(d["text"].split()) for d in docs)
        logger.info(f"[{topic_id}] {len(docs)} articoli Wikipedia (~{total_words} parole) -> {path.name}")

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Manifest KB scritto in {out_dir / 'MANIFEST.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Costruisce la Knowledge Base Wikipedia mirata")
    parser.add_argument("--topics", default="configs/topics.yaml")
    parser.add_argument("--out-dir", default="data/kb")
    parser.add_argument("--per-topic", type=int, default=600, help="Max articoli per tema")
    parser.add_argument("--per-query", type=int, default=6, help="Risultati per query di ricerca")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    topics = load_config(args.topics)["topics"]
    build_kb(topics, Path(args.out_dir), args.per_topic, args.per_query)


if __name__ == "__main__":
    main()
