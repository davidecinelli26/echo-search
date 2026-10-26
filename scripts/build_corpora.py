"""
Echo-Search | Fase 1 — Costruzione dei corpora offline (CC-News)
================================================================
Costruisce un corpus congelato PER TEMA a partire da CC-News, in streaming
(non scarica i ~700k articoli): scorre il dataset, filtra per keyword del tema,
annota ogni documento con i rating del registro fonti e bilancia per dominio
così nessuna testata domina il corpus (requisito anti-echo-chamber: la diversità
di prospettiva deve venire dai dati, non da una singola fonte).

Output (in `data/corpora/`):
- `{topic}.jsonl`   — un documento per riga, congelato
- `MANIFEST.json`   — conteggi, distribuzione fonti/orientamento e SHA-256 di
                      ogni file, per la riproducibilità

SCELTE:
- STREAMING: `load_dataset(..., streaming=True)` evita di materializzare l'intero
  dataset; si itera finché ogni tema raggiunge la quota o si esaurisce lo scan.
- FILTRO KEYWORD: una regex compilata per tema (OR delle keyword, word-boundary)
  su titolo+testo. Semplice e trasparente; a un articolo multi-tema viene
  assegnato il tema con più corrispondenze.
- BILANCIAMENTO: cap per dominio (default 15% della quota) per non far dominare
  una testata. La distribuzione per orientamento finisce nel manifest per il
  controllo del bilanciamento (Fase 7).

Uso:
    uv run --extra data python scripts/build_corpora.py            # 2000/tema
    uv run --extra data python scripts/build_corpora.py --smoke    # 20/tema (test rapido)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

CC_NEWS_DATASET = "vblagoje/cc_news"
MIN_WORDS = 50       # scarta articoli troppo corti (spesso stub o didascalie)
MIN_HITS = 2         # occorrenze keyword minime: esclude le menzioni incidentali
MIN_DENSITY = 0.0015  # match/parole: esclude articoli lunghi solo di sfuggita sul tema

# Report di mercato / comunicati commerciali: rumore, non dibattito giornalistico
_COMMERCIAL_SPAM = re.compile(
    r"market research|industry (?:analysis|forecast|report)|market size|"
    r"market outlook|procurement research|contract manufacturing|forecast to 20\d\d|"
    r"\bCAGR\b|market (?:share|report)",
    re.IGNORECASE,
)

# Marker di articoli d'opinione/editoriali (URL o titolo) — per costruire un
# corpus "opinion-heavy" da confrontare con la cronaca (--opinion-only).
_OPINION_URL = re.compile(
    r"/opinion|/commentisfree|/editorial|/comment/|/blogs?/|/voices|/views|/op-ed|/perspectives?",
    re.IGNORECASE,
)
_OPINION_TITLE = re.compile(
    r"^\s*(?:opinion|editorial|commentary|op-ed|viewpoint|analysis)\b|\b(?:op-ed|editorial)\b",
    re.IGNORECASE,
)


def is_opinion(url: str, title: str) -> bool:
    """True se l'articolo ha marker d'opinione nell'URL o nel titolo."""
    return bool(_OPINION_URL.search(url or "")) or bool(_OPINION_TITLE.search(title or ""))


# ---------------------------------------------------------------------------
# Preparazione
# ---------------------------------------------------------------------------

def compile_topic_patterns(topics: dict) -> dict[str, re.Pattern]:
    """Una regex per tema: OR delle keyword con word-boundary, case-insensitive."""
    patterns = {}
    for topic_id, spec in topics.items():
        alternatives = "|".join(re.escape(kw) for kw in spec["keywords"])
        patterns[topic_id] = re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)
    return patterns


def normalize_domain(domain: Optional[str], url: Optional[str]) -> str:
    """Dominio normalizzato (senza 'www.') per il lookup nel registro fonti."""
    d = (domain or "").strip().lower()
    if not d and url:
        m = re.search(r"https?://([^/]+)", url)
        d = m.group(1).lower() if m else ""
    return d[4:] if d.startswith("www.") else d


def assign_topic(text: str, patterns: dict[str, re.Pattern]) -> tuple[Optional[str], int]:
    """Tema con più corrispondenze keyword e il numero di hit (None, 0 se nessuna)."""
    best_topic, best_hits = None, 0
    for topic_id, pattern in patterns.items():
        hits = len(pattern.findall(text))
        if hits > best_hits:
            best_topic, best_hits = topic_id, hits
    return best_topic, best_hits


# ---------------------------------------------------------------------------
# Costruzione
# ---------------------------------------------------------------------------

def build_corpora(
    topics: dict,
    sources: dict,
    limit_per_topic: int,
    out_dir: Path,
    max_scan: Optional[int],
    dataset_name: str,
    opinion_only: bool = False,
) -> None:
    patterns = compile_topic_patterns(topics)
    source_map = sources.get("sources", {})
    source_default = sources.get("default", {"orientation": "unknown", "factual": "unknown", "tier": 3})

    max_per_domain = max(5, int(0.15 * limit_per_topic))  # cap anti-dominanza
    logger.info(
        f"Target {limit_per_topic} doc/tema | cap {max_per_domain}/dominio | "
        f"temi: {list(topics)}"
    )

    collected: dict[str, list[dict]] = {t: [] for t in topics}
    domain_counts: dict[str, Counter] = {t: Counter() for t in topics}

    stream = load_dataset(dataset_name, split="train", streaming=True)

    scanned = 0
    for row in stream:
        scanned += 1
        if max_scan and scanned > max_scan:
            logger.warning(f"Raggiunto max_scan={max_scan}: mi fermo.")
            break
        if scanned % 20000 == 0:
            done = {t: len(collected[t]) for t in topics}
            logger.info(f"Scansionati {scanned} articoli | raccolti {done}")

        text = (row.get("text") or "").strip()
        title = (row.get("title") or "").strip()
        words = len(text.split())
        if words < MIN_WORDS:
            continue

        blob = f"{title} {text}"
        if _COMMERCIAL_SPAM.search(blob):
            continue
        if opinion_only and not is_opinion(row.get("url", ""), title):
            continue

        topic, hits = assign_topic(blob, patterns)
        # Pertinenza: abbastanza occorrenze E densità sufficiente — un articolo
        # lungo che cita il tema una sola volta è off-topic.
        if topic is None or hits < MIN_HITS or hits / words < MIN_DENSITY:
            continue
        if len(collected[topic]) >= limit_per_topic:
            continue

        domain = normalize_domain(row.get("domain"), row.get("url"))
        if domain_counts[topic][domain] >= max_per_domain:
            continue  # bilanciamento: questa fonte ha già saturato la sua quota

        rating = source_map.get(domain, source_default)
        collected[topic].append({
            "doc_id": f"{topic}-{len(collected[topic]):04d}",
            "topic": topic,
            "title": title,
            "text": text,
            "domain": domain,
            "url": row.get("url", ""),
            "date": row.get("date", ""),
            "orientation": rating.get("orientation", "unknown"),
            "factual": rating.get("factual", "unknown"),
            "tier": rating.get("tier", 3),
            "source_type": "news",
            "stance_gold": None,
        })
        domain_counts[topic][domain] += 1

        if all(len(collected[t]) >= limit_per_topic for t in topics):
            logger.info(f"Tutte le quote raggiunte dopo {scanned} articoli.")
            break

    _write_outputs(collected, out_dir, dataset_name, limit_per_topic, scanned)


def _write_outputs(
    collected: dict[str, list[dict]],
    out_dir: Path,
    dataset_name: str,
    limit_per_topic: int,
    scanned: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "dataset": dataset_name,
        "limit_per_topic": limit_per_topic,
        "articles_scanned": scanned,
        "topics": {},
    }

    for topic, docs in collected.items():
        path = out_dir / f"{topic}.jsonl"
        payload = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs)
        path.write_text(payload, encoding="utf-8")

        manifest["topics"][topic] = {
            "count": len(docs),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "by_orientation": dict(Counter(d["orientation"] for d in docs)),
            "by_tier": dict(Counter(d["tier"] for d in docs)),
            "top_domains": dict(Counter(d["domain"] for d in docs).most_common(10)),
        }
        logger.info(
            f"[{topic}] {len(docs)} doc → {path.name} | "
            f"orientamento: {manifest['topics'][topic]['by_orientation']}"
        )

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Manifest scritto in {out_dir / 'MANIFEST.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Costruisce i corpora offline per tema da CC-News")
    parser.add_argument("--topics", default="configs/topics.yaml")
    parser.add_argument("--sources", default="configs/sources.yaml")
    parser.add_argument("--out-dir", default="data/corpora")
    parser.add_argument("--dataset", default=CC_NEWS_DATASET)
    parser.add_argument("--limit", type=int, default=2000, help="Documenti per tema")
    parser.add_argument("--max-scan", type=int, default=None, help="Max articoli da scansionare")
    parser.add_argument("--smoke", action="store_true", help="Test rapido: 20 doc/tema, scan limitato")
    parser.add_argument("--opinion-only", action="store_true",
                        help="Tieni solo articoli d'opinione/editoriali (corpus opinion-heavy)")
    args = parser.parse_args()

    setup_logging(logging.INFO)

    if args.smoke:
        args.limit, args.max_scan = 20, 30000
        logger.info("Modalità SMOKE: 20 doc/tema, max_scan=30000")

    topics = load_config(args.topics)["topics"]
    sources = load_config(args.sources)

    build_corpora(
        topics=topics,
        sources=sources,
        limit_per_topic=args.limit,
        out_dir=Path(args.out_dir),
        max_scan=args.max_scan,
        dataset_name=args.dataset,
        opinion_only=args.opinion_only,
    )


if __name__ == "__main__":
    main()
