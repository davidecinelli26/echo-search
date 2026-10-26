"""
Echo-Search | Precalcolo della demo
===================================
Per ogni tema e per ogni query canonica (`topics.yaml`) esegue la pipeline
completa e salva il risultato in `data/demo_cache/`, cosi' che la dashboard lo
rilegga in millisecondi: nessun modello caricato, GPU a zero.

Si precalcola una versione COMPLETA — fact-check piu' riassunti — per CIASCUN
motore di verdetto. Il motore fa parte della chiave di cache, quindi nella demo
il selettore NLI/LLM funziona davvero e mostra due verdetti diversi sugli stessi
documenti: e' il confronto controllato del progetto, reso interattivo. La
dashboard nasconde poi cio' che i toggle non chiedono, quindi una sola versione
per motore copre tutte le combinazioni.

Ha senso perche' corpus e query sono fissi e i modelli deterministici (beam
search puro, temperatura 0): il risultato non cambia da un'esecuzione all'altra.

Uso (serve la GPU; il motore LLM richiede Ollama in esecuzione):
    uv run --no-sync --with "torch==2.2.2" --index cu121 python scripts/precompute_demo.py

Offline il tempo non pesa, quindi di default si verificano TUTTI i documenti
mostrati; `--per-column N` limita ai primi N di ogni colonna.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.app import EchoSearch, _free_gpu
from src.demo_cache import save_cached
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="Precalcola i risultati della demo")
    parser.add_argument("--topics", default="configs/topics.yaml")
    parser.add_argument("--engines", default="nli,llm",
                        help="motori di verdetto da precalcolare, separati da virgola")
    parser.add_argument("--per-column", type=int, default=0,
                        help="doc verificati per colonna; 0 = tutti")
    args = parser.parse_args()
    setup_logging(logging.INFO)

    topics = load_config(args.topics)["topics"]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    per_column = args.per_column or None       # 0 -> None -> tutti i documenti

    t0 = time.time()
    n = 0
    for topic, spec in topics.items():
        if not (Path("data/processed") / topic).exists():
            logger.info("[%s] nessun indice -> salto", topic)
            continue
        logger.info("=== %s ===", topic)
        app = EchoSearch(topic=topic)
        app.load()
        for engine in engines:
            app.fact_checker(engine).per_column = per_column
        target = spec["target"]
        for query in spec.get("queries", []):
            for engine in engines:
                # sempre la versione completa: la dashboard nasconde cio' che non serve
                r = app.search(query, target=target, fact_check=True,
                               summarize=True, fact_check_engine=engine)
                p = save_cached(topic, r, query, True, True, engine)
                n += 1
                verdetti = sum(1 for b in r["ecosystems"].values()
                               for docs in b["columns"].values()
                               for d in docs if d.get("fact_checks"))
                logger.info("[%s] %-42s eng=%-3s %2d verdetti -> %s (%.0fs)",
                            topic, query[:42], engine, verdetti, p.name, time.time() - t0)
        del app
        _free_gpu()

    logger.info("FATTO: %d file di cache in %.1f min", n, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
