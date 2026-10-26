"""
Echo-Search | Orchestrazione della pipeline e interfaccia a riga di comando
===========================================================================
`EchoSearch` e' l'unico punto in cui i moduli si incontrano: retrieval, stance,
bilanciamento, fact-check e sintesi. Tutto il resto del codice non sa che gli
altri componenti esistono, e questo e' cio' che permette di valutarli uno per
uno negli script di `scripts/`.

DUE ECOSISTEMI, UNA SOLA PIPELINE. Dato un tema, la classe scopre quali corpora
sono stati indicizzati (news, social, o entrambi) ed esegue su ciascuno la
MEDESIMA catena, con gli stessi modelli e gli stessi parametri. La simmetria non
e' eleganza: e' la condizione che rende confrontabili i due punteggi di Stance
Diversity. Se un ecosistema ricevesse un trattamento diverso, la differenza
misurata potrebbe venire da li'.

MODELLI PESANTI SPENTI DI DEFAULT. Fact-check e sintesi sono proprieta' lazy e
disattivati salvo richiesta esplicita: una ricerca senza di essi costa
centinaia di millisecondi, con l'LLM costa minuti. Fra un ecosistema e l'altro
si libera la cache CUDA, perche' il progetto deve stare in 6 GB di VRAM.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.nlp.fact_checker import ENGINES, FactChecker
from src.nlp.stance_detector import StanceDetector
from src.nlp.summarizer import MultiPerspectiveSummarizer
from src.retrieval.column_balancer import balance_columns
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/config.yaml"
DEFAULT_INDEX_DIR = Path("data/processed")
_ECOSYSTEMS = ("news", "social")


def _free_gpu() -> None:
    """Libera la VRAM cachata: su 6 GB e' cio' che permette due ecosistemi di seguito."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class EchoSearch:
    """
    Ricerca multi-prospettiva su un tema, per ogni ecosistema indicizzato.

    Uso tipico::

        app = EchoSearch(topic="nuclear_energy")
        esito = app.search("is nuclear power safe", target="nuclear energy")
        esito["ecosystems"]["news"]["diversity"]["diversity_score"]
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG,
                 index_dir: str | Path = DEFAULT_INDEX_DIR,
                 topic: str = "nuclear_energy"):
        self.config_path = config_path
        self.topic = topic
        self.topic_dir = Path(index_dir) / topic
        self.retrievers: dict[str, HybridRetriever] = {}
        self._reranker: Optional[CrossEncoderReranker] = None
        self._stance_detector: Optional[StanceDetector] = None
        self._summarizer: Optional[MultiPerspectiveSummarizer] = None
        self._fact_checkers: dict[str, FactChecker] = {}

    def discover_ecosystems(self) -> list[str]:
        """Ecosistemi effettivamente indicizzati per il tema (il nucleare ha solo news)."""
        return [eco for eco in _ECOSYSTEMS
                if (self.topic_dir / eco / "bm25_index.pkl").exists()]

    def load(self, skip_dense: bool = False) -> None:
        """Carica un retriever per ogni ecosistema che risulta indicizzato per il tema."""
        ecosystems = self.discover_ecosystems()
        if not ecosystems:
            raise FileNotFoundError(
                f"Nessun indice per '{self.topic}' in {self.topic_dir}. "
                "Eseguire: uv run python scripts/index_corpus.py"
            )
        for eco in ecosystems:
            logger.info("Carico ecosistema '%s' da %s", eco, self.topic_dir / eco)
            self.retrievers[eco] = HybridRetriever.load_from_disk(
                str(self.topic_dir / eco), config_path=self.config_path, skip_dense=skip_dense)

    # --- componenti pesanti, costruiti solo se qualcuno li chiede ---

    @property
    def stance_detector(self) -> StanceDetector:
        if self._stance_detector is None:
            self._stance_detector = StanceDetector(config_path=self.config_path)
        return self._stance_detector

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(config_path=self.config_path)
        return self._reranker

    @property
    def summarizer(self) -> MultiPerspectiveSummarizer:
        if self._summarizer is None:
            # beam e token ridotti: FLAN-T5 su CPU e' lento, e la sintesi non e'
            # una metrica del progetto ma un aiuto alla lettura
            self._summarizer = MultiPerspectiveSummarizer(
                config_path=self.config_path, num_beams=2, max_new_tokens=120)
        return self._summarizer

    def fact_checker(self, engine: str = "nli") -> FactChecker:
        """Un fact-checker per motore, tenuto in cache: cambiarlo non ricarica l'altro."""
        if engine not in self._fact_checkers:
            # per_column=2: verificare l'intera colonna col motore LLM costerebbe minuti
            self._fact_checkers[engine] = FactChecker(
                self.topic, engine=engine, per_column=2, config_path=self.config_path)
        return self._fact_checkers[engine]

    # --- pipeline ---

    def _run_pipeline(self, retriever: HybridRetriever, query: str, target: str,
                      pool_size: int, fact_check: bool, summarize: bool,
                      fact_check_engine: str) -> dict:
        """Un ecosistema: recupero -> riordino -> colonne -> (verdetti) -> (sintesi)."""
        pool = retriever.search(query, top_k=pool_size)            # stadio 1: recupero
        reranked = self.reranker.rerank(query, pool)               # stadio 2: pertinenza
        balanced = balance_columns(reranked, self.stance_detector, target)  # stadio 3: colonne
        columns = balanced["columns"]
        if fact_check:
            columns = self.fact_checker(fact_check_engine).check_columns(columns)
        block = {
            "results": reranked,
            "columns": columns,
            "diversity": balanced["diversity"],
            "pool_distribution": balanced["pool_distribution"],
        }
        if summarize:
            block["summaries"] = self.summarizer.summarize_clusters(columns, query, target)
        return block

    def search(self, query: str, target: str, pool_size: int = 50,
               fact_check: bool = False, summarize: bool = False,
               fact_check_engine: str = "nli") -> dict:
        """
        Esegue la pipeline su ogni ecosistema del tema.

        Returns
        -------
        dict
            {query, target, topic, ecosystems: {eco: {results, columns,
             diversity, pool_distribution, [summaries]}}}
        """
        if not self.retrievers:
            self.load()
        ecosystems = {}
        for eco, retriever in self.retrievers.items():
            ecosystems[eco] = self._run_pipeline(retriever, query, target, pool_size,
                                                 fact_check, summarize, fact_check_engine)
            _free_gpu()   # il picco di VRAM e' fra un ecosistema e il successivo
        return {"query": query, "target": target, "topic": self.topic, "ecosystems": ecosystems}


# ---------------------------------------------------------------------------
# Interfaccia a riga di comando
# ---------------------------------------------------------------------------

def _format_results(response: dict) -> str:
    """Resa testuale dei due blocchi: diversity, colonne, primi documenti."""
    lines = ["", f"Query:  {response['query']}",
             f"Topic:  {response['topic']} | target: {response['target']}"]
    etichette = {"news": "MEDIA (news)", "social": "SOCIAL (tweet)"}
    for eco, block in response["ecosystems"].items():
        lines.append("")
        lines.append(f"=== {etichette.get(eco, eco)} ===")
        div = block["diversity"]
        lines.append(f"  Stance diversity: {div.get('diversity_score', 0):.3f} "
                     f"| pool {block['pool_distribution']}")
        summaries = block.get("summaries") or {}
        for label in ("FAVOR", "AGAINST", "NEUTRAL"):
            docs = block["columns"].get(label, [])
            lines.append(f"  {label} ({len(docs)})")
            riassunto = summaries.get(label, {}).get("summary")
            if riassunto:
                lines.append(f"    riassunto: {riassunto[:200]}")
            for doc in docs[:3]:
                anteprima = doc["text"][:80].replace("\n", " ")
                lines.append(f"    - conf={doc['stance_confidence']:.2f} | {anteprima}...")
                for fc in doc.get("fact_checks", []):
                    lines.append(f"        fact: {fc['verdict']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Echo-Search — ricerca multi-prospettiva")
    p.add_argument("query", nargs="?", help="query di ricerca (omettibile con --interactive)")
    p.add_argument("--topic", default="nuclear_energy",
                   help="tema da interrogare (indici in data/processed/{topic}/)")
    p.add_argument("--target", default="nuclear energy",
                   help="proposizione su cui si misura la stance dei documenti")
    p.add_argument("--fact-check", action="store_true",
                   help="verifica le affermazioni contro la KB Wikipedia [lento]")
    p.add_argument("--engine", choices=ENGINES, default="nli",
                   help="motore di verdetto del fact-check (default: nli)")
    p.add_argument("--summarize", action="store_true",
                   help="riassunto per prospettiva di stance [lento]")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="percorso del config YAML")
    p.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR),
                   help="directory degli indici BM25 e FAISS")
    p.add_argument("--skip-dense", action="store_true",
                   help="carica il solo indice BM25 (avvio rapido, niente FAISS)")
    p.add_argument("--interactive", "-i", action="store_true",
                   help="modalita' interattiva: query fino a 'quit'")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    setup_logging(logging.INFO)
    args = build_parser().parse_args(argv)
    app = EchoSearch(config_path=args.config, index_dir=args.index_dir, topic=args.topic)
    app.load(skip_dense=args.skip_dense)

    def esegui(q: str) -> None:
        print(_format_results(app.search(
            q, target=args.target, fact_check=args.fact_check,
            summarize=args.summarize, fact_check_engine=args.engine)))

    if args.interactive:
        print("Echo-Search — modalita' interattiva. 'quit' per uscire.")
        while True:
            try:
                q = input("\nQuery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"quit", "exit", "q"}:
                break
            esegui(q)
        return 0

    if not args.query:
        build_parser().print_help()
        return 1
    esegui(args.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
