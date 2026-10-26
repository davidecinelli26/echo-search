"""
Echo-Search | Sintesi multi-prospettiva (FLAN-T5)
=================================================
Un riassunto per ciascuna delle tre colonne di stance, cosi' che il lettore possa
farsi un'idea di ogni posizione senza aprire sette articoli.

ESTRARRE, POI RIASSUMERE. La finestra di contesto di FLAN-T5 e' di circa 512
token, mentre una colonna di sette articoli ne vale qualche migliaio: non entra.
Una prima fase estrattiva ordina le frasi per pertinenza alla query con BM25 e
ne tiene le piu' salienti, poi il modello riscrive. Oltre a far entrare l'input,
l'estrattiva rimuove le ripetizioni — utile, perche' un corpus di cronaca
contiene molte riprese della stessa notizia d'agenzia.

GENERAZIONE DETERMINISTICA. Beam search puro, campionamento disattivato: la
stessa colonna deve produrre sempre lo stesso riassunto. Un sistema che si vuole
valutare non puo' dare due risposte diverse alla stessa domanda, e la demo
precalcolata non sarebbe nemmeno possibile. La temperatura non compare perche'
avrebbe effetto solo col campionamento attivo: indicarla sarebbe fuorviante.

SOLO DAL TESTO FORNITO. Il prompt vincola il modello alle frasi estratte, mai
alla conoscenza acquisita in addestramento: ogni affermazione del riassunto
risale a un documento del pool.

LIMITE DICHIARATO: la qualita' dei riassunti non e' valutata. Mancano riassunti
di riferimento di dominio e senza quelli ROUGE e BERTScore non sono calcolabili.
E' una baseline ragionevole, non un componente validato.
"""

import logging
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import pipeline

from src.utils.config import load_section
from src.utils.text import split_into_sentences

logger = logging.getLogger(__name__)


class MultiPerspectiveSummarizer:
    """
    Genera riassunti multi-prospettici separati per ogni stance cluster.

    Metodo principale: summarize_clusters()
    Input:  output di StanceDetector.cluster()
    Output: dict con riassunto per FAVOR, AGAINST, NEUTRAL

    Parametri
    ----------
    model_name : str
        Modello generativo. Default: "google/flan-t5-base".
    extractive_top_sentences : int
        Numero di frasi estratte per cluster. Default 10.
    num_beams : int
        Beam size per il beam search. Default 4.
    max_new_tokens : int
        Lunghezza massima del riassunto generato. Default 200.
    config_path : str, opzionale
        Path a configs/config.yaml.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        extractive_top_sentences: int = 10,
        num_beams: int = 4,
        max_new_tokens: int = 200,
        config_path: Optional[str] = None,
    ):
        summ_cfg = load_section(config_path, "summarization")
        model_name = summ_cfg.get("model_name", model_name)
        extractive_top_sentences = summ_cfg.get("extractive_top_sentences", extractive_top_sentences)
        self.extractive_top_sentences = extractive_top_sentences
        self.num_beams = summ_cfg.get("num_beams", num_beams)
        self.max_new_tokens = summ_cfg.get("max_new_tokens", max_new_tokens)
        self.do_sample = summ_cfg.get("do_sample", False)
        self.no_repeat_ngram_size = summ_cfg.get("no_repeat_ngram_size", 3)

        logger.info(f"Caricamento FLAN-T5: {model_name}")
        self.generator = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
        )

        logger.info(
            f"Summarizer pronto | model={model_name} | "
            f"beams={self.num_beams} | max_tokens={self.max_new_tokens} | "
            f"extractive_top_n={extractive_top_sentences}"
        )

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def summarize_clusters(
        self,
        clusters: dict[str, list[dict]],
        query: str,
        target: str,
    ) -> dict[str, dict]:
        """
        Genera un riassunto per ogni stance cluster.
        Entry point principale della pipeline.

        Parametri
        ----------
        clusters : dict
            Output di StanceDetector.cluster().
        query : str
            Query originale dell'utente (usata per il filtro estrattivo).
        target : str
            Topic del dibattito (es. "nuclear energy").

        Returns
        -------
        dict
            {
                "FAVOR":   {"summary": str, "sources": list[str], "num_docs": int},
                "AGAINST": {...},
                "NEUTRAL": {...}
            }
        """
        results = {}
        for stance_label, docs in clusters.items():
            if not docs:
                results[stance_label] = {
                    "summary": f"No {stance_label.lower()} documents found for this query.",
                    "sources": [],
                    "num_docs": 0,
                }
                continue

            logger.info(f"Summarizing {stance_label} cluster ({len(docs)} docs)...")
            summary_data = self._summarize_cluster(
                docs=docs,
                query=query,
                target=target,
                stance_label=stance_label,
            )
            results[stance_label] = summary_data

        return results

    # ------------------------------------------------------------------
    # Pipeline Extractive-then-Abstractive
    # ------------------------------------------------------------------

    def _summarize_cluster(
        self,
        docs: list[dict],
        query: str,
        target: str,
        stance_label: str,
    ) -> dict:
        """
        Pipeline completa per una colonna: frasi -> selezione -> riscrittura.

        Le frasi restano appaiate al documento di provenienza per tutto il
        percorso, cosi' il riassunto puo' dichiarare le proprie fonti.
        """
        # segmentazione, tenendo traccia del documento da cui viene ogni frase
        # (lo splitter e' lo stesso usato per le evidenze — src/utils/text.py)
        all_sentences: list[tuple[str, str]] = []  # (sentence, doc_id)
        for doc in docs:
            sentences = split_into_sentences(doc["text"])
            doc_id = doc.get("doc_id", "?")
            for sent in sentences:
                all_sentences.append((sent, str(doc_id)))

        if not all_sentences:
            return {"summary": "No content available.", "sources": [], "num_docs": 0}

        # le frasi si ordinano per pertinenza usando la query dell'utente come query BM25
        sentences_text = [s[0] for s in all_sentences]
        top_sentences_with_ids = self._extract_salient_sentences(
            sentences=sentences_text,
            doc_ids=[s[1] for s in all_sentences],
            query=query,
        )

        prompt = self._build_prompt(self._build_input(top_sentences_with_ids),
                                    target, stance_label)
        generated = self.generator(
            prompt,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            early_stopping=True,
        )
        final_summary = self._postprocess_summary(generated[0]["generated_text"])

        # i documenti da cui provengono le frasi scelte, senza ripetizioni
        sources = list(dict.fromkeys(doc_id for _, doc_id in top_sentences_with_ids))

        return {
            "summary": final_summary,
            "sources": sources,
            "num_docs": len(docs),
        }

    def _extract_salient_sentences(
        self,
        sentences: list[str],
        doc_ids: list[str],
        query: str,
    ) -> list[tuple[str, str]]:
        """
        Seleziona le top-N frasi più salienti rispetto alla query tramite BM25.
        Include deduplicazione per rimuovere frasi quasi-identiche.
        """
        # Filtra frasi troppo corte (< 5 parole) — spesso artefatti di segmentazione
        valid = [
            (sent, did) for sent, did in zip(sentences, doc_ids)
            if len(sent.split()) >= 5
        ]

        if not valid:
            return []

        valid_sents, valid_ids = zip(*valid)

        # BM25 scoring delle frasi rispetto alla query
        tokenized_sents = [s.lower().split() for s in valid_sents]
        tokenized_query = query.lower().split()

        bm25 = BM25Okapi(tokenized_sents)
        scores = np.asarray(bm25.get_scores(tokenized_query))

        # Ranking vettoriale per score BM25 (stable: riproducibile a parità di score)
        ranked_order = np.argsort(-scores, kind="stable")

        # Selezione Top-N con deduplicazione (similarity > 0.8 → skippa)
        selected: list[tuple[str, str]] = []
        seen_tokens: list[set] = []

        for i in ranked_order.tolist():
            sent, did = valid_sents[i], valid_ids[i]
            sent_tokens = set(sent.lower().split())
            # Controllo duplicazione: jaccard similarity con frasi già selezionate
            is_duplicate = any(
                len(sent_tokens & seen) / max(len(sent_tokens | seen), 1) > 0.8
                for seen in seen_tokens
            )
            if not is_duplicate:
                selected.append((sent, did))
                seen_tokens.append(sent_tokens)

            if len(selected) >= self.extractive_top_sentences:
                break

        logger.debug(
            f"Extractive: {len(sentences)} frasi → {len(selected)} selezionate (top-{self.extractive_top_sentences})"
        )
        return selected

    @staticmethod
    def _build_input(sentences_with_ids: list[tuple[str, str]]) -> str:
        """Concatena le frasi selezionate. Niente marker di provenienza nel testo:
        FLAN-T5-base li ricopiava nell'output invece di usarli, e le fonti si
        riportano comunque a parte."""
        return " ".join(sent for sent, _ in sentences_with_ids)

    @staticmethod
    def _build_prompt(testo: str, target: str, stance_label: str) -> str:
        """Prompt vincolato al solo testo fornito: e' l'argine alle allucinazioni."""
        stance_description = {
            "FAVOR": "in favor of",
            "AGAINST": "against",
            "NEUTRAL": "with a neutral stance on",
        }.get(stance_label, "about")

        return (
            f"Summarize in 2-3 sentences the following excerpts, all {stance_description} {target}. "
            f"Use ONLY the provided text. Text: {testo}"
        )

    @staticmethod
    def _postprocess_summary(summary: str) -> str:
        """Ripulisce l'output da eventuali marker residui e spazi doppi."""
        summary = re.sub(r"\[Doc [^\]]*\]", "", summary)
        summary = re.sub(r"\s{2,}", " ", summary).strip()
        return summary or "Sintesi non disponibile."
