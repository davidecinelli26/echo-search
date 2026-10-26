"""
Echo-Search | Tier 3 — Stance Detector (Cross-Encoder RoBERTa)
==============================================================
Classifica ogni documento recuperato nella sua posizione argomentativa
rispetto al topic del dibattito: FAVOR, AGAINST, NEUTRAL.

SCELTE ARCHITETTURALI:
-----------------------
1. CROSS-ENCODER vs BI-ENCODER:
   Scegliamo un Cross-Encoder perché modella la relazione Target↔Documento
   tramite cross-attention completa (Reimers & Gurevych, 2019).
   Un Bi-Encoder encode query e documento SEPARATAMENTE e poi confronta
   i vettori: semanticamente meno preciso per task classificativi.
   Trade-off: il Cross-Encoder è O(N) su N documenti da classificare,
   non O(1) come il Bi-Encoder. Accettabile perché operiamo su Top-K (K≤20).

2. NO MIXTURE-OF-EXPERTS, NO DANN — DECISIONE DEFINITIVA:
   - MoE: introduce routing non-deterministico. Con diversi forward pass sulla
     stessa istanza si possono attivare esperti diversi, rendendo il Macro F1
     non riproducibile su un test set fisso. Incompatibile con la metodologia
     di valutazione accademica.
   - DANN (Domain-Adversarial Neural Network, Ganin et al. 2016): rilevante
     per la domain generalization (addestrare su Clima, testare su Energia
     Nucleare). Nel nostro setting, il dominio di train e test è lo stesso
     (argomenti energetici/climatici). DANN è documentato come ESTENSIONE
     FUTURA per la versione multi-dominio.
   - SCELTA DEFINITIVA: Fine-Tuning di RoBERTa-base su dominio ristretto.
     Motivazione: Macro F1 su classi sbilanciate (NEUTRAL tipicamente
     sottorappresentato) migliora con l'adattamento al dominio specifico
     rispetto a modelli NLI generalisti (Sun et al., 2019).

3. DATASET DI FINE-TUNING:
   - SemEval-2016 Task 6 (stance su Twitter, 5 topic)
   - VAST (Allaway & McKeown, 2020) — zero-shot stance detection multi-topic
   - Dati propri: subset annotato manualmente su Clima/Nucleare
"""

import logging
import os
from typing import Optional

from transformers import AutoTokenizer, pipeline

from src.nlp.stance_data import ID2LABEL
from src.utils.config import load_section
from src.utils.device import resolve_device

logger = logging.getLogger(__name__)


def _load_ft_model(model_dir: str):
    """Carica (model, tokenizer) del classificatore fine-tuned su device auto."""
    import torch
    from transformers import AutoModelForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use_fast=False: tokenizer FT salvato come SentencePiece (spm.model), niente protobuf.
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    return model, tok


# Modello NLI usato per lo zero-shot quando non c'è un modello fine-tuned locale.
_ZEROSHOT_FALLBACK = "cross-encoder/nli-roberta-base"


def _resolve_zeroshot_model(model_name: str) -> str:
    """
    Sceglie il modello per la pipeline zero-shot. Se `model_name` è un path locale
    (un modello FT atteso ma assente, es. non ancora addestrato) degrada al fallback
    NLI; se è un hub-id NLI lo usa direttamente.
    """
    looks_local = bool(model_name) and (
        os.path.isabs(model_name)
        or model_name.startswith(("data/", "data\\", "./", ".\\", "../", "..\\"))
        or os.sep in model_name
    )
    return _ZEROSHOT_FALLBACK if looks_local else (model_name or _ZEROSHOT_FALLBACK)


# Truncation manuale: RoBERTa max 512 token ≈ prime 400 parole del documento
_MAX_DOC_WORDS = 400


class StanceDetector:
    """
    Classificatore di stance argomentativa basato su Cross-Encoder.

    Input:  Coppia (target_topic, documento)
    Output: {"label": "FAVOR"|"AGAINST"|"NEUTRAL", "confidence": float}

    Formato input al modello:
    "[CLS] {target_topic} [SEP] {documento} [SEP]"
    — standard per i task NLI con Cross-Encoder Transformer.

    Parametri
    ----------
    model_name : str
        Path del modello FT locale (Fase 3) o hub-id NLI per il fallback zero-shot.
        Default: "data/models/stance-deberta-v3-base". Se il path FT non esiste,
        si degrada allo zero-shot NLI (_ZEROSHOT_FALLBACK). In produzione il valore
        arriva da configs/config.yaml (passare config_path).
    batch_size : int
        Dimensione del batch per l'inference (config: stance_detection.batch_size).
    config_path : str, opzionale
        Path a configs/config.yaml.
    device : str, opzionale
        "cuda" per GPU, "cpu" per CPU. Default: auto-detect.
    """

    def __init__(
        self,
        model_name: str = "data/models/stance-deberta-v3-base",
        batch_size: int = 16,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        stance_cfg = load_section(config_path, "stance_detection")
        self.model_name = stance_cfg.get("model_name", model_name)
        self.batch_size = stance_cfg.get("batch_size", batch_size)
        self.device = resolve_device(device)
        self.finetuned = self._is_finetuned(self.model_name)

        logger.info(
            f"Caricamento Stance Detector: {self.model_name} | "
            f"modalità={'fine-tuned' if self.finetuned else 'zero-shot'} | device={self.device}"
        )

        if self.finetuned:
            # Backend supervisionato a 3 classi (Fase 3): DeBERTa-v3 fine-tuned.
            self._model, self._tokenizer = _load_ft_model(self.model_name)
        else:
            # Fallback zero-shot NLI. Se model_name puntava a un modello FT locale
            # assente (es. non ancora addestrato), degrada a un NLI valido (spec §8).
            # Tokenizer slow (use_fast=False): il tokenizer.json "fast" di alcuni
            # checkpoint NLI è più recente del parser Rust di `tokenizers` pinnato
            # da transformers 4.40 e solleva "untagged enum ModelWrapper".
            zs_model = _resolve_zeroshot_model(self.model_name)
            if zs_model != self.model_name:
                logger.warning(
                    f"Modello FT non trovato in '{self.model_name}': "
                    f"fallback zero-shot su {zs_model}"
                )
            tokenizer = AutoTokenizer.from_pretrained(zs_model, use_fast=False)
            self.classifier = pipeline(
                "zero-shot-classification",
                model=zs_model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

        logger.info("StanceDetector pronto.")

    @staticmethod
    def _is_finetuned(model_name: str) -> bool:
        """True se model_name è una directory locale di modello (contiene config.json)."""
        return os.path.isdir(model_name) and os.path.exists(
            os.path.join(model_name, "config.json")
        )

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def classify(self, document: str, target: str) -> dict:
        """
        Classifica la stance di un singolo documento rispetto al target.

        Parametri
        ----------
        document : str
            Testo del documento da classificare.
        target : str
            Topic del dibattito (es. "nuclear energy", "climate change").

        Returns
        -------
        dict
            {
                "label": "FAVOR" | "AGAINST" | "NEUTRAL",
                "confidence": float,     # score della classe vincente
                "scores": {              # distribuzione completa
                    "FAVOR": float,
                    "AGAINST": float,
                    "NEUTRAL": float
                }
            }
        """
        return self.classify_batch([document], target)[0]

    def classify_batch(
        self,
        documents: list[str],
        target: str,
        batch_size: Optional[int] = None,
    ) -> list[dict]:
        """
        Classifica un batch di documenti con UNA sola chiamata alla pipeline HF.

        Il batching avviene dentro la pipeline (DataLoader interno): un forward
        pass per batch, non per documento. Evita il loop Python per-documento
        che vanificherebbe il parallelismo di GPU/BLAS.

        Parametri
        ----------
        documents : list[str]
            Lista di documenti da classificare.
        target : str
            Topic del dibattito.
        batch_size : int, opzionale
            Sovrascrive il batch_size di default (config: stance_detection.batch_size).

        Returns
        -------
        list[dict]
            Lista di risultati nell'ordine dei documenti in input.
        """
        if not documents:
            return []

        # Truncation manuale: prime _MAX_DOC_WORDS parole (≈ 512 token RoBERTa)
        truncated = [" ".join(doc.split()[:_MAX_DOC_WORDS]) for doc in documents]

        if self.finetuned:
            return self._classify_supervised(truncated, target)

        outputs = self.classifier(
            truncated,
            candidate_labels=[
                "in favor of " + target,
                "against " + target,
                "neutral about " + target,
            ],
            hypothesis_template="This text is {}.",
            batch_size=batch_size or self.batch_size,
        )
        # Con un solo documento la pipeline restituisce un dict, non una lista
        if isinstance(outputs, dict):
            outputs = [outputs]

        return [self._to_stance_result(out) for out in outputs]

    def _classify_supervised(self, documents: list[str], target: str) -> list[dict]:
        """Predizione a 3 classi col modello fine-tuned (coppie target/documento)."""
        import torch
        enc = self._tokenizer(
            [target] * len(documents), documents,
            truncation=True, max_length=512, padding=True, return_tensors="pt",
        ).to(self.device if self.device == "cuda" else "cpu")
        with torch.no_grad():
            probs = self._model(**enc).logits.softmax(dim=-1).cpu().numpy()
        results = []
        for row in probs:
            scores = {ID2LABEL[i]: round(float(row[i]), 4) for i in range(len(row))}
            best = max(scores, key=scores.get)
            results.append({"label": best, "confidence": scores[best], "scores": scores})
        return results

    @staticmethod
    def _to_stance_result(result: dict) -> dict:
        """
        Mappa l'output zero-shot HF ({labels, scores}) al formato del progetto.
        Corrispondenza NLI → stance: entailment della label "in favor of" = FAVOR,
        "against" = AGAINST, "neutral about" = NEUTRAL.
        """
        label_scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            if "in favor" in label:
                label_scores["FAVOR"] = score
            elif "against" in label:
                label_scores["AGAINST"] = score
            else:
                label_scores["NEUTRAL"] = score

        best_label = max(label_scores, key=label_scores.get)

        return {
            "label": best_label,
            "confidence": round(label_scores[best_label], 4),
            "scores": {k: round(v, 4) for k, v in label_scores.items()},
        }

    def cluster(
        self,
        retrieval_results: list[dict],
        target: str,
    ) -> dict[str, list[dict]]:
        """
        Classifica tutti i documenti recuperati e li raggruppa per stance.
        Questo è il metodo chiamato dalla pipeline principale (Tier 3).

        Parametri
        ----------
        retrieval_results : list[dict]
            Output di HybridRetriever.search() — lista di documenti con score.
        target : str
            Topic del dibattito per la classificazione stance.

        Returns
        -------
        dict
            {
                "FAVOR":   [{"text": ..., "hybrid_score": ..., "stance_confidence": ...}, ...],
                "AGAINST": [...],
                "NEUTRAL": [...]
            }
        """
        documents = [r["text"] for r in retrieval_results]

        logger.info(f"Avvio stance detection su {len(documents)} documenti | target='{target}'")
        stance_results = self.classify_batch(documents, target)

        clusters: dict[str, list[dict]] = {
            "FAVOR": [],
            "AGAINST": [],
            "NEUTRAL": [],
        }

        for retrieval_doc, stance in zip(retrieval_results, stance_results):
            enriched_doc = {
                **retrieval_doc,
                "stance_label": stance["label"],
                "stance_confidence": stance["confidence"],
                "stance_scores": stance["scores"],
            }
            clusters[stance["label"]].append(enriched_doc)

        # Log distribuzione stance (utile per calcolare la Stance Diversity Score)
        distribution = {k: len(v) for k, v in clusters.items()}
        logger.info(f"Distribuzione stance: {distribution}")

        # Ordina ogni cluster per hybrid_score decrescente
        for label in clusters:
            clusters[label].sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

        return clusters
