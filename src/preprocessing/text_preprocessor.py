"""
Echo-Search | Tier 1 — Text Preprocessor
=========================================
Responsabilità: normalizzazione linguistica dei documenti prima dell'indicizzazione.

SCELTE ARCHITETTURALI (documentate per la tesi):
-------------------------------------------------
1. LEMMATIZZAZIONE (non Stemming):
   Il progetto adotta spaCy WordNetLemmatizer invece di PorterStemmer.
   Motivazione: lo Stemmer produce forme troncate spesso illeggibili ("nuclear" →
   "nuclear", ma "studies" → "studi", "running" → "run"). La Lemmatizzazione produce
   la forma canonica del dizionario, mantenendo la leggibilità per il Fact-Checking
   (dove le evidenze testuali vengono mostrate all'utente) e migliorando la qualità
   dell'Inverted Index su termini tecnici del dominio energetico/climatico.

2. GESTIONE DELLE NEGAZIONI:
   Le negazioni ("not", "no", "never", "n't") NON vengono rimosse.
   Motivazione: sono segnali critici per la Stance Detection. "Nuclear power is NOT safe"
   e "Nuclear power is safe" hanno stance opposte ma token quasi identici senza negazioni.
   La pipeline utilizza una whitelist di negatori che vengono preservati anche se
   tecnicamente stopwords in NLTK.

3. PREPROCESSING SIMMETRICO:
   La STESSA pipeline viene applicata a documenti e query.
   Asimmetrie di preprocessing sono una fonte comune di vocabulary mismatch artificiale.
"""

import re
import logging
from typing import Optional

import spacy
import nltk
from nltk.corpus import stopwords

# Download risorse NLTK necessarie
nltk.download("stopwords", quiet=True)

logger = logging.getLogger(__name__)


# Negatori da NON rimuovere MAI — critici per Stance Detection
_NEGATION_TOKENS = frozenset({
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "hardly", "scarcely", "barely", "without",
    # Forme contratte (post-tokenizzazione spaCy)
    "n't", "nt",
})


class TextPreprocessor:
    """
    Preprocessor per l'Inverted Index BM25 e per la pipeline NLP.

    Parametri
    ----------
    language : str
        Lingua del corpus. Default "english".
        Per italiano: usare "italian" + caricare "it_core_news_sm".
    remove_stopwords : bool
        Se True, rimuove le stopwords NLTK (ma NON i negatori).
        Default True per l'indicizzazione BM25; impostare False per i modelli Transformer
        che gestiscono le stopwords internamente.
    spacy_model : str
        Modello spaCy per la lemmatizzazione. Default "en_core_web_sm".
    """

    def __init__(
        self,
        language: str = "english",
        remove_stopwords: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        self.language = language
        self.remove_stopwords = remove_stopwords

        # Caricamento stopwords NLTK — i negatori vengono esclusi dalla lista
        raw_stopwords = set(stopwords.words(language))
        self.stop_words = raw_stopwords - _NEGATION_TOKENS  # preserva negatori

        # Caricamento modello spaCy per lemmatizzazione
        # disable=["parser", "ner"] → +30% velocità: non ci servono parse tree né entità
        try:
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
        except OSError:
            logger.error(
                f"Modello spaCy '{spacy_model}' non trovato. "
                f"Eseguire: python -m spacy download {spacy_model}"
            )
            raise

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def process(self, text: str) -> list[str]:
        """
        Pipeline completa di preprocessing. Entry point principale.

        Fasi:
        1. Lowercasing
        2. Normalizzazione Unicode (apostrofi, trattini)
        3. Rimozione punteggiatura (con preservazione negazioni contratte)
        4. Lemmatizzazione via spaCy
        5. Stopwords removal (con whitelist negatori)
        6. Filtraggio token troppo corti o numerici puri

        Parametri
        ----------
        text : str
            Testo grezzo del documento o della query.

        Returns
        -------
        list[str]
            Lista di lemmi normalizzati, pronti per l'Inverted Index.
        """
        if not text or not text.strip():
            return []

        # Step 1: Lowercasing
        text = text.lower()

        # Step 2: Normalizzazione — espandi contrazioni negatives prima di pulire
        # (es. "isn't" → "is not", "can't" → "can not")
        text = self._expand_negation_contractions(text)

        # Step 3: Rimozione punteggiatura (mantiene spazi e alfanumerici)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Step 4 + 5 + 6: Lemmatizzazione, stopwords, filtraggio — in un unico pass spaCy
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            lemma = token.lemma_.lower().strip()

            # Preserva negatori esplicitamente
            if token.text in _NEGATION_TOKENS or lemma in _NEGATION_TOKENS:
                tokens.append("not")  # normalizziamo tutti i negatori a "not"
                continue

            # Stopwords removal
            if self.remove_stopwords and lemma in self.stop_words:
                continue

            # Filtraggio: scarta token < 2 caratteri e puramente numerici
            if len(lemma) < 2 or lemma.isdigit():
                continue

            tokens.append(lemma)

        return tokens

    def process_batch(self, texts: list[str], batch_size: int = 64) -> list[list[str]]:
        """
        Preprocessing batch-ottimizzato via spaCy pipe().
        Significativamente più veloce di chiamare process() in loop su corpora grandi.
        """
        results = []
        for doc in self.nlp.pipe(
            (self._prepare_text(t) for t in texts),
            batch_size=batch_size,
        ):
            tokens = self._extract_tokens_from_doc(doc)
            results.append(tokens)
        return results

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _prepare_text(self, text: str) -> str:
        """Lowercasing + espansione contrazioni + pulizia — senza lemmatizzazione."""
        text = text.lower()
        text = self._expand_negation_contractions(text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_tokens_from_doc(self, doc) -> list[str]:
        """Estrae lemmi da un doc spaCy già processato."""
        tokens = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            if token.text in _NEGATION_TOKENS or lemma in _NEGATION_TOKENS:
                tokens.append("not")
                continue
            if self.remove_stopwords and lemma in self.stop_words:
                continue
            if len(lemma) < 2 or lemma.isdigit():
                continue
            tokens.append(lemma)
        return tokens

    @staticmethod
    def _expand_negation_contractions(text: str) -> str:
        """
        Espande le forme contratte delle negazioni inglesi.
        Questo garantisce che "isn't" venga trattato come "is not"
        prima della rimozione della punteggiatura — evitando la perdita
        del segnale negativo critico per la Stance Detection.
        """
        contractions = {
            r"\bisn't\b": "is not",
            r"\baren't\b": "are not",
            r"\bwasn't\b": "was not",
            r"\bweren't\b": "were not",
            r"\bcan't\b": "can not",
            r"\bcannot\b": "can not",
            r"\bcouldn't\b": "could not",
            r"\bwon't\b": "will not",
            r"\bwouldn't\b": "would not",
            r"\bdon't\b": "do not",
            r"\bdoesn't\b": "does not",
            r"\bdidn't\b": "did not",
            r"\bhasn't\b": "has not",
            r"\bhaven't\b": "have not",
            r"\bhadn't\b": "had not",
            r"\bshouldn't\b": "should not",
            r"\bmustn't\b": "must not",
            r"\bneedn't\b": "need not",
            r"\bdaren't\b": "dare not",
            r"\busn't\b": "used not",
        }
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text)
        return text
