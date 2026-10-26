"""
Test Suite — TextPreprocessor
==============================
Testa i comportamenti critici del preprocessore:
1. Che le negazioni vengano PRESERVATE (mai rimosse)
2. Che venga usata la LEMMATIZZAZIONE (non lo stemming)
3. Che il preprocessing sia SIMMETRICO (stessa pipeline per doc e query)
4. Che le contrazioni negative vengano espanse correttamente

Esecuzione: pytest tests/test_preprocessor.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNegationHandling:
    """Test CRITICO: le negazioni NON devono mai essere rimosse."""

    def setup_method(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        self.preprocessor = TextPreprocessor()

    def test_explicit_not_preserved(self):
        """'not' esplicito deve sopravvivere al preprocessing."""
        tokens = self.preprocessor.process("Nuclear power is not safe.")
        assert "not" in tokens, f"'not' rimosso! Tokens: {tokens}"

    def test_contraction_isnt_expanded(self):
        """"isn't" deve essere espanso e preservato come 'not'."""
        tokens = self.preprocessor.process("Nuclear power isn't safe.")
        assert "not" in tokens, f"Negazione in 'isn't' persa! Tokens: {tokens}"

    def test_contraction_doesnt_expanded(self):
        """"doesn't" deve essere espanso a 'not'."""
        tokens = self.preprocessor.process("This doesn't solve climate change.")
        assert "not" in tokens, f"Negazione in 'doesn't' persa! Tokens: {tokens}"

    def test_contraction_wont_expanded(self):
        """"won't" deve essere espanso a 'not'."""
        tokens = self.preprocessor.process("Renewables won't solve the energy crisis alone.")
        assert "not" in tokens, f"Negazione in 'won't' persa! Tokens: {tokens}"

    def test_never_preserved(self):
        """'never' deve essere normalizzato a 'not'."""
        tokens = self.preprocessor.process("Nuclear accidents never happen in modern plants.")
        assert "not" in tokens, f"'never' perso! Tokens: {tokens}"

    def test_stance_critical_sentence(self):
        """Una frase con stance negativa deve contenere il segnale di negazione."""
        positive = self.preprocessor.process("Nuclear power is safe.")
        negative = self.preprocessor.process("Nuclear power is not safe.")
        assert "not" not in positive, "Falso positivo: 'not' trovato nella frase positiva"
        assert "not" in negative, "Negazione persa nella frase negativa"


class TestLemmatization:
    """Test: la lemmatizzazione deve produrre forme canoniche del dizionario."""

    def setup_method(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        self.preprocessor = TextPreprocessor()

    def test_plural_to_singular(self):
        """"plants" → "plant" (non "plant" da stemmer che produrrebbe "plant" in questo caso,
        ma "studies" → "study" (lo stemmer produrrebbe "studi"))."""
        tokens = self.preprocessor.process("Nuclear studies show positive results.")
        assert "study" in tokens, f"Lemma 'study' non trovato. Tokens: {tokens}"
        assert "studi" not in tokens, f"Stem 'studi' trovato: lemmatizer non usato! Tokens: {tokens}"

    def test_verb_to_infinitive(self):
        """"running" → "run" (lemma), non "run" già (corretto sia per stemmer che lemmatizer)."""
        tokens = self.preprocessor.process("Climate change is accelerating rapidly.")
        assert "accelerate" in tokens, f"Lemma 'accelerate' non trovato. Tokens: {tokens}"

    def test_technical_terms_preserved(self):
        """Termini tecnici del dominio non devono essere troncati erroneamente."""
        tokens = self.preprocessor.process("Nuclear power plants emit no CO2.")
        assert "nuclear" in tokens
        assert "plant" in tokens  # lemma di "plants"
        assert "emit" in tokens   # lemma di "emit"


class TestPreprocessingSymmetry:
    """Test: la stessa pipeline su documento e query produce risultati comparabili."""

    def setup_method(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        self.preprocessor = TextPreprocessor()

    def test_shared_tokens_between_doc_and_query(self):
        """Un documento e una query sul medesimo argomento devono condividere token."""
        doc_tokens = set(self.preprocessor.process(
            "Nuclear power plants produce no carbon dioxide emissions."
        ))
        query_tokens = set(self.preprocessor.process(
            "nuclear energy carbon emissions"
        ))
        overlap = doc_tokens & query_tokens
        assert len(overlap) >= 2, (
            f"Overlap troppo basso ({overlap}) tra documento e query — "
            "possibile asimmetria di preprocessing"
        )

    def test_negation_preserved_in_query(self):
        """Le negazioni nelle query utente devono essere preservate."""
        query_tokens = self.preprocessor.process("nuclear is not safe")
        assert "not" in query_tokens, f"Negazione in query persa! Tokens: {query_tokens}"


class TestEdgeCases:
    """Test per casi limite."""

    def setup_method(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        self.preprocessor = TextPreprocessor()

    def test_empty_string(self):
        """Stringa vuota → lista vuota, senza eccezioni."""
        tokens = self.preprocessor.process("")
        assert tokens == []

    def test_very_short_tokens_removed(self):
        """Token con meno di 2 caratteri devono essere rimossi."""
        tokens = self.preprocessor.process("a b c nuclear power")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens

    def test_pure_numbers_removed(self):
        """Token puramente numerici (es. "2024") devono essere rimossi."""
        tokens = self.preprocessor.process("In 2024, nuclear generated 10% of power.")
        assert "2024" not in tokens
        assert "10" not in tokens

    def test_batch_equals_individual(self):
        """process_batch deve produrre lo stesso risultato di process() applicato uno per uno."""
        texts = [
            "Nuclear power is not safe.",
            "Climate change is accelerating.",
        ]
        individual = [self.preprocessor.process(t) for t in texts]
        batch = self.preprocessor.process_batch(texts)
        assert individual == batch, "process_batch e process() producono risultati diversi!"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
