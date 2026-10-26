"""
Echo-Search | Test — utility condivise
======================================
`top_k_indices` e `split_into_sentences` sono usate da quasi tutti i moduli
(retrieval sparso e denso, fusione, evidenze, sintesi). Un errore qui non
fallirebbe in modo visibile: cambierebbe silenziosamente un ranking. Per questo
sono coperte a parte e non insieme al modulo che le usa.

Esecuzione: uv run pytest tests/test_utils.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ranking import top_k_indices  # noqa: E402
from src.utils.text import split_into_sentences  # noqa: E402


class TestTopKIndices:
    def test_ordinamento_decrescente(self):
        assert top_k_indices(np.array([0.1, 0.9, 0.5, 0.7]), 2).tolist() == [1, 3]

    def test_k_maggiore_di_n(self):
        assert top_k_indices(np.array([0.3, 0.1, 0.2]), 10).tolist() == [0, 2, 1]

    def test_k_zero_e_array_vuoto(self):
        assert top_k_indices(np.array([1.0]), 0).size == 0
        assert top_k_indices(np.array([]), 5).size == 0

    def test_stabile_a_parita_di_score(self):
        # a parita' di punteggio vince l'indice piu' basso: senza questo il
        # ranking non sarebbe riproducibile fra esecuzioni
        assert top_k_indices(np.array([0.5, 0.5, 0.5, 0.9]), 4).tolist() == [3, 0, 1, 2]

    def test_coerente_con_argsort_completo(self):
        rng = np.random.default_rng(42)
        scores = rng.random(500)
        np.testing.assert_array_equal(top_k_indices(scores, 20), np.argsort(-scores)[:20])


class TestSplitIntoSentences:
    def test_segmentazione_base(self):
        frasi = split_into_sentences(
            "Nuclear power is clean. It produces no CO2. Is it safe? Experts disagree!")
        assert len(frasi) == 4
        assert frasi[0] == "Nuclear power is clean."

    def test_non_spezza_sui_decimali(self):
        assert len(split_into_sentences("The plant produces 3.5 gigawatts of power.")) == 1

    def test_testo_vuoto(self):
        assert split_into_sentences("") == []
