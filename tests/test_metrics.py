"""
Test Suite — src/evaluation/metrics
====================================
Copre le funzioni che producono i numeri riportati in relazione: metriche di
ranking, Cohen's kappa e Stance Diversity. Sono funzioni pure: nessun modello,
nessun file, esecuzione in millisecondi.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (  # noqa: E402
    average_precision,
    compute_cohen_kappa,
    compute_ndcg,
    compute_recall_at_k,
    compute_stance_diversity,
)


def _run(*doc_ids):
    return [{"doc_id": d} for d in doc_ids]


class TestNdcg:
    def test_perfect_ranking_is_one(self):
        qrels = {"a": 2, "b": 1}
        assert compute_ndcg(_run("a", "b"), qrels, k=10) == 1.0

    def test_inverted_ranking_is_penalised(self):
        qrels = {"a": 2, "b": 1}
        assert compute_ndcg(_run("b", "a"), qrels, k=10) < 1.0

    def test_discount_rewards_early_positions(self):
        qrels = {"a": 2, "b": 0, "c": 0}
        early = compute_ndcg(_run("a", "b", "c"), qrels, k=10)
        late = compute_ndcg(_run("b", "c", "a"), qrels, k=10)
        assert early > late

    def test_cutoff_is_respected(self):
        qrels = {"a": 2}
        assert compute_ndcg(_run("b", "c", "a"), qrels, k=2) == 0.0

    def test_no_relevant_documents_gives_zero(self):
        # senza rilevanti l'IDCG e' nullo: la query non e' valutabile
        assert compute_ndcg(_run("a"), {"a": 0}, k=10) == 0.0

    def test_unjudged_documents_count_as_non_relevant(self):
        # convenzione di Cranfield: fuori dal pool = non rilevante
        assert compute_ndcg(_run("ignoto", "a"), {"a": 2}, k=10) < 1.0


class TestAveragePrecision:
    def test_all_relevant_first_is_one(self):
        assert average_precision(_run("a", "b"), {"a": 1, "b": 1}) == 1.0

    def test_precision_drops_with_a_miss_before_the_hit(self):
        assert average_precision(_run("x", "a"), {"a": 1}) == pytest.approx(0.5, abs=1e-4)

    def test_empty_run_is_zero(self):
        assert average_precision([], {"a": 1}) == 0.0

    def test_normalises_by_total_relevant_not_by_retrieved(self):
        # un rilevante trovato su due esistenti: 0,5, non 1,0
        assert average_precision(_run("a"), {"a": 1, "b": 1}) == pytest.approx(0.5, abs=1e-4)


class TestRecall:
    def test_counts_relevant_within_cutoff(self):
        assert compute_recall_at_k(_run("a", "b", "c"), {"a": 1, "d": 1}, k=3) == 0.5

    def test_cutoff_excludes_later_hits(self):
        assert compute_recall_at_k(_run("x", "a"), {"a": 1}, k=1) == 0.0

    def test_grade_zero_is_not_relevant(self):
        assert compute_recall_at_k(_run("a"), {"a": 0, "b": 1}, k=10) == 0.0


class TestCohenKappa:
    def test_perfect_agreement(self):
        res = compute_cohen_kappa(["0", "1", "2", "0"], ["0", "1", "2", "0"], labels=["0", "1", "2"])
        assert res["kappa"] == 1.0
        assert res["observed_agreement"] == 1.0

    def test_chance_level_agreement_is_near_zero(self):
        # marginali identici e accordo pari al caso -> kappa ~ 0
        y1 = ["0", "0", "1", "1"]
        y2 = ["0", "1", "0", "1"]
        assert abs(compute_cohen_kappa(y1, y2, labels=["0", "1"])["kappa"]) < 1e-6

    def test_systematic_disagreement_is_negative(self):
        res = compute_cohen_kappa(["0", "1"], ["1", "0"], labels=["0", "1"])
        assert res["kappa"] < 0

    def test_quadratic_weights_forgive_adjacent_disagreement(self):
        # su scala ordinale "0 vs 1" deve pesare meno di "0 vs 2"
        y1 = ["0", "0", "1", "2", "2", "1"]
        adjacent = ["1", "0", "1", "2", "2", "1"]
        far = ["2", "0", "1", "2", "2", "1"]
        k_adj = compute_cohen_kappa(y1, adjacent, labels=["0", "1", "2"], weights="quadratic")["kappa"]
        k_far = compute_cohen_kappa(y1, far, labels=["0", "1", "2"], weights="quadratic")["kappa"]
        assert k_adj > k_far

    def test_weighted_kappa_exceeds_unweighted_on_ordinal_drift(self):
        # disaccordo tutto di un gradino: il pesato deve essere piu' alto
        y1 = ["0", "0", "1", "1", "2", "2"]
        y2 = ["1", "0", "2", "1", "2", "1"]
        plain = compute_cohen_kappa(y1, y2, labels=["0", "1", "2"])["kappa"]
        weighted = compute_cohen_kappa(y1, y2, labels=["0", "1", "2"], weights="quadratic")["kappa"]
        assert weighted > plain

    def test_confusion_matrix_orientation(self):
        # righe = primo annotatore, colonne = secondo
        res = compute_cohen_kappa(["0", "0"], ["0", "1"], labels=["0", "1"])
        assert res["confusion"] == [[1, 1], [0, 0]]

    def test_single_label_used_by_both_is_undefined(self):
        res = compute_cohen_kappa(["1", "1"], ["1", "1"], labels=["1"])
        assert math.isnan(res["kappa"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            compute_cohen_kappa(["0"], ["0", "1"])

    def test_undeclared_label_raises(self):
        with pytest.raises(ValueError):
            compute_cohen_kappa(["0", "9"], ["0", "0"], labels=["0", "1"])

    def test_unknown_weighting_raises(self):
        with pytest.raises(ValueError):
            compute_cohen_kappa(["0"], ["0"], weights="cubic")


class TestStanceDiversity:
    def _clusters(self, favor, neutral, against):
        return {"FAVOR": [None] * favor, "NEUTRAL": [None] * neutral, "AGAINST": [None] * against}

    def test_uniform_distribution_is_maximal(self):
        assert compute_stance_diversity(self._clusters(10, 10, 10))["diversity_score"] == 1.0

    def test_degenerate_distribution_is_zero(self):
        assert compute_stance_diversity(self._clusters(30, 0, 0))["diversity_score"] == 0.0

    def test_matches_documented_examples(self):
        # Valori verificati a mano: per p=(0.8,0.1,0.1) l'entropia e' 0.6390 nats
        # e 0.6390/ln(3) = 0.5817. La prima stesura della relazione riportava 0.42
        # per questo caso: era un errore del documento, non del codice, ed e' questo
        # test ad averlo scoperto. Non allentare le tolleranze per farlo passare.
        assert compute_stance_diversity(self._clusters(8, 1, 1))["diversity_score"] == pytest.approx(0.5817, abs=1e-3)
        assert compute_stance_diversity(self._clusters(4, 3, 3))["diversity_score"] == pytest.approx(0.9912, abs=1e-3)
        assert compute_stance_diversity(self._clusters(28, 1, 1))["diversity_score"] == pytest.approx(0.2650, abs=1e-3)
        assert compute_stance_diversity(self._clusters(15, 10, 5))["diversity_score"] == pytest.approx(0.9206, abs=1e-3)

    def test_insensitive_to_sample_size(self):
        small = compute_stance_diversity(self._clusters(8, 1, 1))["diversity_score"]
        large = compute_stance_diversity(self._clusters(80, 10, 10))["diversity_score"]
        assert small == large

    def test_invariant_to_class_permutation(self):
        a = compute_stance_diversity(self._clusters(8, 1, 1))["diversity_score"]
        b = compute_stance_diversity(self._clusters(1, 8, 1))["diversity_score"]
        assert a == b

    def test_empty_input_does_not_crash(self):
        assert compute_stance_diversity(self._clusters(0, 0, 0))["diversity_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
