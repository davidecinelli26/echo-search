"""
Test Suite — src/evaluation/qrels + logica pura di eval_retrieval
=================================================================
Verifica il parsing dei fogli annotati (dove un errore silenzioso produrrebbe
metriche sbagliate senza segnalarlo) e la taratura di alpha in leave-one-topic-out.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.qrels import parse_qrels_rows, relevant_count  # noqa: E402


def _row(query, doc_id, relevance):
    return {"query": query, "doc_id": doc_id, "relevance": relevance}


class TestParseQrelsRows:
    def test_groups_by_query(self):
        qrels, skipped = parse_qrels_rows([
            _row("q1", "a", "2"), _row("q1", "b", "0"), _row("q2", "a", "1"),
        ])
        assert qrels == {"q1": {"a": 2, "b": 0}, "q2": {"a": 1}}
        assert skipped == 0

    def test_uncompiled_rows_are_skipped_not_zeroed(self):
        # una coppia non giudicata NON e' un giudizio di non-rilevanza
        qrels, skipped = parse_qrels_rows([_row("q1", "a", ""), _row("q1", "b", "2")])
        assert qrels == {"q1": {"b": 2}}
        assert skipped == 1
        assert "a" not in qrels["q1"]

    def test_whitespace_only_counts_as_uncompiled(self):
        _, skipped = parse_qrels_rows([_row("q1", "a", "   ")])
        assert skipped == 1

    def test_grades_are_parsed_as_int(self):
        qrels, _ = parse_qrels_rows([_row("q1", "a", "2")])
        assert isinstance(qrels["q1"]["a"], int)

    def test_invalid_grade_raises(self):
        with pytest.raises(ValueError, match="non valida"):
            parse_qrels_rows([_row("q1", "a", "3")])

    def test_contradictory_duplicate_raises(self):
        with pytest.raises(ValueError, match="contraddittorio"):
            parse_qrels_rows([_row("q1", "a", "2"), _row("q1", "a", "0")])

    def test_consistent_duplicate_is_tolerated(self):
        qrels, _ = parse_qrels_rows([_row("q1", "a", "2"), _row("q1", "a", "2")])
        assert qrels == {"q1": {"a": 2}}

    def test_empty_input(self):
        assert parse_qrels_rows([]) == ({}, 0)


class TestRelevantCount:
    def test_counts_only_positive_grades(self):
        assert relevant_count({"a": 2, "b": 1, "c": 0}) == 2

    def test_all_zero(self):
        assert relevant_count({"a": 0}) == 0


class TestTuneAlphaLoto:
    def setup_method(self):
        from scripts.eval_retrieval import tune_alpha_loto
        self.tune = tune_alpha_loto

    def test_selects_alpha_from_other_topics_only(self):
        # su t1 alpha=0.1 e' ottimo, su t2 lo e' 0.9: ogni fold deve scegliere
        # l'ottimo dell'ALTRO tema, mai il proprio
        ndcg = {
            "t1": {a: {"q": 1.0 if a == 0.1 else 0.0} for a in [round(0.1*i,1) for i in range(1,10)]},
            "t2": {a: {"q": 1.0 if a == 0.9 else 0.0} for a in [round(0.1*i,1) for i in range(1,10)]},
        }
        chosen = self.tune(ndcg)
        assert chosen["t1"] == 0.9
        assert chosen["t2"] == 0.1

    def test_deterministic_tie_break_picks_smallest_alpha(self):
        grid = [round(0.1*i,1) for i in range(1,10)]
        ndcg = {"t1": {a: {"q": 0.5} for a in grid}, "t2": {a: {"q": 0.5} for a in grid}}
        assert self.tune(ndcg)["t1"] == 0.1

    def test_single_topic_has_no_training_data(self):
        # con un solo tema non esistono query di addestramento: fallback deterministico
        grid = [round(0.1*i,1) for i in range(1,10)]
        chosen = self.tune({"t1": {a: {"q": 0.7} for a in grid}})
        assert chosen["t1"] == 0.1


class TestKendallTau:
    def setup_method(self):
        from scripts.eval_retrieval import kendall_tau
        self.tau = kendall_tau

    def test_identical_ordering_is_one(self):
        assert self.tau([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]) == 1.0

    def test_reversed_ordering_is_minus_one(self):
        assert self.tau([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0

    def test_partial_agreement_is_between(self):
        t = self.tau([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
        assert -1.0 < t < 1.0


class TestAlign:
    """L'appaiamento fra due annotatori: un disallineamento qui falserebbe il kappa."""

    def setup_method(self):
        from scripts.eval_agreement import align
        self.align = align

    def test_pairs_are_matched_by_query_and_doc(self):
        a = {"q1": {"d1": 2, "d2": 0}}
        b = {"q1": {"d2": 1, "d1": 2}}  # ordine di inserimento diverso
        y_a, y_b, _ = self.align(a, b)
        assert list(zip(y_a, y_b)) == [("2", "2"), ("0", "1")]  # d1, d2 in ordine

    def test_only_the_intersection_is_compared(self):
        # una coppia giudicata da uno solo e' un dato mancante, non un disaccordo
        a = {"q1": {"d1": 2, "d2": 0}}
        b = {"q1": {"d1": 2}}
        y_a, y_b, cov = self.align(a, b)
        assert (y_a, y_b) == (["2"], ["2"])
        assert cov["common"] == 1 and cov["only_a"] == 1 and cov["only_b"] == 0

    def test_same_doc_id_under_different_queries_is_not_confused(self):
        a = {"q1": {"d1": 2}, "q2": {"d1": 0}}
        b = {"q1": {"d1": 2}, "q2": {"d1": 0}}
        y_a, y_b, cov = self.align(a, b)
        assert cov["common"] == 2 and y_a == y_b == ["2", "0"]

    def test_deterministic_order(self):
        a = {"q2": {"d2": 1}, "q1": {"d1": 0}}
        b = {"q1": {"d1": 0}, "q2": {"d2": 1}}
        assert self.align(a, b)[0] == self.align(b, a)[0] == ["0", "1"]


class TestAgreementStats:
    def setup_method(self):
        from scripts.eval_agreement import agreement_stats
        self.stats = agreement_stats

    def test_perfect_agreement(self):
        res = self.stats(["0", "1", "2"], ["0", "1", "2"])
        assert res["observed_agreement"] == 1.0
        assert res["kappa"]["quadratic"] == 1.0
        assert res["two_step_conflicts"] == 0

    def test_two_step_conflicts_counted_only_at_distance_two(self):
        res = self.stats(["0", "0", "1"], ["2", "1", "2"])
        assert res["two_step_conflicts"] == 1

    def test_quadratic_kappa_is_kinder_to_adjacent_disagreement(self):
        y1 = ["0", "1", "2", "0", "1", "2"]
        adiacenti = ["1", "2", "1", "1", "0", "1"]
        res = self.stats(y1, adiacenti)
        assert res["kappa"]["quadratic"] > res["kappa"]["linear"] > res["kappa"]["unweighted"]

    def test_labels_are_fixed_even_if_a_grade_is_unused(self):
        # senza etichette dichiarate K passerebbe da 3 a 2 e i pesi cambierebbero
        res = self.stats(["0", "1"], ["0", "1"])
        assert res["labels"] == ["0", "1", "2"]
        assert len(res["confusion"]) == 3

    def test_marginals_expose_a_systematic_threshold_difference(self):
        res = self.stats(["1", "1", "1"], ["2", "2", "2"])
        assert res["marginals"]["a"] == {"0": 0, "1": 3, "2": 0}
        assert res["marginals"]["b"] == {"0": 0, "1": 0, "2": 3}


class TestInterpret:
    def setup_method(self):
        from scripts.eval_agreement import interpret
        self.interpret = interpret

    def test_landis_koch_bands(self):
        assert self.interpret(0.867) == "almost perfect"
        assert self.interpret(0.675) == "substantial"
        assert self.interpret(0.45) == "moderate"

    def test_nan_is_not_defined(self):
        assert self.interpret(float("nan")) == "non definito"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
