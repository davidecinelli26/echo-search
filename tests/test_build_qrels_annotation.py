"""
Test Suite — pooling e formattazione dei fogli qrels
====================================================
Verifica la logica pura di scripts/build_qrels_annotation: unione multi-run
(pooling di Cranfield), ordine di presentazione randomizzato ma riproducibile,
ed estratto troncato al confine di frase.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestUnionPool:
    def setup_method(self):
        from scripts.build_qrels_annotation import union_pool
        self.union_pool = union_pool

    def test_union_includes_docs_from_every_run(self):
        # il senso del pooling: un doc entra se ALMENO UN run lo propone
        pool = self.union_pool({
            "bm25": ["a", "b"],
            "dense": ["b", "c"],
            "fusion": ["a", "b"],
        })
        assert set(pool) == {"a", "b", "c"}

    def test_provenance_records_all_runs(self):
        pool = self.union_pool({"bm25": ["a"], "dense": ["a"], "rrf": ["b"]})
        assert pool["a"] == ["bm25", "dense"]
        assert pool["b"] == ["rrf"]

    def test_no_duplicate_run_names(self):
        # lo stesso run non deve comparire due volte se ripete il doc_id
        pool = self.union_pool({"fusion": ["a", "a"]})
        assert pool["a"] == ["fusion"]

    def test_exclusive_docs_are_identifiable(self):
        # e' il dato che dice quanto il pooling multi-run stia effettivamente aggiungendo
        pool = self.union_pool({"bm25": ["a", "b"], "dense": ["b"]})
        exclusive = [d for d, runs in pool.items() if len(runs) == 1]
        assert exclusive == ["a"]

    def test_empty_runs(self):
        assert self.union_pool({"bm25": [], "dense": []}) == {}


class TestShuffledIds:
    def setup_method(self):
        from scripts.build_qrels_annotation import shuffled_ids
        self.shuffled_ids = shuffled_ids

    def test_deterministic_for_same_seed(self):
        ids = [f"d-{i:03d}" for i in range(50)]
        assert self.shuffled_ids(ids, seed=42) == self.shuffled_ids(ids, seed=42)

    def test_preserves_the_set(self):
        ids = [f"d-{i:03d}" for i in range(50)]
        assert sorted(self.shuffled_ids(ids, seed=42)) == sorted(ids)

    def test_hides_original_order(self):
        # il rango del sistema non deve trasparire dall'ordine di presentazione
        ids = [f"d-{i:03d}" for i in range(50)]
        assert self.shuffled_ids(ids, seed=42) != ids

    def test_insensitive_to_input_order(self):
        # un dict di provenienza puo' arrivare in ordine qualsiasi: l'output no
        ids = [f"d-{i:03d}" for i in range(20)]
        assert self.shuffled_ids(ids, seed=7) == self.shuffled_ids(list(reversed(ids)), seed=7)


class TestExcerpt:
    def setup_method(self):
        from scripts.build_qrels_annotation import excerpt
        self.excerpt = excerpt

    def test_short_text_is_untouched(self):
        assert self.excerpt("Nuclear power is safe.", 1200) == "Nuclear power is safe."

    def test_flattens_newlines(self):
        # una cella CSV multi-riga e' illeggibile in Excel
        assert self.excerpt("Prima riga.\n\nSeconda riga.", 1200) == "Prima riga. Seconda riga."

    def test_truncates_on_sentence_boundary(self):
        text = " ".join(f"Sentence number {i} about reactors." for i in range(200))
        out = self.excerpt(text, 200)
        assert len(out) <= 200 + len(" [...]")
        assert out.endswith(" [...]")
        assert "reactors." in out          # non taglia a meta' frase
        assert not out.replace(" [...]", "").endswith("reactor")

    def test_very_long_first_sentence_falls_back_to_word_boundary(self):
        text = "word " * 500  # nessun confine di frase entro il limite
        out = self.excerpt(text, 100)
        assert out.endswith(" [...]")
        assert "wor " not in out           # mai a meta' parola

    def test_handles_empty_text(self):
        assert self.excerpt("", 1200) == ""
        assert self.excerpt(None, 1200) == ""


class TestBuildRows:
    def setup_method(self):
        from scripts.build_qrels_annotation import build_rows
        self.build_rows = build_rows

    def _meta(self):
        return {
            "a": {"title": "Titolo A", "text": "Testo A."},
            "b": {"title": "Titolo B", "text": "Testo B."},
        }

    def test_row_schema_and_empty_judgment(self):
        rows = self.build_rows({"a": ["bm25"]}, "q1", self._meta(), 1200)
        assert set(rows[0]) == {"query", "doc_id", "title", "text", "relevance", "nota"}
        assert rows[0]["relevance"] == ""      # da compilare a mano
        assert rows[0]["query"] == "q1"

    def test_rank_and_score_are_not_exposed(self):
        # position bias: l'annotatore non deve poter dedurre il rango del sistema
        rows = self.build_rows({"a": ["bm25"], "b": ["dense"]}, "q1", self._meta(), 1200)
        assert not any(k in rows[0] for k in ("rank", "score", "hybrid_score"))

    def test_missing_metadata_does_not_crash(self):
        rows = self.build_rows({"ignoto": ["bm25"]}, "q1", self._meta(), 1200)
        assert rows[0]["title"] == "" and rows[0]["text"] == ""


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
