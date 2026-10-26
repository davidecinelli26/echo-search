"""
Test Suite — index_corpus: enumerazione (tema × ecosistema)
===========================================================
Verifica che iter_corpus_specs trovi news e social presenti, salti i mancanti
(nucleare = solo news) e produca out_name = {tema}/{ecosistema}.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIterCorpusSpecs:
    def setup_method(self):
        from scripts.index_corpus import iter_corpus_specs
        self.iter = iter_corpus_specs

    def _make_corpora(self, tmp_path):
        # news per tutti e 3; social solo per climate e vaccines
        for t in ("nuclear_energy", "climate_change", "vaccines"):
            (tmp_path / f"{t}.jsonl").write_text('{"text":"x","doc_id":"a"}\n', encoding="utf-8")
        for t in ("climate_change", "vaccines"):
            (tmp_path / f"{t}_social.jsonl").write_text('{"text":"y","doc_id":"b"}\n', encoding="utf-8")

    def test_enumerates_present_ecosystems(self, tmp_path):
        self._make_corpora(tmp_path)
        topics = {"nuclear_energy": {}, "climate_change": {}, "vaccines": {}}
        specs = self.iter(topics, tmp_path)
        pairs = {(s["topic"], s["ecosystem"]) for s in specs}
        assert ("nuclear_energy", "news") in pairs
        assert ("nuclear_energy", "social") not in pairs   # niente social sul nucleare
        assert ("climate_change", "social") in pairs
        assert len(specs) == 5                              # 3 news + 2 social

    def test_out_name_layout(self, tmp_path):
        self._make_corpora(tmp_path)
        specs = self.iter({"vaccines": {}}, tmp_path)
        by_eco = {s["ecosystem"]: s for s in specs}
        assert by_eco["news"]["out_name"] == "vaccines/news"
        assert by_eco["social"]["out_name"] == "vaccines/social"
        assert by_eco["social"]["corpus_path"].name == "vaccines_social.jsonl"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
