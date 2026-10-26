"""
Test Suite — build_social_corpus (funzioni pure)
================================================
Parsing/mappatura label, dedup, campionamento stratificato deterministico,
costruzione documenti con schema allineato ai corpora news.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

CLIMATE_MAP = {"1": "FAVOR", "-1": "AGAINST", "0": "NEUTRAL"}


class TestParseLabeledRows:
    def setup_method(self):
        from scripts.build_social_corpus import parse_labeled_rows
        self.parse = parse_labeled_rows

    def test_maps_and_filters(self):
        csv_text = (
            "sentiment,message,tweetid\n"
            "1,Pro climate tweet,1\n"
            "-1,Denier tweet,2\n"
            "0,Neutral tweet,3\n"
            "2,News headline tweet,4\n"          # label '2' non nel map -> scartata
        )
        rows = self.parse(csv_text, "message", "sentiment", CLIMATE_MAP)
        assert [r["label"] for r in rows] == ["FAVOR", "AGAINST", "NEUTRAL"]
        assert rows[0]["text"] == "Pro climate tweet"

    def test_dedup_by_text(self):
        csv_text = (
            "sentiment,message,tweetid\n"
            "1,Same tweet,1\n"
            "1,same TWEET ,2\n"                   # duplicato (normalizzato) -> rimosso
            "0,Other,3\n"
        )
        rows = self.parse(csv_text, "message", "sentiment", CLIMATE_MAP)
        assert len(rows) == 2

    def test_skips_empty_text(self):
        csv_text = "sentiment,message,tweetid\n1,,1\n1,Real,2\n"
        rows = self.parse(csv_text, "message", "sentiment", CLIMATE_MAP)
        assert len(rows) == 1


class TestStratifiedSample:
    def setup_method(self):
        from scripts.build_social_corpus import stratified_sample
        self.sample = stratified_sample

    def _rows(self):
        r = []
        for i in range(100): r.append({"text": f"f{i}", "label": "FAVOR"})
        for i in range(50):  r.append({"text": f"a{i}", "label": "AGAINST"})
        for i in range(10):  r.append({"text": f"n{i}", "label": "NEUTRAL"})
        return r

    def test_caps_per_class(self):
        out = self.sample(self._rows(), n_per_class=30, seed=42)
        from collections import Counter
        c = Counter(r["label"] for r in out)
        assert c == {"FAVOR": 30, "AGAINST": 30, "NEUTRAL": 10}  # NEUTRAL ha solo 10

    def test_deterministic(self):
        a = self.sample(self._rows(), 30, 42)
        b = self.sample(self._rows(), 30, 42)
        assert [r["text"] for r in a] == [r["text"] for r in b]


class TestBuildSocialDocs:
    def setup_method(self):
        from scripts.build_social_corpus import build_social_docs
        self.build = build_social_docs

    def test_schema_and_fields(self):
        rows = [{"text": "a tweet", "label": "AGAINST"}]
        docs = self.build(rows, "vaccines", "vaccine safety")
        d = docs[0]
        assert d["doc_id"] == "vaccines_social-0000"
        assert d["source_type"] == "social"
        assert d["stance_gold"] == "AGAINST"
        assert d["tier"] == 3 and d["topic"] == "vaccines"
        # schema allineato ai news
        assert set(d) == {"doc_id", "topic", "title", "text", "domain", "url",
                          "date", "orientation", "factual", "tier",
                          "source_type", "stance_gold"}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
