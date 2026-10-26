"""
Test Suite — evaluate_pivot (funzioni pure)
===========================================
Derivazione del gold social dai corpora e aggregazione della diversity.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGoldFromSocialCorpus:
    def setup_method(self):
        from scripts.evaluate_pivot import gold_from_social_corpus
        self.gold = gold_from_social_corpus

    def test_maps_stance_gold_to_label(self, tmp_path):
        p = tmp_path / "vaccines_social.jsonl"
        rows = [
            {"text": "provax tweet", "stance_gold": "FAVOR", "topic": "vaccines"},
            {"text": "antivax tweet", "stance_gold": "AGAINST", "topic": "vaccines"},
        ]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        gold = self.gold(p, target="vaccine safety")
        assert gold == [
            {"text": "provax tweet", "target": "vaccine safety", "label": "FAVOR"},
            {"text": "antivax tweet", "target": "vaccine safety", "label": "AGAINST"},
        ]


class TestMeanDiversity:
    def setup_method(self):
        from scripts.evaluate_pivot import mean_diversity_by_ecosystem
        self.mean = mean_diversity_by_ecosystem

    def test_averages_per_ecosystem(self):
        per_query = [
            {"news": 0.1, "social": 0.6},
            {"news": 0.3, "social": 0.8},
        ]
        out = self.mean(per_query)
        assert out == {"news": 0.2, "social": 0.7}

    def test_handles_missing_ecosystem(self):
        # il nucleare ha solo news in alcune query
        per_query = [{"news": 0.2}, {"news": 0.4, "social": 0.6}]
        out = self.mean(per_query)
        assert out["news"] == 0.3
        assert out["social"] == 0.6


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
