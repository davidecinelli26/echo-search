"""
Test Suite — evaluate_by_source
===============================
Verifica l'aggregazione del Macro-F1 complessivo e per source (VAST/SemEval),
a partire da label vere (nelle rows) e predette.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEvaluateBySource:
    def setup_method(self):
        from scripts.eval_stance import evaluate_by_source
        self.eval = evaluate_by_source

    def test_overall_and_per_source_keys(self):
        rows = [
            {"text": "a", "target": "t", "label": "FAVOR", "source": "vast"},
            {"text": "b", "target": "t", "label": "AGAINST", "source": "vast"},
            {"text": "c", "target": "t", "label": "NEUTRAL", "source": "semeval"},
        ]
        y_pred = ["FAVOR", "AGAINST", "NEUTRAL"]
        report = self.eval(rows, y_pred)
        assert report["overall"]["macro_f1"] == 1.0
        assert set(report["by_source"].keys()) == {"vast", "semeval"}
        # Nel subset VAST le predizioni sono corrette; il Macro-F1 media su 3 classi e
        # NEUTRAL è assente nel subset (support 0 → f1 0). Verifichiamo le classi presenti.
        assert report["by_source"]["vast"]["per_class"]["FAVOR"]["f1"] == 1.0
        assert report["by_source"]["vast"]["per_class"]["AGAINST"]["f1"] == 1.0

    def test_length_mismatch_raises(self):
        rows = [{"text": "a", "target": "t", "label": "FAVOR", "source": "vast"}]
        import pytest
        with pytest.raises(ValueError):
            self.eval(rows, ["FAVOR", "AGAINST"])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
