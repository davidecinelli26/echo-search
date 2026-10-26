"""
Test Suite — stance_data
========================
Verifica il mapping label canonico, il caricamento jsonl e lo split stratificato
(proporzioni di classe preservate, determinismo del seed, nessun overlap).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLabelMapping:
    def test_mapping_is_canonical(self):
        from src.nlp.stance_data import LABEL2ID, ID2LABEL
        assert LABEL2ID == {"AGAINST": 0, "FAVOR": 1, "NEUTRAL": 2}
        assert ID2LABEL == {0: "AGAINST", 1: "FAVOR", 2: "NEUTRAL"}


class TestLoadJsonl:
    def setup_method(self):
        from src.nlp.stance_data import load_stance_jsonl
        self.load = load_stance_jsonl

    def test_loads_rows(self, tmp_path):
        p = tmp_path / "mini.jsonl"
        rows = [
            {"text": "a", "target": "t1", "label": "FAVOR", "source": "vast"},
            {"text": "b", "target": "t2", "label": "AGAINST", "source": "semeval"},
        ]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        loaded = self.load(p)
        assert loaded == rows


class TestStratifiedSplit:
    def setup_method(self):
        from src.nlp.stance_data import stratified_split
        self.split = stratified_split

    def _rows(self):
        # 100 righe: 50 FAVOR, 30 AGAINST, 20 NEUTRAL
        rows = []
        for i in range(50):
            rows.append({"text": f"f{i}", "target": "t", "label": "FAVOR", "source": "vast"})
        for i in range(30):
            rows.append({"text": f"a{i}", "target": "t", "label": "AGAINST", "source": "vast"})
        for i in range(20):
            rows.append({"text": f"n{i}", "target": "t", "label": "NEUTRAL", "source": "vast"})
        return rows

    def test_val_fraction_and_no_overlap(self):
        train, val = self.split(self._rows(), val_frac=0.1, seed=42)
        assert len(val) == 10
        assert len(train) == 90
        train_texts = {r["text"] for r in train}
        val_texts = {r["text"] for r in val}
        assert train_texts.isdisjoint(val_texts)

    def test_stratification_preserves_proportions(self):
        _, val = self.split(self._rows(), val_frac=0.1, seed=42)
        counts = {"FAVOR": 0, "AGAINST": 0, "NEUTRAL": 0}
        for r in val:
            counts[r["label"]] += 1
        # 10% di 50/30/20 → 5/3/2
        assert counts == {"FAVOR": 5, "AGAINST": 3, "NEUTRAL": 2}

    def test_seed_is_deterministic(self):
        a = self.split(self._rows(), val_frac=0.1, seed=42)[1]
        b = self.split(self._rows(), val_frac=0.1, seed=42)[1]
        assert [r["text"] for r in a] == [r["text"] for r in b]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
