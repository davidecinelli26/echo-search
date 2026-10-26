"""
Test Suite — sample_for_annotation
==================================
Verifica il campionamento deterministico dei documenti da annotare e la
selezione del sottoinsieme in overlap (per Cohen's kappa).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSampleForAnnotation:
    def setup_method(self):
        from scripts.build_gold_annotation import sample_for_annotation
        self.sample = sample_for_annotation

    def _docs(self, n=100):
        return [{"doc_id": f"d-{i:04d}", "title": f"t{i}", "text": f"body {i}"} for i in range(n)]

    def test_sample_size_and_overlap(self):
        sheet, overlap = self.sample(self._docs(), n_sample=30, overlap_frac=0.2, seed=42)
        assert len(sheet) == 30
        assert len(overlap) == 6  # 20% di 30
        overlap_ids = {d["doc_id"] for d in overlap}
        sheet_ids = {d["doc_id"] for d in sheet}
        assert overlap_ids.issubset(sheet_ids)

    def test_deterministic(self):
        a, _ = self.sample(self._docs(), n_sample=30, overlap_frac=0.2, seed=42)
        b, _ = self.sample(self._docs(), n_sample=30, overlap_frac=0.2, seed=42)
        assert [d["doc_id"] for d in a] == [d["doc_id"] for d in b]

    def test_caps_at_available(self):
        sheet, _ = self.sample(self._docs(10), n_sample=30, overlap_frac=0.2, seed=42)
        assert len(sheet) == 10  # non si inventano documenti


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
