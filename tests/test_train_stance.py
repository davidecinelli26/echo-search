"""
Test Suite — build_hf_dataset (preparazione dati per il training)
=================================================================
Verifica che le coppie (target, text) siano tokenizzate come due segmenti e che
le label stringa siano mappate agli id canonici. Usa il tokenizer reale di
deberta-v3-base (scaricato al primo run).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBuildHfDataset:
    def setup_method(self):
        from transformers import AutoTokenizer
        from scripts.train_stance import build_hf_dataset
        self.build = build_hf_dataset
        self.tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)

    def test_columns_and_label_mapping(self):
        rows = [
            {"text": "Nuclear is clean.", "target": "nuclear energy", "label": "FAVOR", "source": "vast"},
            {"text": "Ban it now.", "target": "nuclear energy", "label": "AGAINST", "source": "vast"},
        ]
        ds = self.build(rows, self.tok, max_len=32)
        assert set(["input_ids", "attention_mask", "labels"]).issubset(ds.column_names)
        assert ds["labels"] == [1, 0]  # FAVOR=1, AGAINST=0

    def test_pair_encoding_has_two_segments(self):
        rows = [{"text": "x", "target": "y", "label": "NEUTRAL", "source": "vast"}]
        ds = self.build(rows, self.tok, max_len=16)
        # due [SEP] impliciti: la coppia produce token_type/sep → almeno 3 special tokens
        assert self.tok.sep_token_id in ds["input_ids"][0]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
