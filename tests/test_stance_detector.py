"""
Test Suite — StanceDetector (selezione backend FT/zero-shot)
============================================================
Verifica che il modulo scelga la modalità supervisionata quando model_name è una
directory locale esistente, e che il formato di output sia invariato.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBackendSelection:
    def test_is_finetuned_true_for_existing_dir(self, tmp_path):
        from src.nlp.stance_detector import StanceDetector
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        assert StanceDetector._is_finetuned(str(tmp_path)) is True

    def test_is_finetuned_false_for_hub_id(self):
        from src.nlp.stance_detector import StanceDetector
        assert StanceDetector._is_finetuned("cross-encoder/nli-roberta-base") is False


class _FakeTok:
    """Tokenizer fittizio: restituisce tensori reali con .to() no-op."""
    def __call__(self, targets, texts, **k):
        import torch
        class _Enc(dict):
            def to(self, *a, **kw):
                return self
        n = len(targets)
        return _Enc(
            input_ids=torch.zeros((n, 3), dtype=torch.long),
            attention_mask=torch.ones((n, 3), dtype=torch.long),
        )


class _FakeModel:
    """Modello fittizio: logits fissi per due documenti (AGAINST, FAVOR)."""
    def __call__(self, **enc):
        import torch
        class _Out:
            logits = torch.tensor([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1]])
        return _Out()


class TestSupervisedOutputFormat:
    def test_classify_batch_format(self, monkeypatch, tmp_path):
        """Con backend FT mockato, classify_batch restituisce il formato canonico."""
        from src.nlp import stance_detector as sd
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")

        # Config vuoto: l'argomento model_name esplicito deve vincere sul config.yaml.
        monkeypatch.setattr(sd, "load_section", lambda *a, **k: {})
        monkeypatch.setattr(sd, "_load_ft_model", lambda d: (_FakeModel(), _FakeTok()))

        det = sd.StanceDetector(model_name=str(tmp_path))
        assert det.finetuned is True
        out = det.classify_batch(["doc one", "doc two"], "nuclear energy")
        assert out[0]["label"] == "AGAINST"  # id 0, logit massimo
        assert out[1]["label"] == "FAVOR"    # id 1, logit massimo
        assert set(out[0]["scores"].keys()) == {"FAVOR", "AGAINST", "NEUTRAL"}
        assert 0.0 <= out[0]["confidence"] <= 1.0


class TestZeroshotModelResolution:
    def test_hub_id_used_directly(self):
        from src.nlp.stance_detector import _resolve_zeroshot_model
        assert _resolve_zeroshot_model("cross-encoder/nli-roberta-base") == "cross-encoder/nli-roberta-base"

    def test_local_ft_path_falls_back(self):
        from src.nlp.stance_detector import _resolve_zeroshot_model, _ZEROSHOT_FALLBACK
        assert _resolve_zeroshot_model("data/models/stance-deberta-v3-base") == _ZEROSHOT_FALLBACK

    def test_none_falls_back(self):
        from src.nlp.stance_detector import _resolve_zeroshot_model, _ZEROSHOT_FALLBACK
        assert _resolve_zeroshot_model(None) == _ZEROSHOT_FALLBACK


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
