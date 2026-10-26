"""
Echo-Search | Test — motore di verdetto LLM (funzioni PURE, senza rete)
======================================================================
Copre la robustezza del parsing del verdetto LLM: JSON valido, sinonimi,
output sporco non-JSON, vuoto. Nessuna chiamata a Ollama.

Esecuzione: uv run pytest tests/test_llm_verdict.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.llm_verdict import (  # noqa: E402
    _format_prompt,
    canon_verdict,
    parse_verdict,
)


class TestCanonVerdict:
    def test_canonical_passthrough(self):
        assert canon_verdict("SUPPORTED") == "SUPPORTED"
        assert canon_verdict("REFUTED") == "REFUTED"
        assert canon_verdict("NOT_ENOUGH_EVIDENCE") == "NOT_ENOUGH_EVIDENCE"

    def test_aliases_and_case(self):
        assert canon_verdict("supports") == "SUPPORTED"
        assert canon_verdict("True") == "SUPPORTED"
        assert canon_verdict("contradiction") == "REFUTED"
        assert canon_verdict("insufficient") == "NOT_ENOUGH_EVIDENCE"

    def test_spaces_vs_underscores(self):
        assert canon_verdict("NOT ENOUGH EVIDENCE") == "NOT_ENOUGH_EVIDENCE"
        assert canon_verdict("not-enough-evidence") == "NOT_ENOUGH_EVIDENCE"

    def test_unknown_defaults_to_nei(self):
        assert canon_verdict("banana") == "NOT_ENOUGH_EVIDENCE"
        assert canon_verdict("") == "NOT_ENOUGH_EVIDENCE"


class TestParseVerdict:
    def test_valid_json_all_fields(self):
        out = parse_verdict('{"verdict": "SUPPORTED", "reasoning": "Evidence [1] confirms it."}')
        assert out["verdict"] == "SUPPORTED"
        assert out["reasoning"] == "Evidence [1] confirms it."

    def test_json_with_alias_verdict(self):
        out = parse_verdict('{"verdict": "refutes", "reasoning": "x"}')
        assert out["verdict"] == "REFUTED"

    def test_non_json_but_contains_label(self):
        out = parse_verdict("The claim appears REFUTED by the passage.")
        assert out["verdict"] == "REFUTED"

    def test_empty_defaults_to_nei(self):
        assert parse_verdict("")["verdict"] == "NOT_ENOUGH_EVIDENCE"
        assert parse_verdict("   ")["verdict"] == "NOT_ENOUGH_EVIDENCE"

    def test_garbage_defaults_to_nei(self):
        assert parse_verdict("blah blah no structure")["verdict"] == "NOT_ENOUGH_EVIDENCE"

    def test_reasoning_truncated(self):
        long_reason = "x" * 500
        out = parse_verdict('{"verdict": "SUPPORTED", "reasoning": "' + long_reason + '"}')
        assert len(out["reasoning"]) <= 300


class TestFormatPrompt:
    def test_numbers_evidence(self):
        p = _format_prompt("The sky is blue.", ["Rayleigh scattering.", "Daytime observation."])
        assert "CLAIM: The sky is blue." in p
        assert "[1] Rayleigh scattering." in p
        assert "[2] Daytime observation." in p


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
