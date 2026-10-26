"""Test — rendering HTML della dashboard (funzioni pure)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _block(with_fact=False, with_summary=False):
    doc = {"text": "Nuclear power is a low-carbon source.", "stance_confidence": 0.91, "doc_id": "a"}
    if with_fact:
        doc = {**doc,
               "factuality": {"SUPPORTED": 1},
               "fact_checks": [{"claim": "France uses nuclear power.", "verdict": "SUPPORTED",
                                "reasoning": "Confirmed by IEA.",
                                "sources": [{"title": "IEA", "url": "https://iea.org"}]}]}
    block = {"columns": {"FAVOR": [doc], "NEUTRAL": [], "AGAINST": []},
             "diversity": {"diversity_score": 0.41},
             "pool_distribution": {"FAVOR": 25, "AGAINST": 5, "NEUTRAL": 0}}
    if with_summary:
        block["summaries"] = {"FAVOR": {"summary": "Sintesi favorevole."}}
    return block


class TestRenderBlock:
    def test_has_three_stance_columns_and_score(self):
        from src.gui.render import render_block
        html = render_block(_block(), None, False)
        for label in ("FAVOR", "NEUTRAL", "AGAINST"):
            assert label in html
        assert "es-meter__score" in html and "0.41" in html

    def test_focus_sets_flex_grow(self):
        from src.gui.render import render_block
        html = render_block(_block(), "AGAINST", False)
        assert "flex-grow:6" in html

    def test_summary_rendered(self):
        from src.gui.render import render_block
        assert "Sintesi favorevole." in render_block(_block(with_summary=True), None, False)


class TestRenderCard:
    def test_preview_full_and_expandable(self):
        from src.gui.render import render_card
        html = render_card({"text": "x" * 300, "stance_confidence": 0.5, "doc_id": "a"}, False)
        assert "es-clamp" in html and "es-full" in html and "<details" in html

    def test_badge_and_sources_when_fact_on(self):
        from src.gui.render import render_card
        doc = _block(with_fact=True)["columns"]["FAVOR"][0]
        html = render_card(doc, True)
        assert "es-badge" in html
        assert "iea.org" in html          # fonte web mostrata
        assert "Confirmed by IEA." in html
        assert "es-badge" not in render_card(doc, False)

    def test_html_escaped(self):
        from src.gui.render import render_card
        html = render_card({"text": "a < b & c > d", "stance_confidence": 0.5, "doc_id": "a"}, False)
        assert "&lt;" in html and "&amp;" in html and "&gt;" in html


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
