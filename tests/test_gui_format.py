"""Test — funzioni pure di formatting della dashboard."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestColumnWidths:
    def test_no_focus_equal(self):
        from src.gui.format import column_widths
        assert column_widths(None) == [1, 1, 1]

    def test_focus_widens_one_others_visible(self):
        from src.gui.format import column_widths
        assert column_widths("FAVOR") == [6, 2, 2]
        assert column_widths("NEUTRAL") == [2, 6, 2]
        assert column_widths("AGAINST") == [2, 2, 6]


class TestVerdictBadge:
    def test_refuted_wins(self):
        from src.gui.format import verdict_badge
        b = verdict_badge({"SUPPORTED": 2, "REFUTED": 1, "NOT_ENOUGH_EVIDENCE": 0})
        assert b["label"] == "REFUTED" and b["tone"] == "danger"

    def test_supported_when_no_refuted(self):
        from src.gui.format import verdict_badge
        b = verdict_badge({"SUPPORTED": 3, "REFUTED": 0, "NOT_ENOUGH_EVIDENCE": 1})
        assert b["label"] == "SUPPORTED" and b["tone"] == "success"

    def test_none_when_no_claims(self):
        from src.gui.format import verdict_badge
        assert verdict_badge({"SUPPORTED": 0, "REFUTED": 0, "NOT_ENOUGH_EVIDENCE": 0}) is None
        assert verdict_badge({}) is None


class TestTruncate:
    def test_short_unchanged(self):
        from src.gui.format import truncate
        assert truncate("ciao", 10) == "ciao"

    def test_long_ellipsis(self):
        from src.gui.format import truncate
        assert truncate("abcdefghij", 5) == "abcde…"


class TestDiversityTone:
    def test_bands(self):
        from src.gui.format import diversity_tone
        assert diversity_tone(0.70) == "success"
        assert diversity_tone(0.45) == "warning"
        assert diversity_tone(0.20) == "danger"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
