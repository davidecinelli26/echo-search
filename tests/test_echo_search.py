"""
Test Suite — EchoSearch multi-ecosistema
========================================
Scoperta degli ecosistemi indicizzati da disco (news/social; nucleare solo news).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDiscoverEcosystems:
    def _mk(self, base: Path, topic: str, ecos: list[str]):
        for eco in ecos:
            d = base / topic / eco
            d.mkdir(parents=True, exist_ok=True)
            (d / "bm25_index.pkl").write_bytes(b"x")

    def test_news_and_social(self, tmp_path):
        from src.app import EchoSearch
        self._mk(tmp_path, "vaccines", ["news", "social"])
        es = EchoSearch(index_dir=tmp_path, topic="vaccines")
        assert es.discover_ecosystems() == ["news", "social"]

    def test_news_only(self, tmp_path):
        from src.app import EchoSearch
        self._mk(tmp_path, "nuclear_energy", ["news"])
        es = EchoSearch(index_dir=tmp_path, topic="nuclear_energy")
        assert es.discover_ecosystems() == ["news"]

    def test_none_when_missing(self, tmp_path):
        from src.app import EchoSearch
        es = EchoSearch(index_dir=tmp_path, topic="absent")
        assert es.discover_ecosystems() == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
