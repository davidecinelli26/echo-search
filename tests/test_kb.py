"""Test — loader del KB Wikipedia per il fact-check KB-grounded."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoadKb:
    def test_loads_and_projects_fields(self, tmp_path):
        from src.nlp.kb import load_kb
        kb = tmp_path / "wikipedia_nuclear_energy.jsonl"
        rows = [
            {"doc_id": "wiki-nuclear_energy-000", "topic": "nuclear_energy",
             "title": "T", "url": "u", "text": "France uses nuclear power.", "source": "wikipedia", "tier": 1},
        ]
        kb.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        out = load_kb("nuclear_energy", kb_dir=tmp_path)
        assert out == [{"doc_id": "wiki-nuclear_energy-000", "text": "France uses nuclear power.",
                        "title": "T", "url": "u"}]

    def test_missing_topic_raises(self, tmp_path):
        from src.nlp.kb import load_kb
        import pytest
        with pytest.raises(FileNotFoundError):
            load_kb("absent", kb_dir=tmp_path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
