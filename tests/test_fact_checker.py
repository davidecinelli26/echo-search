"""
Echo-Search | Test — fact-checking
==================================
Copre le parti DETERMINISTICHE delle tre fasi, quelle in cui un errore
produrrebbe un verdetto sbagliato senza segnalarsi:

- l'aggregazione delle probabilita' NLI in un verdetto (logica NumPy pura);
- la raccolta delle fonti da mostrare all'utente;
- l'orchestrazione su colonne, verificata con motore ed evidenze finti.

Modelli e rete non entrano mai: i motori sono lazy e qui vengono sostituiti da
doppi, quindi la suite gira senza GPU, senza download e senza Ollama.

Esecuzione: uv run pytest tests/test_fact_checker.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.fact_checker import FactChecker  # noqa: E402
from src.nlp.kb_evidence import unique_sources  # noqa: E402
from src.nlp.nli_verdict import (  # noqa: E402
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
    aggregate,
)

SUPPORT_TH, REFUTE_TH = 0.75, 0.95


class TestAggregate:
    def test_supported_sopra_soglia(self):
        entail = np.array([0.80, 0.30])
        contra = np.array([0.10, 0.30])
        assert aggregate(entail, contra, SUPPORT_TH, REFUTE_TH) == (VERDICT_SUPPORTED, 0)

    def test_refuted_richiede_contraddizione_fortissima(self):
        entail, contra = np.array([0.20]), np.array([0.96])
        assert aggregate(entail, contra, SUPPORT_TH, REFUTE_TH)[0] == VERDICT_REFUTED

    def test_contraddizione_alta_ma_non_fortissima_resta_nei(self):
        # 0,90 basterebbe con soglie simmetriche: qui no, ed e' voluto
        entail, contra = np.array([0.05]), np.array([0.90])
        assert aggregate(entail, contra, SUPPORT_TH, REFUTE_TH)[0] == VERDICT_NEI

    def test_nessun_segnale_forte(self):
        entail, contra = np.array([0.40, 0.30]), np.array([0.30, 0.35])
        assert aggregate(entail, contra, SUPPORT_TH, REFUTE_TH)[0] == VERDICT_NEI

    def test_soglia_inclusiva(self):
        assert aggregate(np.array([0.75]), np.array([0.10]),
                         SUPPORT_TH, REFUTE_TH)[0] == VERDICT_SUPPORTED

    def test_basta_una_evidenza_convincente(self):
        # la media sarebbe sotto soglia: si guarda il massimo, non la media
        entail = np.array([0.05, 0.05, 0.90])
        contra = np.array([0.10, 0.10, 0.05])
        assert aggregate(entail, contra, SUPPORT_TH, REFUTE_TH) == (VERDICT_SUPPORTED, 2)

    def test_indice_dell_evidenza_decisiva(self):
        entail, contra = np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.97, 0.1])
        verdetto, decisiva = aggregate(entail, contra, SUPPORT_TH, REFUTE_TH)
        assert (verdetto, decisiva) == (VERDICT_REFUTED, 1)


class TestUniqueSources:
    def test_deduplica_per_url_conservando_l_ordine(self):
        evidenze = [
            {"text": "a", "title": "Prima", "url": "u1"},
            {"text": "b", "title": "Prima", "url": "u1"},
            {"text": "c", "title": "Seconda", "url": "u2"},
        ]
        assert unique_sources(evidenze) == [{"title": "Prima", "url": "u1"},
                                            {"title": "Seconda", "url": "u2"}]

    def test_scarta_le_evidenze_senza_url(self):
        assert unique_sources([{"text": "a", "title": "T", "url": ""}]) == []

    def test_rispetta_il_limite(self):
        evidenze = [{"text": "x", "title": str(i), "url": f"u{i}"} for i in range(10)]
        assert len(unique_sources(evidenze, limit=3)) == 3


class _MotoreFinto:
    """Motore di verdetto deterministico: registra le chiamate ricevute."""

    def __init__(self, verdetto=VERDICT_SUPPORTED):
        self.verdetto = verdetto
        self.chiamate = []

    def verdict(self, claim, evidences):
        self.chiamate.append((claim, tuple(evidences)))
        return {"verdict": self.verdetto, "reasoning": "motivo"}


def _fact_checker(monkeypatch, evidenze, claims=("una claim",), per_column=None):
    """FactChecker con fasi 1 e 2 sostituite da doppi: nessun modello, nessuna KB."""
    fc = FactChecker("nuclear_energy", per_column=per_column)
    monkeypatch.setattr(fc.claims, "extract", lambda text, max_claims=1: list(claims))
    monkeypatch.setattr(fc.evidence, "retrieve", lambda claim: evidenze)
    motore = _MotoreFinto()
    fc._engine = motore
    return fc, motore


class TestCheckClaim:
    def test_senza_evidenze_pertinenti_non_interroga_il_motore(self, monkeypatch):
        # il cancello di pertinenza ha gia' deciso: interrogare il motore
        # significherebbe pagarlo per produrre un falso REFUTED
        fc, motore = _fact_checker(monkeypatch, evidenze=[])
        esito = fc.check_claim("una claim")
        assert esito == {"claim": "una claim", "verdict": VERDICT_NEI,
                         "reasoning": "", "sources": []}
        assert motore.chiamate == []

    def test_passa_al_motore_i_soli_testi_delle_evidenze(self, monkeypatch):
        evidenze = [{"text": "prova", "title": "T", "url": "u"}]
        fc, motore = _fact_checker(monkeypatch, evidenze)
        esito = fc.check_claim("una claim")
        assert motore.chiamate == [("una claim", ("prova",))]
        assert esito["verdict"] == VERDICT_SUPPORTED
        assert esito["sources"] == [{"title": "T", "url": "u"}]


class TestCheckColumns:
    def _colonne(self, n=3):
        return {"FAVOR": [{"doc_id": str(i), "text": "t"} for i in range(n)],
                "AGAINST": [{"doc_id": "x", "text": "t"}]}

    def test_arricchisce_i_documenti_verificati(self, monkeypatch):
        fc, _ = _fact_checker(monkeypatch, [{"text": "e", "title": "T", "url": "u"}])
        out = fc.check_columns(self._colonne(1))
        doc = out["FAVOR"][0]
        assert doc["factuality"] == {VERDICT_SUPPORTED: 1}
        assert doc["fact_checks"][0]["claim"] == "una claim"

    def test_per_column_limita_i_documenti_verificati(self, monkeypatch):
        fc, _ = _fact_checker(monkeypatch, [{"text": "e", "title": "T", "url": "u"}],
                              per_column=2)
        out = fc.check_columns(self._colonne(3))
        assert [("fact_checks" in d) for d in out["FAVOR"]] == [True, True, False]

    def test_documento_senza_claim_passa_invariato(self, monkeypatch):
        fc, _ = _fact_checker(monkeypatch, [{"text": "e", "title": "T", "url": "u"}],
                              claims=())
        out = fc.check_columns(self._colonne(1))
        assert out["FAVOR"][0] == {"doc_id": "0", "text": "t"}

    def test_non_muta_le_colonne_in_ingresso(self, monkeypatch):
        fc, _ = _fact_checker(monkeypatch, [{"text": "e", "title": "T", "url": "u"}])
        colonne = self._colonne(1)
        fc.check_columns(colonne)
        assert "fact_checks" not in colonne["FAVOR"][0]

    def test_preserva_le_etichette_delle_colonne(self, monkeypatch):
        fc, _ = _fact_checker(monkeypatch, [{"text": "e", "title": "T", "url": "u"}])
        assert set(fc.check_columns(self._colonne()).keys()) == {"FAVOR", "AGAINST"}


class TestEngineSelection:
    def test_motore_sconosciuto_rifiutato_subito(self):
        # meglio fallire alla costruzione che a meta' di una ricerca
        with pytest.raises(ValueError, match="engine"):
            FactChecker("nuclear_energy", engine="gemini")

    def test_i_motori_validi_non_caricano_nulla_alla_costruzione(self):
        for engine in ("nli", "llm"):
            assert FactChecker("nuclear_energy", engine=engine)._engine is None
