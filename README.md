# Echo-Search

**Un motore di ricerca che, sui temi divisivi, non ti chiude in una sola prospettiva.**

Progetto per il corso di *Information Retrieval e Natural Language Processing* — Università della Calabria, A.A. 2025/2026.
[Davide Cinelli](https://github.com/davidecinelli26) · [Giuseppe Cuiuri](https://github.com/GiuseppeCuiuri)

---

## Il problema

Un motore di ricerca ottimizza la **pertinenza**, non la **diversità**: non è il suo compito. Su un tema divisivo questo ha una conseguenza precisa — la query porta con sé un'inclinazione, anche solo nelle parole scelte, e il ranking la amplifica.

Echo-Search risponde con tre mosse:

1. **I risultati non sono una lista, sono tre colonne** affiancate — a favore, neutrale, contrario.
2. **L'omogeneità diventa un numero**: la *Stance Diversity*, entropia di Shannon normalizzata sulle posizioni dei documenti recuperati.
3. **La stessa identica pipeline gira su due ecosistemi**, articoli di cronaca e tweet, così che i due numeri siano confrontabili.

Tre temi: energia nucleare, cambiamento climatico, vaccini. Tutto in locale, entro 6 GB di VRAM, senza API a pagamento.

---

## Architettura

```
                                   ┌─ stance (DeBERTa-v3 FT) ─→ colonne ─→ diversity
corpus ─→ retrieval ibrido ─→ doc ─┤                                    └→ riassunti
          BM25 ⊕ bge/FAISS         └─ fact-check (NLI | LLM) ────────────→ dashboard
             + reranker
```

Stance e fact-check sono **rami indipendenti** sugli stessi documenti: non si scambiano nulla. È ciò che tiene il fact-checking — di gran lunga il modulo più lento — fuori dal cammino critico.

**Retrieval a due stadi.** Il primo *recupera*: BM25 (`b = 0,60`, i documenti sono tutti lunghi) fuso con un retrieval denso `bge-base` su indice FAISS, punteggi normalizzati min-max e combinati con α = 0,45. Il secondo *riordina*: un cross-encoder `bge-reranker` legge query e documento insieme sui soli 50 candidati. Il tetto del reranker è il recall del primo stadio.

**Stance detection.** `DeBERTa-v3-base` con fine-tuning completo su VAST ∪ SemEval-2016 (16 097 esempi, 4 644 target distinti). La generalizzazione a temi mai visti viene dalla varietà dei target in addestramento, non da accorgimenti architetturali.

**Fact-checking in tre fasi**, di cui le prime due condivise e la terza sostituibile:

| Fase | Modulo | Cosa fa |
|---|---|---|
| 1 | `nlp/claims.py` | estrae l'affermazione più verificabile del documento |
| 2 | `nlp/kb_evidence.py` | cerca le evidenze nella KB Wikipedia e applica il **cancello di pertinenza** |
| 3 | `nlp/nli_verdict.py` **oppure** `nlp/llm_verdict.py` | emette il verdetto |

Il cancello è la parte che conta: su una base tematica il retrieval restituisce *sempre* qualcosa, e un'evidenza solo tematica diventa un falso REFUTED. Sotto la soglia di somiglianza nessun motore viene interrogato. Poiché cambia solo la terza fase, la differenza misurata fra NLI e LLM è attribuibile al motore e non al recupero.

---

## Risultati

| Componente | Benchmark | Risultato |
|---|---|---|
| Retrieval, ablazione a 6 sistemi | qrels propri, 380 coppie | **NDCG@10 0,715** (fusione + reranking) |
| Accordo fra annotatori | 2 annotatori umani in cieco | **κ quadratico 0,867** (*almost perfect*) |
| Stance detection | VAST ∪ SemEval, test held-out | **Macro-F1 0,771** (zero-shot: 0,390) |
| Stance, cross-dataset | VAST / SemEval | 0,768 / 0,741 |
| Fact-check, motore NLI | ClimateFEVER, 1381 claim | Macro-F1 0,350 |
| Fact-check, motore LLM | ClimateFEVER, 1381 claim | **Macro-F1 0,640** |
| Stance Diversity | query canoniche | news 0,229–0,545 · social 0,600–0,621 |

Tre cose che i numeri dicono, e una che non dicono.

Il **reranking** è il contributo singolo più grande (+6,1% di NDCG, a recall invariato: riordina, non recupera). **Tarare α peggiora**: con 11 query la curva è piatta e la ricerca a griglia seleziona rumore, quindi il default scelto per ragioni di dominio resta entro lo 0,4% dall'ottimo misurabile. A parità di evidenze, l'**LLM locale supera l'NLI proprio dove questo falliva** — SUPPORTED da 0,065 a 0,753 — ma è più debole nel confutare: i due sbagliano in modo complementare.

Ciò che i numeri **non** dicono: che i social siano più diversi delle news. Il corpus social è campionato in modo bilanciato per costruzione, quindi il *confronto* fra i due ecosistemi è confondato e sovrastima la differenza. I valori assoluti sulle news, campione naturale, restano validi. La contromisura — il *diversity lift* — è definita nella relazione e non ancora calcolata.

Documentazione completa in [`docs/relazione/`](docs/relazione/).

---

## Installazione

Richiede **Python 3.11** e [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run python -m spacy download en_core_web_sm
uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Il motore di verdetto LLM è facoltativo e richiede [Ollama](https://ollama.com) in esecuzione in locale:

```bash
ollama pull qwen2.5:7b
```

---

## Uso

### Dashboard

```bash
uv run streamlit run dashboard.py
```

Le query canoniche sono precalcolate: la demo è istantanea e non richiede GPU. Selezionando una posizione la colonna si allarga e **le altre restano visibili** — un filtro rigido, qui, ricostruirebbe l'echo chamber.

### Riga di comando

```bash
uv run python main.py "is nuclear power safe" --topic nuclear_energy
```

```bash
uv run python main.py "are vaccines safe" --topic vaccines --fact-check --engine llm
```

```bash
uv run python main.py -i --topic climate_change
```

Dopo `uv sync` è disponibile anche come comando installato: `uv run echo-search -i`.

> **GPU.** L'ambiente di base installa `torch` in versione CPU: tutto funziona, ma una ricerca
> con fact-check richiede minuti invece di secondi. Per usare la scheda video serve un ambiente
> effimero con l'indice CUDA — e va passato **sempre lo stesso identico insieme** di `--with`,
> altrimenti uv riusa il torch CPU già presente senza dirlo:
>
> ```bash
> uv run --no-sync --with "torch==2.2.2" --with "accelerate>=0.30" --with "datasets>=2.19" --index "https://download.pytorch.org/whl/cu121" python main.py "is nuclear power safe" --topic nuclear_energy --fact-check
> ```
>
> Per verificare che sia andata a buon fine: `torch.cuda.is_available()` deve essere `True`.

### Da codice

```python
from src.app import EchoSearch

app = EchoSearch(topic="climate_change")
esito = app.search("is climate change caused by humans", target="global warming")

for eco, blocco in esito["ecosystems"].items():
    print(eco, blocco["diversity"]["diversity_score"], blocco["pool_distribution"])
    for etichetta, docs in blocco["columns"].items():
        print(" ", etichetta, len(docs))
```

---

## Ricostruire tutto da zero

Gli indici e i modelli non sono versionati: si rigenerano dagli script. Ogni script di costruzione dati emette un `MANIFEST.json` con gli hash SHA-256 degli artefatti prodotti.

```bash
uv run --extra data python scripts/build_corpora.py       # corpus news da CC-News
uv run python scripts/build_social_corpus.py              # corpus social (tweet etichettati)
uv run python scripts/build_kb.py                         # KB Wikipedia
uv run python scripts/build_stance_splits.py              # split VAST + SemEval
uv run --extra data python scripts/index_corpus.py        # indici BM25 + FAISS
uv run python scripts/train_stance.py                     # fine-tuning dello stance detector
uv run python scripts/precompute_demo.py                  # cache della demo
```

## Riprodurre le misure

```bash
uv run python scripts/eval_retrieval.py                   # ablazione a 6 sistemi
uv run python scripts/eval_agreement.py                   # κ fra i due annotatori
uv run python scripts/eval_stance.py                      # stance su benchmark
uv run python scripts/evaluate_pivot.py                   # stance social + diversity
uv run python scripts/eval_climatefever.py --engine nli   # fact-check, motore NLI
uv run python scripts/eval_climatefever.py --engine llm   # fact-check, motore LLM
```

I giudizi di rilevanza annotati a mano sono in `data/gold/` (versionati: sono lavoro umano, non rigenerabili). I fogli vuoti da annotare si producono con `scripts/build_qrels_annotation.py`; le linee guida seguite sono in [`docs/annotazione-qrels.md`](docs/annotazione-qrels.md).

---

## Struttura

```
src/
  app.py               orchestrazione della pipeline + CLI
  retrieval/           sparse (BM25) · dense (bge+FAISS) · hybrid · reranker · column_balancer
  nlp/                 stance_detector · claims · kb_evidence · nli_verdict · llm_verdict
                       fact_checker (compone le tre fasi) · summarizer · kb
  evaluation/          metrics (pure) · qrels (caricamento dei giudizi)
  preprocessing/       text_preprocessor
  gui/                 format (decisioni pure) · render (HTML)
  utils/               config · device · logger · ranking · text
scripts/               costruzione dei dati e valutazione (dipendenze pesanti, mai importati dalla pipeline)
configs/               config.yaml (iperparametri, ognuno motivato) · topics.yaml · sources.yaml
data/                  corpora, KB, indici, gold set, cache della demo
tests/                 suite pytest
docs/                  relazione, presentazione, linee guida di annotazione
```

Una convenzione attraversa tutto il codice: **funzioni pure separate dall'I/O**. Parsing, campionamento, metriche e aggregazioni non toccano disco né rete, e solo `main()` lo fa. È la ragione per cui la suite copre le parti che contano senza GPU e senza download.

---

## Test

```bash
uv run pytest tests -q
```

174 test, circa 30 secondi, nessuna rete e nessuna GPU.

---

## Dati e licenze

Il codice è sotto licenza [MIT](LICENSE). I dati che il progetto usa hanno licenze proprie:

- **Notizie:** articoli di [CC-News](https://commoncrawl.org/blog/news-dataset-available), un crawl del web pubblico. Il testo resta degli editori e viene incluso solo in `data/gold/` (giudizi di rilevanza) e `data/demo_cache/` (demo precalcolata), a scopo di ricerca e didattica.
- **Stance:** [VAST](https://github.com/emilyallaway/zero-shot-stance) e SemEval-2016 (Task 6).
- **Fact-checking:** [ClimateFEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html) per la valutazione e Wikipedia (CC BY-SA) per la base di conoscenza.

Corpora, KB e modelli addestrati non sono versionati: si rigenerano con gli script indicati sopra.
