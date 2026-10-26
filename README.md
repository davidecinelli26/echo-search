# Echo-Search

**A search engine that, on divisive topics, doesn't lock you into a single perspective.**

Project for the *Information Retrieval and Natural Language Processing* course, University of Calabria, A.Y. 2025/2026.
[Davide Cinelli](https://github.com/davidecinelli26) · [Giuseppe Cuiuri](https://github.com/GiuseppeCuiuri)

---

## The problem

A search engine optimizes **relevance**, not **diversity**: that is not its job. On a divisive topic this has a precise consequence: the query carries a slant, even just in the words chosen, and the ranking amplifies it.

Echo-Search answers with three moves:

1. **Results are not a list, they are three columns** side by side: in favor, neutral, against.
2. **Homogeneity becomes a number**: the *Stance Diversity*, the normalized Shannon entropy of the stances of the retrieved documents.
3. **The exact same pipeline runs on two ecosystems**, news articles and tweets, so that the two numbers are comparable.

Three topics: nuclear energy, climate change, vaccines. Everything runs locally, within 6 GB of VRAM, with no paid APIs.

---

## Architecture

```
                                   ┌─ stance (DeBERTa-v3 FT) ─→ columns ─→ diversity
corpus ─→ hybrid retrieval ─→ doc ─┤                                    └→ summaries
          BM25 ⊕ bge/FAISS         └─ fact-check (NLI | LLM) ────────────→ dashboard
             + reranker
```

Stance and fact-check are **independent branches** over the same documents: they exchange nothing. This is what keeps fact-checking, by far the slowest module, off the critical path.

**Two-stage retrieval.** The first stage *retrieves*: BM25 (`b = 0.60`, since all documents are long) fused with `bge-base` dense retrieval on a FAISS index, with min-max normalized scores combined at α = 0.45. The second stage *reorders*: a `bge-reranker` cross-encoder reads query and document together, on the top 50 candidates only. The ceiling of the reranker is the recall of the first stage.

**Stance detection.** `DeBERTa-v3-base`, fully fine-tuned on VAST ∪ SemEval-2016 (16,097 examples, 4,644 distinct targets). Generalization to unseen topics comes from the variety of targets in training, not from architectural tricks.

**Three-phase fact-checking**, of which the first two are shared and the third is swappable:

| Phase | Module | What it does |
|---|---|---|
| 1 | `nlp/claims.py` | extracts the most verifiable claim of the document |
| 2 | `nlp/kb_evidence.py` | searches the Wikipedia KB for evidence and applies the **relevance gate** |
| 3 | `nlp/nli_verdict.py` **or** `nlp/llm_verdict.py` | issues the verdict |

The gate is the part that matters: on a topical knowledge base, retrieval *always* returns something, and evidence that is only on-topic becomes a false REFUTED. Below the similarity threshold no engine is queried. Since only the third phase changes, the measured difference between NLI and LLM is attributable to the engine and not to retrieval.

---

## Results

| Component | Benchmark | Result |
|---|---|---|
| Retrieval, 6-system ablation | own qrels, 380 pairs | **NDCG@10 0.715** (fusion + reranking) |
| Inter-annotator agreement | 2 blind human annotators | **quadratic κ 0.867** (*almost perfect*) |
| Stance detection | VAST ∪ SemEval, held-out test | **Macro-F1 0.771** (zero-shot: 0.390) |
| Stance, cross-dataset | VAST / SemEval | 0.768 / 0.741 |
| Fact-check, NLI engine | ClimateFEVER, 1,381 claims | Macro-F1 0.350 |
| Fact-check, LLM engine | ClimateFEVER, 1,381 claims | **Macro-F1 0.640** |
| Stance Diversity | canonical queries | news 0.229–0.545 · social 0.600–0.621 |

Three things the numbers say, and one they don't.

**Reranking** is the single biggest contribution (+6.1% NDCG, with recall unchanged: it reorders, it doesn't retrieve). **Tuning α makes things worse**: with 11 queries the curve is flat and the grid search selects noise, so the default chosen on domain grounds stays within 0.4% of the measurable optimum. At equal evidence, the **local LLM beats NLI exactly where the latter failed**, with SUPPORTED going from 0.065 to 0.753, but it is weaker at refuting: the two err in complementary ways.

What the numbers do **not** say: that social media are more diverse than news. The social corpus is sampled in a balanced way by construction, so the *comparison* between the two ecosystems is confounded and overstates the difference. The absolute values on news, a natural sample, remain valid. The countermeasure, the *diversity lift*, is defined in the report and not yet computed.

The full documentation is in [`docs/relazione/`](docs/relazione/) (the report is in Italian; so are the code comments and the dashboard interface).

---

## Installation

Requires **Python 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run python -m spacy download en_core_web_sm
uv run python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

The LLM verdict engine is optional and requires [Ollama](https://ollama.com) running locally:

```bash
ollama pull qwen2.5:7b
```

---

## Usage

### Dashboard

```bash
uv run streamlit run dashboard.py
```

The canonical queries are precomputed: the demo is instant and needs no GPU. When you select a stance the column widens and **the others stay visible**: a hard filter, here, would rebuild the echo chamber.

### Command line

```bash
uv run python main.py "is nuclear power safe" --topic nuclear_energy
```

```bash
uv run python main.py "are vaccines safe" --topic vaccines --fact-check --engine llm
```

```bash
uv run python main.py -i --topic climate_change
```

After `uv sync` it is also available as an installed command: `uv run echo-search -i`.

> **GPU.** The base environment installs the CPU build of `torch`: everything works, but a search
> with fact-checking takes minutes instead of seconds. To use the graphics card you need an
> ephemeral environment with the CUDA index, and you must **always pass the exact same set** of
> `--with`, otherwise uv silently reuses the CPU torch already present:
>
> ```bash
> uv run --no-sync --with "torch==2.2.2" --with "accelerate>=0.30" --with "datasets>=2.19" --index "https://download.pytorch.org/whl/cu121" python main.py "is nuclear power safe" --topic nuclear_energy --fact-check
> ```
>
> To check that it worked: `torch.cuda.is_available()` must be `True`.

### From code

```python
from src.app import EchoSearch

app = EchoSearch(topic="climate_change")
result = app.search("is climate change caused by humans", target="global warming")

for eco, block in result["ecosystems"].items():
    print(eco, block["diversity"]["diversity_score"], block["pool_distribution"])
    for label, docs in block["columns"].items():
        print(" ", label, len(docs))
```

---

## Rebuilding everything from scratch

Indexes and models are not versioned: they are regenerated by the scripts. Every data-building script emits a `MANIFEST.json` with the SHA-256 hashes of the artifacts it produced.

```bash
uv run --extra data python scripts/build_corpora.py       # news corpus from CC-News
uv run python scripts/build_social_corpus.py              # social corpus (labeled tweets)
uv run python scripts/build_kb.py                         # Wikipedia KB
uv run python scripts/build_stance_splits.py              # VAST + SemEval splits
uv run --extra data python scripts/index_corpus.py        # BM25 + FAISS indexes
uv run python scripts/train_stance.py                     # stance detector fine-tuning
uv run python scripts/precompute_demo.py                  # demo cache
```

## Reproducing the measurements

```bash
uv run python scripts/eval_retrieval.py                   # 6-system ablation
uv run python scripts/eval_agreement.py                   # κ between the two annotators
uv run python scripts/eval_stance.py                      # stance on benchmarks
uv run python scripts/evaluate_pivot.py                   # social stance + diversity
uv run python scripts/eval_climatefever.py --engine nli   # fact-check, NLI engine
uv run python scripts/eval_climatefever.py --engine llm   # fact-check, LLM engine
```

The hand-annotated relevance judgments are in `data/gold/` (versioned: they are human work and cannot be regenerated). The empty sheets to annotate are produced with `scripts/build_qrels_annotation.py`; the guidelines followed are in [`docs/annotation-qrels.md`](docs/annotation-qrels.md).

---

## Structure

```
src/
  app.py               pipeline orchestration + CLI
  retrieval/           sparse (BM25) · dense (bge+FAISS) · hybrid · reranker · column_balancer
  nlp/                 stance_detector · claims · kb_evidence · nli_verdict · llm_verdict
                       fact_checker (composes the three phases) · summarizer · kb
  evaluation/          metrics (pure) · qrels (judgment loading)
  preprocessing/       text_preprocessor
  gui/                 format (pure decisions) · render (HTML)
  utils/               config · device · logger · ranking · text
scripts/               data building and evaluation (heavy dependencies, never imported by the pipeline)
configs/               config.yaml (hyperparameters, each one justified) · topics.yaml · sources.yaml
data/                  corpora, KB, indexes, gold set, demo cache
tests/                 pytest suite
docs/                  report, presentation, annotation guidelines
```

One convention runs through the whole code: **pure functions separated from I/O**. Parsing, sampling, metrics and aggregations touch neither disk nor network, and only `main()` does. This is why the suite covers the parts that matter without a GPU and without downloads.

---

## Tests

```bash
uv run pytest tests -q
```

174 tests, about 30 seconds, no network and no GPU.

---

## Data and licenses

The code is under the [MIT](LICENSE) license. The data the project uses has its own licenses:

- **News:** articles from [CC-News](https://commoncrawl.org/blog/news-dataset-available), a crawl of the public web. The text remains the publishers' and is included only in `data/gold/` (relevance judgments) and `data/demo_cache/` (precomputed demo), for research and teaching purposes.
- **Stance:** [VAST](https://github.com/emilyallaway/zero-shot-stance) and SemEval-2016 (Task 6).
- **Fact-checking:** [ClimateFEVER](https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html) for evaluation and Wikipedia (CC BY-SA) for the knowledge base.

Corpora, KBs and trained models are not versioned: they are regenerated with the scripts listed above.
