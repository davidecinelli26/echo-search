# Annotation guidelines: Qrels (retrieval relevance)

We annotate the **relevance of the document to the query**, NOT its stance, NOT its
truthfulness, NOT the quality of the source. It is the judgment that feeds NDCG@10, MAP and Recall@20.

## The rule that matters most

> **Relevant ≠ agreeing. Relevant ≠ true.**

The corpus is made of divisive topics, so the temptation to penalize wrong or unpleasant
documents is strong. It must be resisted: a climate-denial article that answers the query
`is climate change caused by humans` is **fully relevant (2)**.

If we confuse relevance and position, we measure the search engine on its ability to hide
uncomfortable opinions, which is exactly the opposite of what Echo-Search claims to do.
Truthfulness is the fact-checker's job, position is the stance detector's job: neither
has anything to do with it here.

## The scale

| Grade | Meaning | Operational criterion |
|---|---|---|
| **2** | Fully relevant | Answers the query, or covers its core substantially. If a user searched for that query, they would be glad to find it on the first page. |
| **1** | Marginally relevant | Covers the topic but does not answer the query: touches it in passing, approaches it from a different angle, or contains only a useful fragment. |
| **0** | Not relevant | Does not answer the query. Includes both pure off-topic and articles that discuss the general topic but have no relation to what the query asks. |

**Telling 2 from 1** is the decision that makes the difference for NDCG: ask yourself *"does this
document answer the question, or does it only talk about the subject of the question?"*

Examples on the real queries:

- `risks of nuclear energy`: an article on the Fukushima accident and its health consequences
  → **2**. An article on the construction costs of a plant, mentioning the risks in one line → **1**.
  A press release on a trade agreement for uranium → **0**.
- `do vaccines cause autism`: an article on the Wakefield study and its retraction
  → **2**. An article on vaccine hesitancy that cites autism among the widespread fears →
  **1**. An article on a polio campaign in Ghana → **0**.
- `economic cost of climate policy`: an analysis of the economic impact of a carbon tax →
  **2**. An article on a climate summit that mentions costs in passing → **1**. An article on the
  physical effects of warming → **0**.

## The 0/1/2 boundary: operational specification (v2)

The first version of these guidelines defined 0 and 1 in an overlapping way ("talks about the general
topic but has no relation to the query" for 0, "covers the topic but does not answer" for
1): the same sentence said twice. Defined that way, the boundary is an intention and not a
criterion, and the ambiguity bites where the query is abstract and the corpus is saturated on the
topic, that is, on the climate queries. This is the specification that resolves it, formulated as a
**test that can be verified on the text**.

| Grade | Operational test |
|---|---|
| **2** | You can **point to the sentence** of the document that constitutes the answer to the query. |
| **1** | There is no answer sentence, but the document provides useful context or touches the question sideways. You would open it as the third or fourth result and something would come of it. |
| **0** | The link with the query is **only thematic**: the document talks about the topic (nuclear / climate / vaccines) but not about the question. Includes pure off-topic. |

### Corollaries, deduced from real cases in the corpus

- **If the query asks for a cost, you need a cost.** A figure, an estimate, or an explicit
  statement about who pays and how much. A dispute over fund transfers between public bodies
  is not the cost of a policy. And the cost *of damages* (social cost of carbon) is not the
  cost *of the policy*: they have opposite signs.
- **If the query asks for a relation between two things** (climate↔extreme events, nuclear↔climate),
  the text must **assert the relation**. It is not enough for both terms to appear.
- **If the query asks for a risk or an effect,** the document must talk about that
  risk, not about the topic in which the risk exists. The financial failure of a project
  is not a "risk of nuclear energy" unless the text frames it as such.
- **The source does not matter** (already said above, worth repeating here): a
  denialist document that answers the question is a 2.
- **A syndicated duplicate gets the same grade as its copy.** The corpus contains
  many re-runs of the same wire story: they are a single editorial decision.

## Edge cases

- **The excerpt is not enough to decide.** The sheet shows the title and the opening of the article
  (~1,200 characters). If that is not enough, retrieve the full text from the `doc_id` in
  `data/corpora/{topic}.jsonl`. Failing that, judge on what you see: it is the same
  information the models have on the first chunks.
- **Off-topic by homonymy.** The keyword filter of the corpora lets some noise through
  (the known case: the *VACCINE* research center at Purdue, an acronym that has nothing to do
  with vaccines). Grade **0**, and flag it in the `nota` column: the share of these cases
  measures the residual noise of the corpus.
- **The same document under different queries.** Normal and intended: relevance is per
  *(query, document)* pair, not per document. The same article can be a 2 for one query and a 0
  for another. Judge it each time against the query of that row.
- **Pertinent but old, short or poorly written article.** Editorial quality does not enter
  the judgment. If it answers the query, it is relevant.
- **In doubt between 1 and 0** → 0. **In doubt between 2 and 1** → 1. The conservative default keeps
  drift between annotators low.

## Protocol (for Cohen's kappa)

1. Each annotator fills in the `relevance` column of **their own** sheet:
   `data/gold/qrels_{topic}_davide.csv` and `data/gold/qrels_{topic}_giuseppe.csv`.
   The two sheets are identical in content and order: **380 pairs in total**
   (137 nuclear, 109 climate, 134 vaccines).
2. **Complete and independent double annotation.** Each person judges everything, without consulting
   the other and without comparing sheets during the work: kappa measures agreement, and if you
   align along the way the value is inflated and means nothing.
3. The `nota` column is free-form: use it for doubtful cases, it is needed during adjudication.
4. At the end of annotation, **Cohen's kappa is computed on all 380 pairs**. Reference:
   κ ≥ 0.6 substantial, κ ≥ 0.8 almost perfect. If κ < 0.5 it is better to revise these guidelines
   and re-annotate, rather than proceed with unreliable qrels.
5. **Adjudication**: only the rows in disagreement are examined together and the final grade is
   decided. The consolidated result becomes `qrels_{topic}.csv`, the artifact used by the metrics.
6. Report in the paper: κ per topic and overall, number of divergences, and the
   distribution of grades (how many 0/1/2).

## Outcome and notes on the data

Annotation completed on the 380 pairs by both annotators. Quadratic-weighted Cohen's κ
**0.867** overall (nuclear 0.904, climate 0.814, vaccines 0.856), unweighted
0.675, raw agreement 0.797; 77 divergences, only **2 of them by two grades**. The computation is
reproducible with `uv run --no-sync python scripts/eval_agreement.py`, which saves the
detail (confusion matrices and marginals) in `data/gold/agreement_report.json`.

A word of caution on the data: the query `nuclear waste disposal` has 28 relevant out of 31, and
`are vaccines safe` 24 out of 28 (identical for both annotators). On these two the pool
is almost saturated with relevant documents, so the metrics will discriminate little between systems: this
should be said in the report rather than left to be inferred.

## Practical notes

- The order of rows is **randomized within each query** and the `rank`/`score` columns do not
  appear: this is deliberate. Seeing that a document is in first place predisposes you to judge it
  relevant (*position bias*), and would distort the very metric we are building.
- The pool is the **union** of the documents retrieved by five different runs (BM25, dense, RRF,
  fusion, reranker pool). **143 pairs out of 380, 38%, were retrieved by a
  single run**: they are exactly the ones a pooling from the fusion run alone would have missed,
  counting them as non-relevant and distorting the comparison to the detriment of the baselines.
- The files are UTF-8 with BOM: Excel on Windows opens them with accents correct. If you save
  from Excel, keep the CSV UTF-8 format.
- Estimated time: **~1h40–2h each**, at 15–20 seconds per judgment. It is better done in two
  sessions per topic and not all in one go: judgment drift grows with fatigue.
- Recommended order: one topic at a time, from start to finish, without jumping between sheets.
  Queries on the same topic share context and you go faster.
