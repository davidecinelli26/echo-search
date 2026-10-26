# Annotation guidelines: Stance (Phase 3, task 3.1)

We annotate the **stance of the document toward the topic's `target`**
(nuclear energy / climate change / vaccine safety), NOT the sentiment nor the truthfulness.

## The target of each topic (WHAT to judge)

The stance is toward an **implicit proposition**, not the topic in the abstract. For each topic, judge whether the document supports or opposes this statement:

| Topic | Target proposition | FAVOR = | AGAINST = |
|---|---|---|---|
| **nuclear energy** | nuclear energy should be used/expanded as a source | pro nuclear (clean, safe, build) | against (dangerous, phase out) |
| **climate change** | climate change is a real and urgent threat | recognizes reality/urgency, pro action | skeptical, denies, downplays |
| **vaccine safety** | vaccines are safe and effective | pro vaccines (safe, recommended) | anti / hesitant (dangerous, hidden effects) |

**Examples:**
- Nuclear — FAVOR: "It is the clean, reliable source we need." · AGAINST: "Too dangerous, it must be phased out."
- Climate — FAVOR: "We must act now, it's an emergency." · AGAINST: "The alarmism is exaggerated, models are unreliable."
- Vaccines — FAVOR: "Safe, they prevent millions of deaths." · AGAINST: "They cause serious hidden effects."

**Climate change, edge cases:** "favor/against" does NOT mean being in favor of warming (we follow the SemEval-2016 convention *"Climate Change is a Real Concern"*, which is in the training data). The axis is **reality + urgency**: whoever denies/downplays → AGAINST; whoever recognizes the problem and supports action → FAVOR. Opposition to **a single policy** (e.g. a specific carbon tax) **without** denying the underlying problem → **NEUTRAL**.

## The classes
- **FAVOR**: the document supports/defends the target (arguments in favor prevail).
- **AGAINST**: the document opposes the target (arguments against prevail).
- **NEUTRAL**: the document **talks about the topic** but no position prevails:
  - *neutral*: factual reporting with no stance;
  - *balanced*: presents pros and cons with equal weight.
- **OFF_TOPIC**: the document **is not about the topic**: the keyword appears by homonymy,
  acronym or incidental mention. Real example from the corpus: an article on the Purdue
  *VACCINE* research center (acronym of *Visual Analytics for Command, Control and
  Interoperability Environments*), about social media analytics for civil
  protection, with no link to vaccines. **It is not a stance**: it is noise from the
  corpus filter, and must be marked separately, **never** as NEUTRAL.

> **Why OFF_TOPIC is separate from NEUTRAL:** it keeps the NEUTRAL class clean (which means
> "talks about the topic without taking sides"), does not destroy information during annotation, and
> measures how much noise the corpus contains. When building the gold set you decide whether to
> **exclude** them (default: the gold measures stance on pertinent documents) or **map them to
> NEUTRAL** (measures end-to-end behavior); having the distinct label you can
> report both variants. A share of OFF_TOPIC > 10-15% signals that the keyword
> filter of the corpora needs hardening.

## Protocol (for Cohen's kappa)
1. Each annotator fills in the `label` column of their own `{topic}_to_annotate.csv`.
2. The `doc_id`s in `{topic}_overlap_ids.json` are annotated by BOTH, independently.
3. On the overlap, Cohen's kappa (agreement) is computed. Target: κ ≥ 0.6 (substantial).
   If κ is low, these guidelines are revised and the overlap is re-annotated.
4. Consolidate in `{topic}_annotated.csv` and load with
   `python scripts/build_gold_annotation.py --load`.

## Practical rules
- Decide on the PREVAILING stance of the document, not on single sentences.
- In doubt between a side and NEUTRAL with weak arguments → NEUTRAL.
- Do not use external knowledge about truthfulness: what counts is the position expressed.
