# Linee guida di annotazione — Stance (Fase 3, task 3.1)

Si annota la **stance del documento rispetto al `target` del tema**
(nuclear energy / climate change / vaccine safety), NON il sentiment né la veridicità.

## Il target di ogni tema (COSA valutare)

La stance è verso una **proposizione implicita**, non il tema in astratto. Per ogni tema, giudica se il documento sostiene o si oppone a questa affermazione:

| Tema | Proposizione-target | FAVOR = | AGAINST = |
|---|---|---|---|
| **nuclear energy** | l'energia nucleare va usata/espansa come fonte | pro nucleare (pulito, sicuro, costruire) | contro (pericoloso, dismettere) |
| **climate change** | il climate change è una minaccia reale e urgente | riconosce realtà/urgenza, pro azione | scettico, nega, minimizza |
| **vaccine safety** | i vaccini sono sicuri ed efficaci | pro vaccini (sicuri, raccomandati) | anti / esitante (pericolosi, effetti nascosti) |

**Esempi:**
- Nuclear — FAVOR: "È la fonte pulita e affidabile che ci serve." · AGAINST: "Troppo pericolosa, va dismessa."
- Climate — FAVOR: "Serve agire subito, è un'emergenza." · AGAINST: "L'allarmismo è esagerato, modelli inaffidabili."
- Vaccines — FAVOR: "Sicuri, prevengono milioni di morti." · AGAINST: "Causano effetti gravi nascosti."

**Climate change — casi limite:** "favor/against" NON significa essere a favore del riscaldamento (si segue la convenzione SemEval-2016 *"Climate Change is a Real Concern"*, che è nei dati di training). L'asse è **realtà + urgenza**: chi nega/minimizza → AGAINST; chi riconosce il problema e sostiene l'azione → FAVOR. L'opposizione a **una singola politica** (es. una specifica carbon tax) **senza** negare il problema di fondo → **NEUTRAL**.

## Le classi
- **FAVOR** — il documento sostiene/difende il target (argomenti a favore prevalenti).
- **AGAINST** — il documento si oppone al target (argomenti contrari prevalenti).
- **NEUTRAL** — il documento **parla del tema** ma nessuna posizione prevale:
  - *neutrale*: cronaca fattuale senza presa di posizione;
  - *bilanciato*: presenta pro e contro con pari peso.
- **OFF_TOPIC** — il documento **non riguarda il tema**: la keyword compare per omonimia,
  acronimo o citazione incidentale. Esempio reale dal corpus: un articolo sul centro di
  ricerca *VACCINE* di Purdue (acronimo di *Visual Analytics for Command, Control and
  Interoperability Environments*), che parla di social media analytics per la protezione
  civile — nessun legame con i vaccini. **Non è una stance**: è rumore del filtro del
  corpus, e va marcato a parte, **mai** come NEUTRAL.

> **Perché OFF_TOPIC separato da NEUTRAL:** tiene pulita la classe NEUTRAL (che significa
> "parla del tema senza schierarsi"), non distrugge informazione durante l'annotazione, e
> misura quanto rumore contiene il corpus. In fase di costruzione del gold si decide se
> **escluderli** (default: il gold misura la stance su documenti pertinenti) o **mapparli a
> NEUTRAL** (misura il comportamento end-to-end); avendo la label distinta si possono
> riportare entrambe le varianti. Una quota di OFF_TOPIC > 10-15% segnala che il filtro
> per keyword dei corpora va irrobustito.

## Protocollo (per Cohen's kappa)
1. Ogni annotatore compila la colonna `label` del proprio `{tema}_to_annotate.csv`.
2. I `doc_id` in `{tema}_overlap_ids.json` sono annotati da ENTRAMBI, indipendentemente.
3. Sull'overlap si calcola Cohen's kappa (accordo). Target: κ ≥ 0.6 (sostanziale).
   Se κ è basso, si rivedono queste linee guida e si riannota l'overlap.
4. Consolidare in `{tema}_annotated.csv` e caricare con
   `python scripts/build_gold_annotation.py --load`.

## Regole pratiche
- Decidere sulla stance PREVALENTE del documento, non su singole frasi.
- In dubbio tra una fazione e NEUTRAL con argomenti deboli → NEUTRAL.
- Non usare conoscenza esterna sulla veridicità: conta la posizione espressa.
