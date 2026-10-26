# Linee guida di annotazione — Qrels (rilevanza per il retrieval)

Si annota la **rilevanza del documento rispetto alla query**, NON la sua stance, NON la sua
veridicità, NON la qualità della fonte. È il giudizio che alimenta NDCG@10, MAP e Recall@20.

## La regola che conta più di tutte

> **Rilevante ≠ d'accordo. Rilevante ≠ vero.**

Il corpus è fatto di temi divisivi, quindi la tentazione di penalizzare i documenti
sbagliati o sgradevoli è forte. Va resistita: un articolo negazionista sul clima, se
risponde alla query `is climate change caused by humans`, è **pienamente rilevante (2)**.

Se confondiamo rilevanza e posizione, misuriamo il motore di ricerca sulla sua capacità di
nascondere le opinioni scomode — cioè l'esatto contrario di ciò che Echo-Search dichiara di
fare. La veridicità è compito del fact-checker, la posizione è compito dello stance
detector: qui non c'entrano né l'una né l'altra.

## La scala

| Voto | Significato | Criterio operativo |
|---|---|---|
| **2** | Pienamente rilevante | Risponde alla query, o ne tratta il nucleo in modo sostanziale. Se un utente cercasse quella query, sarebbe contento di trovarlo in prima pagina. |
| **1** | Marginalmente rilevante | Tratta il tema ma non risponde alla query: la tocca di sfuggita, la sfiora da un'angolazione diversa, o contiene solo un frammento utile. |
| **0** | Non rilevante | Non risponde alla query. Include sia il fuori tema puro sia gli articoli che parlano del tema generale ma non hanno alcun rapporto con ciò che la query chiede. |

**Distinguere 2 da 1** è la decisione che fa la differenza per l'NDCG: chiedersi *"questo
documento risponde alla domanda, o parla soltanto dell'argomento della domanda?"*

Esempi sulle query reali:

- `risks of nuclear energy` — un articolo sull'incidente di Fukushima e le sue conseguenze
  sanitarie → **2**. Un articolo sui costi di costruzione di una centrale, che cita i rischi
  in una riga → **1**. Un comunicato su un accordo commerciale per l'uranio → **0**.
- `do vaccines cause autism` — un articolo sullo studio di Wakefield e la sua ritrattazione
  → **2**. Un articolo sull'esitazione vaccinale che cita l'autismo fra le paure diffuse →
  **1**. Un articolo su una campagna antipolio in Ghana → **0**.
- `economic cost of climate policy` — un'analisi dell'impatto economico di una carbon tax →
  **2**. Un articolo su un vertice sul clima che accenna ai costi → **1**. Un articolo sugli
  effetti fisici del riscaldamento → **0**.

## Il confine 0/1/2 — specifica operativa (v2)

La prima versione di queste linee guida definiva 0 e 1 in modo sovrapposto ("parla del tema
generale ma non ha rapporto con la query" per lo 0, "tratta il tema ma non risponde" per
l'1): la stessa frase detta due volte. Definito così, il confine è un'intenzione e non un
criterio, e l'ambiguità morde dove la query è astratta e il corpus tematicamente saturo —
cioè sulle query climatiche. Questa è la specifica che lo risolve, formulata come **test
verificabile sul testo**.

| Voto | Test operativo |
|---|---|
| **2** | Sai **indicare la frase** del documento che costituisce la risposta alla query. |
| **1** | Non c'è una frase-risposta, ma il documento fornisce contesto utile o tocca la domanda di lato. Lo apriresti come terzo o quarto risultato e ne uscirebbe qualcosa. |
| **0** | Il legame con la query è **solo tematico**: il documento parla del tema (nucleare / clima / vaccini) ma non della domanda. Include il fuori tema puro. |

### Corollari, dedotti da casi reali del corpus

- **Se la query chiede un costo, serve un costo.** Una cifra, una stima, o un'affermazione
  esplicita su chi paga e quanto. Una disputa su trasferimenti di fondi fra enti pubblici
  non è il costo di una politica. E il costo *dei danni* (social cost of carbon) non è il
  costo *della politica*: hanno segno opposto.
- **Se la query chiede una relazione fra due cose** (clima↔eventi estremi, nucleare↔clima),
  il testo deve **asserire la relazione**. Non basta che entrambi i termini compaiano.
- **Se la query chiede un rischio o un effetto,** il documento deve parlare di quel
  rischio, non del tema in cui il rischio esiste. Il fallimento finanziario di un progetto
  non è un "rischio dell'energia nucleare" a meno che il testo non lo inquadri come tale.
- **La fonte non conta** (già detto sopra, vale la pena ripeterlo qui): un documento
  negazionista che risponde alla domanda è 2.
- **Il duplicato sindacato prende lo stesso voto della sua copia.** Il corpus contiene
  molte riprese della stessa storia d'agenzia: sono un'unica decisione editoriale.

## Casi limite

- **L'estratto non basta per decidere.** Il foglio mostra titolo e apertura dell'articolo
  (~1200 caratteri). Se non basta, recupera il testo integrale dal `doc_id` in
  `data/corpora/{tema}.jsonl`. In mancanza di meglio, giudica su ciò che vedi: è la stessa
  informazione che hanno i modelli sui primi chunk.
- **Fuori tema per omonimia.** Il filtro per keyword dei corpora lascia passare del rumore
  (il caso noto: il centro di ricerca *VACCINE* di Purdue, acronimo che non ha nulla a che
  fare con i vaccini). Voto **0**, e segnalalo nella colonna `nota`: la quota di questi casi
  misura il rumore residuo del corpus.
- **Lo stesso documento sotto query diverse.** Normale e voluto: la rilevanza è per coppia
  *(query, documento)*, non per documento. Lo stesso articolo può valere 2 per una query e 0
  per un'altra. Giudicalo ogni volta rispetto alla query di quella riga.
- **Articolo pertinente ma vecchio, breve o mal scritto.** La qualità editoriale non entra
  nel giudizio. Se risponde alla query, è rilevante.
- **In dubbio fra 1 e 0** → 0. **In dubbio fra 2 e 1** → 1. Il default conservativo tiene
  bassa la deriva fra annotatori.

## Protocollo (per Cohen's kappa)

1. Ognuno compila la colonna `relevance` del **proprio** foglio:
   `data/gold/qrels_{tema}_davide.csv` e `data/gold/qrels_{tema}_giuseppe.csv`.
   I due fogli sono identici per contenuto e ordine: **380 coppie in totale**
   (137 nucleare, 109 clima, 134 vaccini).
2. **Doppia annotazione completa e indipendente.** Ognuno giudica tutto, senza consultare
   l'altro e senza confrontare i fogli durante il lavoro: il kappa misura l'accordo, e se ci
   si allinea strada facendo il valore è gonfiato e non significa più nulla.
3. La colonna `nota` è libera: usala per i casi dubbi, serve in fase di adjudicazione.
4. A fine annotazione si calcola **Cohen's kappa su tutte le 380 coppie**. Riferimento:
   κ ≥ 0,6 sostanziale, κ ≥ 0,8 quasi perfetto. Se κ < 0,5 conviene rivedere queste linee
   guida e riannotare, invece di procedere con qrels inaffidabili.
5. **Adjudicazione**: si esaminano insieme solo le righe in disaccordo e si decide il voto
   finale. Il consolidato diventa `qrels_{tema}.csv`, che è l'artefatto usato dalle metriche.
6. Riportare nella relazione: κ per tema e complessivo, numero di divergenze, e la
   distribuzione dei voti (quanti 0/1/2).

## Esito e note sui dati

Annotazione completata sulle 380 coppie da entrambi gli annotatori. κ di Cohen a pesi
quadratici **0,867** complessivo (nucleare 0,904, clima 0,814, vaccini 0,856), non pesato
0,675, accordo grezzo 0,797; 77 divergenze, di cui solo **2 a due gradini**. Il calcolo è
riproducibile con `uv run --no-sync python scripts/eval_agreement.py`, che salva il
dettaglio (matrici di confusione e marginali) in `data/gold/agreement_report.json`.

Una nota di cautela sui dati: la query `nuclear waste disposal` ha 28 rilevanti su 31, e
`are vaccines safe` 24 su 28 (identiche per entrambi gli annotatori). Su queste due il pool
è quasi saturo di rilevanti, quindi le metriche discrimineranno poco fra i sistemi: va detto
in relazione anziché lasciato dedurre.

## Note pratiche

- L'ordine delle righe è **randomizzato entro ogni query** e le colonne `rank`/`score` non
  compaiono: è deliberato. Vedere che un documento è al primo posto predispone a giudicarlo
  rilevante (*position bias*), e falserebbe proprio la metrica che stiamo costruendo.
- Il pool è l'**unione** dei documenti recuperati da cinque run diversi (BM25, denso, RRF,
  fusione, pool del reranker). **143 coppie su 380 — il 38% — sono state recuperate da un
  solo run**: sono esattamente quelle che un pooling dal solo run di fusione avrebbe perso,
  facendole conteggiare come non rilevanti e falsando il confronto a sfavore delle baseline.
- I file sono in UTF-8 con BOM: Excel su Windows li apre con gli accenti corretti. Se salvi
  da Excel, mantieni il formato CSV UTF-8.
- Tempo stimato: **~1h40–2h a testa**, a 15–20 secondi per giudizio. Conviene farlo in due
  sessioni per tema e non tutto d'un fiato: la deriva di giudizio cresce con la stanchezza.
- Ordine consigliato: un tema per volta, dall'inizio alla fine, senza saltare fra i fogli.
  Le query dello stesso tema condividono il contesto e si va più veloci.
