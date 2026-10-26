"""
Echo-Search | Dashboard "nudge etico" (Streamlit)
=================================================
Avvio: `streamlit run dashboard.py`

Consumatore puro di `EchoSearch`: non contiene logica di ricerca, solo
presentazione. Le tre posizioni si mostrano AFFIANCATE, e il filtro e' morbido:
scegliere una fazione ne allarga la colonna senza far sparire le altre. Un
filtro rigido, in un sistema nato contro le echo chamber, ne ricostruirebbe una
al primo clic dell'utente.

Le query canoniche sono precalcolate (vedi `scripts/precompute_demo.py`): la
dimostrazione e' istantanea e non richiede GPU, mentre dal vivo il motore LLM
costerebbe minuti per ecosistema.
"""

import os
# GPU da 6 GB: riduce la frammentazione della VRAM (evita OOM col fact-check).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from src.app import EchoSearch  # noqa: E402
from src.demo_cache import load_cached  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.gui.render import DASHBOARD_CSS, render_block  # noqa: E402

TOPICS = load_config("configs/topics.yaml")["topics"]
LABELS = ["FAVOR", "NEUTRAL", "AGAINST"]
ECO_LABEL = {"news": "Media (news)", "social": "Social (tweet)"}


@st.cache_resource(max_entries=1, show_spinner="Carico i modelli del tema...")
def get_app(topic: str) -> EchoSearch:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # libera la VRAM del tema precedente (GPU 6 GB)
    app = EchoSearch(topic=topic)
    app.load()
    return app


@st.cache_data(show_spinner="Interrogo la pipeline...")
def run_search(topic: str, query: str, fact_check: bool, summarize: bool, fc_engine: str) -> dict:
    # Per le query canoniche esiste una versione precalcolata COMPLETA (verdetti +
    # riassunti) per ciascun motore: si carica quella del motore scelto e la GUI ne
    # mostra il sottoinsieme richiesto dai toggle. Il motore fa parte della chiave,
    # altrimenti scegliere NLI mostrerebbe i verdetti dell'LLM.
    cached = load_cached(topic, query, True, True, fc_engine)
    if cached is not None:
        cached["_cached"] = True
        if not summarize:                     # rispetta il toggle "Riassunti"
            for blk in cached["ecosystems"].values():
                blk.pop("summaries", None)
        return cached                         # i verdetti li filtra il render, con show_fact
    app = get_app(topic)                       # query fuori dalle canoniche -> calcolo dal vivo
    return app.search(query, target=TOPICS[topic]["target"],
                      fact_check=fact_check, summarize=summarize, fact_check_engine=fc_engine)


st.set_page_config(page_title="Echo-Search", layout="wide")
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.header("Echo-Search")
    topic = st.selectbox("Tema", list(TOPICS), format_func=lambda t: TOPICS[t]["label"])
    suggestions = TOPICS[topic]["queries"]
    query = st.text_input("Query", value=suggestions[0])
    st.caption("Suggerimenti: " + " · ".join(suggestions))
    fact_check = st.checkbox("Fact-check", value=False, help="Verifica le claim contro Wikipedia (locale) [lento]")
    fc_engine = "nli"
    if fact_check:
        fc_engine = st.radio(
            "Motore fact-check",
            ["nli", "llm"],
            format_func=lambda e: {"nli": "NLI · Wikipedia (veloce)",
                                   "llm": "LLM qwen2.5 · RAG (lento, ~min)"}[e],
            help="Stesse evidenze, verdetto diverso. NLI: DeBERTa addestrato su FEVER, "
                 "decine di ms. LLM: qwen locale che ragiona sulle evidenze, più accurato "
                 "ma secondi per affermazione. Sulle query canoniche entrambi sono "
                 "precalcolati, quindi il confronto è immediato.",
        )
    summarize = st.checkbox("Riassunti", value=False, help="Sintesi per prospettiva [lento]")
    focus = st.radio("Approfondisci", ["—"] + LABELS, horizontal=True)

st.title("Nudge etico — prospettive a confronto")

if not query.strip():
    st.info("Scrivi una query nella barra laterale.")
    st.stop()

resp = run_search(topic, query, fact_check, summarize, fc_engine)
if resp.get("_cached"):
    st.caption("⚡ Risultato precalcolato (demo) — caricato da cache, nessun modello in esecuzione.")
ecosystems = resp["ecosystems"]

tabs = st.tabs([ECO_LABEL.get(e, e) for e in ecosystems])
for tab, (eco, block) in zip(tabs, ecosystems.items()):
    with tab:
        html = render_block(block, None if focus == "—" else focus, fact_check)
        st.markdown(html, unsafe_allow_html=True)
