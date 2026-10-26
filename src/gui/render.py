"""
Echo-Search | GUI — rendering HTML della dashboard.
Look "prospettive in tensione": type Space Grotesk + Inter, palette divergente
per stance, barra di Stance Diversity (signature), card espandibili (<details>).
Tutto HTML/CSS iniettato in Streamlit; funzioni testabili (input dati -> stringa).
"""

from .format import column_widths, truncate, verdict_badge

LABELS = ["FAVOR", "NEUTRAL", "AGAINST"]
STANCE_COLOR = {"FAVOR": "#2563EB", "NEUTRAL": "#64748B", "AGAINST": "#E8590C"}
_BADGE_CLASS = {"success": "ok", "danger": "no", "neutral": "nei"}

DASHBOARD_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&display=swap');
.es-wrap { font-family:'Inter',system-ui,sans-serif; color:#1A2330; }
.es-meter { display:flex; align-items:center; gap:16px; margin:2px 0 24px; }
.es-meter__bar { display:flex; height:15px; flex:1; border-radius:8px; overflow:hidden; box-shadow:inset 0 0 0 1px rgba(20,30,50,.08); }
.es-meter__seg { transition:width .45s cubic-bezier(.22,1,.36,1); }
.es-meter__score { font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; line-height:1; letter-spacing:-.01em; }
.es-meter__cap { font-size:11px; color:#7A8598; text-transform:uppercase; letter-spacing:.1em; margin-top:3px; }
.es-cols { display:flex; gap:16px; align-items:flex-start; }
.es-col { flex-basis:0; min-width:0; transition:flex-grow .4s cubic-bezier(.22,1,.36,1); }
.es-col__head { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:13px; letter-spacing:.03em; color:#fff; padding:9px 13px; border-radius:12px 12px 0 0; display:flex; justify-content:space-between; }
.es-col__count { opacity:.8; font-weight:500; }
.es-col__body { background:#fff; border:1px solid #E6E9EF; border-top:none; border-radius:0 0 14px 14px; padding:13px; }
.es-summary { font-size:12.5px; line-height:1.55; color:#3A4657; background:#F4F6F9; border-radius:9px; padding:11px 13px; margin-bottom:13px; }
.es-summary__k { font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:10px; text-transform:uppercase; letter-spacing:.11em; color:#8A94A6; display:block; margin-bottom:5px; }
.es-card { background:#fff; border:1px solid #E6E9EF; border-radius:11px; margin-bottom:10px; overflow:hidden; transition:box-shadow .2s ease,border-color .2s ease; }
.es-card:hover { border-color:#C5CCD8; box-shadow:0 2px 10px rgba(20,30,50,.06); }
.es-card[open] { box-shadow:0 8px 26px rgba(20,30,50,.11); border-color:#C5CCD8; }
.es-card>summary { list-style:none; cursor:pointer; padding:12px 13px; }
.es-card>summary::-webkit-details-marker { display:none; }
.es-card__txt { font-size:13px; line-height:1.55; color:#2A3444; }
.es-card[open] .es-clamp { display:none; }
.es-full { display:none; }
.es-card[open] .es-full { display:block; }
.es-meta { display:flex; align-items:center; justify-content:space-between; margin-top:9px; gap:8px; }
.es-conf { font-size:11px; color:#939CAD; font-variant-numeric:tabular-nums; }
.es-right { display:flex; align-items:center; gap:8px; }
.es-badge { font-size:10.5px; font-weight:500; padding:2px 9px; border-radius:20px; white-space:nowrap; }
.es-badge.ok { background:#E4F1E6; color:#1E6B3A; }
.es-badge.no { background:#FBE6DB; color:#B23A10; }
.es-badge.nei { background:#EDEFF3; color:#64748B; }
.es-chev { font-size:11px; color:#B4BCC9; transition:transform .25s ease; }
.es-card[open] .es-chev { transform:rotate(180deg); }
.es-fact { font-size:12px; line-height:1.55; color:#48546A; border-top:1px dashed #E1E5EC; margin:2px 13px 12px; padding-top:9px; }
.es-fact__claim { color:#2A3444; }
.es-fact a { color:#2563EB; text-decoration:none; border-bottom:1px solid rgba(37,99,235,.3); }
.es-hint { font-size:10.5px; color:#AAB2BF; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.015em; }
[data-testid="stSidebar"] { border-right:1px solid #E6E9EF; }
</style>"""


def _esc(text: str) -> str:
    """Neutralizza l'HTML: i testi vengono da articoli e tweet, non sono nostri."""
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_diversity_meter(block: dict) -> str:
    """Barra segmentata proporzionale (pool per stance) + punteggio: il signature element."""
    pool = block.get("pool_distribution", {})
    total = sum(pool.get(k, 0) for k in LABELS) or 1
    segs = "".join(
        f'<div class="es-meter__seg" style="width:{100*pool.get(k,0)/total:.1f}%;'
        f'background:{STANCE_COLOR[k]}"></div>'
        for k in LABELS
    )
    score = block.get("diversity", {}).get("diversity_score", 0)
    return (
        '<div class="es-meter">'
        f'<div><div class="es-meter__score">{score:.2f}</div>'
        '<div class="es-meter__cap">Stance diversity</div></div>'
        f'<div class="es-meter__bar">{segs}</div>'
        '</div>'
    )


def _render_fact(doc: dict) -> str:
    checks = doc.get("fact_checks") or []
    if not checks:
        return ""
    fc = checks[0]
    parts = [f'<div class="es-fact__claim"><b>Claim.</b> {_esc(fc.get("claim", ""))}</div>']
    parts.append(f'Verdetto <b>{_esc(fc.get("verdict", ""))}</b>')
    if fc.get("reasoning"):
        parts.append(_esc(fc["reasoning"]))
    for s in (fc.get("sources") or [])[:3]:
        title = (s.get("title") or s.get("url", ""))[:70]
        parts.append(f'<a href="{_esc(s.get("url", ""))}" target="_blank">{_esc(title)}</a>')
    return '<div class="es-fact">' + "<br>".join(parts) + "</div>"


def render_card(doc: dict, show_fact: bool) -> str:
    """Doc card espandibile: anteprima -> click -> testo pieno (+ fact-check)."""
    full = _esc(doc.get("text", ""))
    preview = _esc(truncate(doc.get("text", ""), 150))
    badge = verdict_badge(doc.get("factuality") or {}) if show_fact else None
    badge_html = (
        f'<span class="es-badge {_BADGE_CLASS.get(badge["tone"], "nei")}">{badge["icon"]} {badge["label"]}</span>'
        if badge and doc.get("fact_checks") else ""
    )
    conf = doc.get("stance_confidence", 0)
    fact = _render_fact(doc) if show_fact else ""
    return (
        '<details class="es-card"><summary>'
        f'<div class="es-card__txt es-clamp">{preview}</div>'
        f'<div class="es-card__txt es-full">{full}</div>'
        f'<div class="es-meta"><span class="es-conf">conf {conf:.2f}</span>'
        f'<span class="es-right">{badge_html}<span class="es-chev">▾</span></span></div>'
        '</summary>'
        f'{fact}'
        '</details>'
    )


def render_block(block: dict, focus, show_fact: bool) -> str:
    """HTML completo di un ecosistema: barra diversity + 3 colonne con card."""
    widths = column_widths(focus)
    summaries = block.get("summaries") or {}
    cols = []
    for label, grow in zip(LABELS, widths):
        docs = block.get("columns", {}).get(label, [])
        summ = summaries.get(label, {}).get("summary")
        summ_html = (
            f'<div class="es-summary"><span class="es-summary__k">Sintesi</span>{_esc(summ)}</div>'
            if summ else ""
        )
        cards = "".join(render_card(d, show_fact) for d in docs)
        cols.append(
            f'<div class="es-col" style="flex-grow:{grow}">'
            f'<div class="es-col__head" style="background:{STANCE_COLOR[label]}">'
            f'<span>{label}</span><span class="es-col__count">{len(docs)}</span></div>'
            f'<div class="es-col__body">{summ_html}{cards}</div></div>'
        )
    return (
        '<div class="es-wrap">'
        + render_diversity_meter(block)
        + '<div class="es-cols">' + "".join(cols) + '</div>'
        + '</div>'
    )
