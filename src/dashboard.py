"""
dashboard.py  -  Live project dashboard (Components A, B, C + graph modelling)
-------------------------------------------------------------------------------
Run from the project folder:
    streamlit run src/dashboard.py

Reads the real project files:
    data/raw/immo_data.csv, data/cleaned/immo_data_clean.csv
    outputs/fair/        F-UJI JSON exports, metadata_record.jsonld, data_dictionary.csv
    outputs/cleaning/    audit_log.csv
    outputs/graph/       district_metrics.csv, graph_schema.png, district_graph.png

The "Ask the data" tab needs an OpenAI API key
in .streamlit/secrets.toml (or the OPENAI_API_KEY environment variable).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "immo_data.csv"
CLEAN = ROOT / "data" / "cleaned" / "immo_data_clean.csv"
FAIR = ROOT / "outputs" / "fair"
CLEANING = ROOT / "outputs" / "cleaning"
GRAPH = ROOT / "outputs" / "graph"
PROFILING = ROOT / "outputs" / "profiling"
WAVES = ["Sep18", "May19", "Oct19", "Feb20"]

st.set_page_config(page_title="Fair Rent Data Quality Dashboard", page_icon="🏠", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Business look (design only: colours, fonts, cards; the content is unchanged)
# ---------------------------------------------------------------------------
BLUE, NAVY = "#2a78d6", "#0f2747"
INK, MUTED, GRID = "#1d2433", "#5b6475", "#e9edf2"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# One Plotly template for every chart: white, light grid, left-aligned titles, brand colours
pio.templates["business"] = go.layout.Template(layout=go.Layout(
    font={"family": "Segoe UI, Inter, Arial, sans-serif", "size": 12, "color": INK},
    title={"font": {"size": 15, "color": INK}, "x": 0, "xanchor": "left"},
    paper_bgcolor="white", plot_bgcolor="white", colorway=CATEGORICAL,
    xaxis={"gridcolor": GRID, "linecolor": GRID, "zeroline": False},
    yaxis={"gridcolor": GRID, "linecolor": GRID, "zeroline": False},
    legend={"title": {"text": ""}},
    margin={"l": 10, "r": 20, "t": 50, "b": 10},
    hoverlabel={"bgcolor": "white", "bordercolor": GRID, "font": {"color": INK}},
))
pio.templates.default = "plotly_white+business"
px.defaults.template = "plotly_white+business"
px.defaults.color_discrete_sequence = CATEGORICAL
BLUE_SCALE = ["#dceafb", "#9cc3f0", "#4f94e0", "#2a78d6", "#1a4f99", "#0f2747"]
px.defaults.color_continuous_scale = BLUE_SCALE

# Every st.plotly_chart uses our template (not Streamlit's theme) and hides the Plotly logo
def polish(fig):
    """Power BI-like finish for any Plotly figure: rounded bars, value labels, clean legend, fixed height."""
    fig.update_layout(
        height=fig.layout.height or 420, bargap=0.28, barcornerradius=5, hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1,
                "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11, "color": MUTED}},
        title={"font": {"size": 15, "color": INK}, "x": 0.01, "xanchor": "left", "y": 0.97},
        margin={"l": 10, "r": 20, "t": 70, "b": 10},
    )
    # Business-friendly labels: "Baden_Württemberg" -> "Baden Württemberg" in legends and category axes
    for tr in fig.data:
        if isinstance(tr.name, str) and "_" in tr.name:
            tr.name = tr.name.replace("_", " ")
        for attr in ("x", "y"):
            vals = getattr(tr, attr, None)
            if vals is not None and len(vals) and all(isinstance(v, str) for v in vals):
                setattr(tr, attr, [v.replace("_", " ") for v in vals])
    # Many series: legend in a column on the right, so it never covers the chart title
    named = [t for t in fig.data if t.name and t.showlegend is not False]
    if len(named) > 3:
        fig.update_layout(legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02,
                                  "title": {"text": ""}},
                          margin={"r": 200})
    else:
        fig.update_layout(legend_title_text="")
    for tr in fig.data:
        if tr.type == "bar":
            tr.update(marker_line_width=0, textfont={"size": 11, "color": INK}, cliponaxis=False)
            if tr.text is not None:
                tr.update(textposition="outside")
        elif tr.type == "scatter" and tr.mode and "markers" in tr.mode and tr.marker.color != "rgba(0,0,0,0)":
            tr.update(marker_line={"width": 0.6, "color": "white"})
            if tr.mode and "lines" in tr.mode:
                tr.update(line={"width": 3}, marker_size=9)
        elif tr.type == "box":
            tr.update(marker_size=4, line_width=1.5)
    # A single bar series gets a light-to-dark blue gradient by value (like Power BI conditional colours)
    bars = [t for t in fig.data if t.type == "bar"]
    if len(bars) == 1 and len(fig.data) == 1 and not fig.layout.coloraxis.colorscale:
        t = bars[0]
        vals = t.x if t.orientation == "h" else t.y
        try:
            vals = [float(v) for v in vals]
            value_axis = fig.layout.xaxis if t.orientation == "h" else fig.layout.yaxis
            if value_axis.type == "log":                       # shade by order of magnitude on log charts
                vals = [float(np.log10(v)) if v > 0 else 0.0 for v in vals]
            t.update(marker={"color": vals, "colorscale": BLUE_SCALE[1:], "showscale": False})
        except (TypeError, ValueError):
            pass
    # Log axes: only label powers of ten (no 2 / 5 minor labels)
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        if ax.type == "log":
            ax.update(dtick=1)
    # Readable axis titles: column names -> business labels, "rows_after" -> "Rows after"
    nice = {"baseRent": "Base rent (€ / month)", "livingSpace": "Living space (m²)", "totalRent": "Total rent (€ / month)",
            "rent_per_m2": "Rent per m² (€)", "serviceCharge": "Service charge (€ / month)", "noRooms": "Rooms",
            "regio1": "Federal state", "regio2": "District", "typeOfFlat": "Flat type", "yearConstructed": "Year built"}
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        if ax.title and ax.title.text in nice:
            ax.title.text = nice[ax.title.text]
    for ax in (fig.layout.xaxis, fig.layout.yaxis):
        if ax.title and ax.title.text and "_" in ax.title.text or (ax.title and ax.title.text == "label"):
            txt = "Step" if ax.title.text == "label" else ax.title.text.replace("_", " ")
            ax.title.text = txt[0].upper() + txt[1:]
    return fig


DeltaGenerator._polish = staticmethod(polish)                      # always the newest version of polish()
if not hasattr(DeltaGenerator, "_original_plotly_chart"):          # patch once, not on every rerun
    DeltaGenerator._original_plotly_chart = DeltaGenerator.plotly_chart

    def _business_plotly_chart(self, fig, *args, **kwargs):
        DeltaGenerator._polish(fig)
        kwargs.setdefault("theme", None)
        kwargs.setdefault("config", {"displaylogo": False})
        return DeltaGenerator._original_plotly_chart(self, fig, *args, **kwargs)
    DeltaGenerator.plotly_chart = _business_plotly_chart
st.plotly_chart = lambda fig, *args, **kwargs: DeltaGenerator.plotly_chart(st._main, fig, *args, **kwargs)

st.markdown(f"""
<style>
.stApp {{ background: #f4f6f9; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.banner {{ background: linear-gradient(90deg, {NAVY} 0%, #1f4f8a 100%); color: white;
          padding: 22px 28px; border-radius: 10px; margin-bottom: 14px; }}
.banner h1 {{ color: white; font-size: 1.7rem; margin: 0; padding: 0; font-weight: 700; }}
.banner p {{ color: #cdd9ea; margin: 6px 0 0 0; font-size: 0.95rem; }}
/* KPI cards */
div[data-testid="stMetric"] {{ background: white; border: 1px solid {GRID}; border-left: 4px solid {BLUE};
    border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
div[data-testid="stMetric"] label p {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: .04em; }}
div[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; font-size: 1.6rem; }}
div[data-testid="stColumn"]:nth-child(5n+2) div[data-testid="stMetric"] {{ border-left-color: #eb6834; }}
div[data-testid="stColumn"]:nth-child(5n+3) div[data-testid="stMetric"] {{ border-left-color: #1baf7a; }}
div[data-testid="stColumn"]:nth-child(5n+4) div[data-testid="stMetric"] {{ border-left-color: #eda100; }}
div[data-testid="stColumn"]:nth-child(5n+5) div[data-testid="stMetric"] {{ border-left-color: #6250d6; }}
/* charts, tables and images sit on white cards */
div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"], div[data-testid="stImage"] {{
    background: white; border: 1px solid {GRID}; border-radius: 10px; padding: 8px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: white; border-radius: 10px; border-color: {GRID}; }}
/* tabs as pills */
div[role="tablist"] {{ gap: 6px; background: white; padding: 6px; border-radius: 10px; border: 1px solid {GRID}; }}
[role="tab"] {{ padding: 8px 18px !important; border-radius: 7px; border-bottom: none !important; }}
[role="tab"] p {{ font-weight: 600; color: {MUTED}; font-size: 0.92rem; }}
[role="tab"][aria-selected="true"] {{ background: {BLUE}; }}
[role="tab"][aria-selected="true"] p {{ color: white; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display: none; }}
/* headings and sidebar */
h2, h3 {{ color: {INK} !important; font-weight: 700 !important; }}
h3 {{ font-size: 1.1rem !important; border-left: 4px solid {BLUE}; padding-left: 10px !important; margin-top: 18px !important; }}
section[data-testid="stSidebar"] {{ background: white; border-right: 1px solid {GRID}; }}
section[data-testid="stSidebar"] h2 {{ font-size: 1.1rem !important; }}
/* business insight cards and callouts */
.insight {{ background: white; border: 1px solid {GRID}; border-top: 4px solid {BLUE}; border-radius: 10px;
    padding: 16px 18px; height: 100%; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.insight .tag {{ color: {MUTED}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
.insight .big {{ color: {INK}; font-size: 1.5rem; font-weight: 700; margin: 4px 0 2px 0; }}
.insight .txt {{ color: #475467; font-size: 0.88rem; line-height: 1.4; }}
.insight.c2 {{ border-top-color: #eb6834; }} .insight.c3 {{ border-top-color: #1baf7a; }}
.insight.c4 {{ border-top-color: #eda100; }}
.callout {{ background: #eef4fc; border-left: 4px solid {BLUE}; border-radius: 8px; padding: 12px 16px;
    color: {INK}; font-size: 0.93rem; margin: 6px 0 14px 0; }}
.callout b {{ color: {NAVY}; }}
.section-note {{ color: {MUTED}; font-size: 0.9rem; margin: -4px 0 10px 0; }}
div[data-testid="stChatMessage"] {{ background: white; border: 1px solid {GRID}; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders (computed once, then instant)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Profiling the raw dataset (first load only)…")
def raw_profile():
    df = pd.read_csv(RAW, dtype={"geo_plz": str}, low_memory=False)
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    hidden = pd.Series(0, index=df.columns)
    for c in text_cols:
        hidden[c] = int(df[c].astype(str).str.strip().str.lower().eq("no_information").sum())
    comp = pd.DataFrame({"empty": df.isna().sum(), "hidden": hidden})
    comp["effective_%"] = (comp.sum(axis=1) / len(df) * 100).round(1)

    calc = df["baseRent"] + df["serviceCharge"].fillna(0) + df["heatingCosts"].fillna(0)
    consistency = {
        "totalRent ≠ its parts": int(((df["totalRent"] - calc).abs() > 1).sum()),
        "floor above building height": int((df["floor"] > df["numberOfFloors"]).sum()),
        "regio1 ≠ geo_bln": int((df["regio1"] != df["geo_bln"]).sum()),
        "refurbished before built": int((df["lastRefurbish"] < df["yearConstructed"]).sum()),
    }
    key = ["geo_plz", "livingSpace", "baseRent", "noRooms", "floor", "streetPlain", "yearConstructed"]
    dups = df.duplicated(subset=key, keep=False)
    waves = pd.crosstab(df["date"], dups).reindex(WAVES)
    waves.columns = ["Unique", "Near-duplicate"]
    info = {"rows": len(df), "cols": df.shape[1], "complete_rows": int(len(df.dropna())),
            "plz_leading_zero": int(df["geo_plz"].str.startswith("0").sum())}
    numeric = df.select_dtypes("number").columns.tolist()
    return comp, consistency, waves, info, numeric


@st.cache_data(show_spinner="Loading rent and size columns…")
def raw_columns(cols):
    return pd.read_csv(RAW, usecols=list(cols))


@st.cache_data(show_spinner="Loading the cleaned dataset…")
def clean_data():
    df = pd.read_csv(CLEAN, dtype={"geo_plz": str}, low_memory=False)
    df["rent_per_m2"] = df["baseRent"] / df["livingSpace"]
    return df


@st.cache_resource(show_spinner="Laying out the district network…")
def network():
    G = nx.read_gexf(GRAPH / "district_graph.gexf")
    pos = nx.spring_layout(G, seed=42, weight="weight", k=0.35)
    return G, pos


@st.cache_data
def fuji(pattern):
    files = sorted(FAIR.glob(pattern))
    if not files:
        return None
    data = json.loads(files[0].read_text(encoding="utf-8"))
    rows = [{"metric": r["metric_identifier"].replace("FsF-", ""),
             "principle": r["metric_identifier"].split("-")[1][0],
             "earned": r["score"]["earned"], "total": r["score"]["total"],
             "maturity": r.get("maturity", 0)} for r in data["results"]]
    return pd.DataFrame(rows)


def stat_row(s_rent, s_space):
    return s_rent.mean(), s_rent.std(), s_rent.corr(s_space)


def insight_cards(cards):
    """Row of plain-language business takeaways: (tag, headline number, explanation)."""
    cols = st.columns(len(cards))
    for i, (col, (tag, big, txt)) in enumerate(zip(cols, cards)):
        col.markdown(f'<div class="insight c{i % 4 + 1}"><div class="tag">{tag}</div>'
                     f'<div class="big">{big}</div><div class="txt">{txt}</div></div>', unsafe_allow_html=True)


def callout(html):
    st.markdown(f'<div class="callout">{html}</div>', unsafe_allow_html=True)


def missing_file(path):
    st.warning(f"File not found: `{path.relative_to(ROOT)}` – run the matching script first.")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="banner"><h1>How Does Data Quality Affect Fair Rent Estimates?</h1>'
            '<p>268,850 German rental listings (ImmoScout24, 2018–2020) · FAIR assessment · data quality profiling · '
            'cleaning pipeline · graph analytics · AI Q&amp;A · '
            '<a href="https://doi.org/10.5281/zenodo.23494643" style="color:#9cc3f0">DOI 10.5281/zenodo.23494643</a></p></div>',
            unsafe_allow_html=True)

ISO_WAVES = ["2018-09", "2019-05", "2019-10", "2020-02"]
# Filter bar (Power BI-style slicers). Applies to the Explore and Graph tabs (cleaned data).
if CLEAN.exists():
    all_states = sorted(clean_data()["regio1"].dropna().unique())
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([3, 2.2, 2.2, 0.8], vertical_alignment="bottom")
        picked = f1.multiselect("Federal states", all_states, placeholder="All states", key="f_states",
                                format_func=lambda x: x.replace("_", " "))
        sel_states = picked or all_states
        sel_waves = f2.pills("Scrape waves", ISO_WAVES, selection_mode="multi", default=ISO_WAVES, key="f_waves")
        sel_waves = sel_waves or ISO_WAVES
        rent_range = f3.slider("Base rent (€ / month)", 50, 5000, (50, 5000), step=50, key="f_rent")
        if f4.button("Reset", icon=":material/restart_alt:", width="stretch"):
            for k in ["f_states", "f_waves", "f_rent"]:
                st.session_state.pop(k, None)
            st.rerun()
        st.caption("Filters apply to the Explore, Graph and Ask the data tabs (cleaned data).")
else:
    all_states, sel_states, sel_waves, rent_range = [], [], ISO_WAVES, (50, 5000)

tab0, tabA, tabB, tabC, tabG, tabE, tabAI = st.tabs(
    ["Overview", "A · FAIR & Metadata", "B · Data Quality", "C · Cleaning Pipeline", "Graph", "Explore",
     "Ask the data"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab0:
    c = st.columns(5)
    c[0].metric("Raw listings", "268,850", "49 variables", delta_color="off", delta_arrow="off")
    c[1].metric("Cleaned listings", "259,690", "−3.4 % (9,160 removed)", delta_color="off", delta_arrow="down")
    c[2].metric("FAIR score (F-UJI)", "84.6 %", "Kaggle and Zenodo", delta_color="off", delta_arrow="off")
    c[3].metric("Quality issues found", "12", "5 dimensions", delta_color="off", delta_arrow="off")
    c[4].metric("Rent–size correlation", "0.71", "was 0.00 before cleaning")

    st.subheader("Executive summary: what the data tells a business reader")
    st.markdown('<div class="section-note">Computed live from the cleaned data and the cleaning audit log.</div>',
                unsafe_allow_html=True)
    cards = []
    audit_file = CLEANING / "audit_log.csv"
    if audit_file.exists():
        al = pd.read_csv(audit_file)
        kept = al["rows_after"].iloc[-1] / al["rows_before"].iloc[0]
        cards.append(("Data you can trust", f"{kept:.1%} kept",
                      f"Cleaning removed {al['rows_before'].iloc[0] - al['rows_after'].iloc[-1]:,} broken or repeated "
                      f"listings; everything else was kept and every change is logged."))
        cards.append(("Errors distort averages", f"€{al['mean_baseRent_before'].iloc[0]:,.0f} → €{al['mean_baseRent_after'].iloc[-1]:,.0f}",
                      f"Average base rent before vs after cleaning. A few impossible rents had pushed the standard "
                      f"deviation to €{al['std_baseRent_before'].iloc[0]:,.0f}; after cleaning it is "
                      f"€{al['std_baseRent_after'].iloc[-1]:,.0f}."))
    if CLEAN.exists():
        cdx = clean_data()
        by_city = cdx.groupby("regio2")["rent_per_m2"].agg(["median", "size"]).query("size >= 100")
        top, low = by_city["median"].idxmax(), by_city["median"].idxmin()
        cards.append(("Location drives price", f"{by_city.loc[top, 'median'] / by_city.loc[low, 'median']:.1f}×",
                      f"{top.replace('_', ' ')} asks €{by_city.loc[top, 'median']:.2f}/m², {low.replace('_', ' ')} "
                      f"€{by_city.loc[low, 'median']:.2f}/m² (median, districts with ≥ 100 listings)."))
    cards.append(("Use with care", "2018 – 2020",
                  "Asking rents from one platform, before COVID-19 and the energy crisis: a historical benchmark, "
                  "not today's market."))
    insight_cards(cards)

    st.subheader("How the components connect")
    st.markdown(
        "**A – FAIR:** is the data findable, accessible, interoperable, reusable? → five gaps, metadata record, PIDs  \n"
        "**B – Quality:** what is wrong inside the data? → 12 issues across five dimensions  \n"
        "**C – Cleaning:** fix every issue from B in a reproducible, audited pipeline  \n"
        "**Graph:** model the cleaned data as a property graph and analyse the district network")

# ---------------------------------------------------------------------------
# A - FAIR
# ---------------------------------------------------------------------------
with tabA:
    kag, zen = fuji("*kaggle*.json"), fuji("*zenodo*.json")
    if kag is None or zen is None:
        st.warning("F-UJI JSON exports not found in `outputs/fair/` (file names must contain "
                   "'kaggle' and 'zenodo').")
    else:
        c = st.columns(2)
        c[0].metric("Kaggle", f"{kag.earned.sum() / kag.total.sum():.1%}", f"{kag.earned.sum():g}/{kag.total.sum():g} points", delta_color="off")
        c[1].metric("Zenodo", f"{zen.earned.sum() / zen.total.sum():.1%}", f"{zen.earned.sum():g}/{zen.total.sum():g} points", delta_color="off")
        by = pd.concat([
            kag.groupby("principle")[["earned", "total"]].sum().assign(platform="Kaggle"),
            zen.groupby("principle")[["earned", "total"]].sum().assign(platform="Zenodo"),
        ]).reset_index()
        by["score_%"] = (by.earned / by.total * 100).round(1)
        by["principle"] = pd.Categorical(by.principle, ["F", "A", "I", "R"], ordered=True)
        fig = px.bar(by.sort_values("principle"), x="principle", y="score_%", color="platform", barmode="group",
                     text="score_%", title="Same total score – opposite profiles",
                     labels={"principle": "FAIR principle", "score_%": "Score (%)"})
        st.plotly_chart(fig, width="stretch")

        comp = kag.merge(zen, on=["metric", "principle"], suffixes=("_kaggle", "_zenodo"))
        comp["Kaggle"] = comp.earned_kaggle.astype(str) + "/" + comp.total_kaggle.astype(str)
        comp["Zenodo"] = comp.earned_zenodo.astype(str) + "/" + comp.total_zenodo.astype(str)
        only_bad = st.toggle("Show only metrics where the platforms differ")
        view = comp[comp.earned_kaggle != comp.earned_zenodo] if only_bad else comp
        st.dataframe(view[["metric", "Kaggle", "maturity_kaggle", "Zenodo", "maturity_zenodo"]],
                     hide_index=True, width="stretch")

        st.subheader("What-if: apply the proposed fixes")
        fixes = {"Add keywords (F2-01M)": ("F", 1), "State the access level with COAR (A1-01M)": ("A", 1),
                 "Use registered vocabularies – EuroVoc, GeoNames (I2-01M)": ("I", 2)}
        cols = st.columns(len(fixes))
        chosen = [k for k, c in zip(fixes, cols) if c.checkbox(k)]
        proj = zen.groupby("principle")[["earned", "total"]].sum()
        for k in chosen:
            proj.loc[fixes[k][0], "earned"] += fixes[k][1]
        proj["earned"] = proj[["earned", "total"]].min(axis=1)
        base = zen.earned.sum() / zen.total.sum()
        new = proj.earned.sum() / proj.total.sum()
        c = st.columns([1, 2])
        c[0].metric("Projected Zenodo score", f"{new:.1%}", f"{(new - base) * 100:+.1f} points vs today")
        pj = (proj.earned / proj.total * 100).round(1).reindex(["F", "A", "I", "R"]).reset_index()
        pj.columns = ["principle", "score_%"]
        c[1].plotly_chart(px.bar(pj, x="principle", y="score_%", range_y=[0, 105], text="score_%",
                                 title="Projected score by principle"), width="stretch")
        st.caption("Projection assumes F-UJI detects each fix. The data dictionary (R1-01M-3) improves "
                   "reusability but is not scored in metric v0.8 – a limitation of automated assessment.")

    st.subheader("Five FAIR gaps in the Zenodo record")
    st.dataframe(hide_index=True, width="stretch", data=pd.DataFrame({
        "Gap": ["Undocumented variables", "No description / keywords", "No access level",
                "No registered vocabularies", "Weak provenance"],
        "Evidence (F-UJI)": ["R1-01M-3 = 0", "F2-01M 1/2", "A1-01M 0/1", "I2-01M 0/2", "R1.2 maturity 2"],
        "Fix in metadata record": ["data dictionary + variableMeasured", "description + 8 keywords",
                                   "COAR open-access URI", "EuroVoc, GeoNames, Wikidata",
                                   "IsDerivedFrom Kaggle, PROV, temporal coverage"],
    }))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metadata record (JSON-LD)")
        rec = FAIR / "metadata_record.jsonld"
        if rec.exists():
            st.json(json.loads(rec.read_text(encoding="utf-8")), expanded=1)
        else:
            missing_file(rec)
    with col2:
        st.subheader("Persistent identifiers")
        st.markdown(
            "- **Version DOI:** 10.5281/zenodo.23494643 (cited – exact file)\n"
            "- **Concept DOI:** 10.5281/zenodo.23494642 (all versions)\n"
            "- **CSV file:** no PID → SHA-256 `895b934b…1994e9`\n"
            "- **Proposed:** own DOI for cleaned data, ORCID, ROR, NUTS codes")
        for img in sorted(FAIR.glob("*validator*.png")):
            st.image(str(img), caption="schema.org validator: 0 errors, 0 warnings")

    st.subheader("Data dictionary")
    dd_path = FAIR / "data_dictionary.csv"
    if dd_path.exists():
        dd = pd.read_csv(dd_path)
        q = st.text_input("Search variables", placeholder="e.g. rent, year, heating")
        if q:
            dd = dd[dd.apply(lambda r: q.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
        st.dataframe(dd, hide_index=True, width="stretch")
    else:
        missing_file(dd_path)

# ---------------------------------------------------------------------------
# B - Quality
# ---------------------------------------------------------------------------
with tabB:
    if not RAW.exists():
        missing_file(RAW)
    else:
        comp, consistency, waves, info, numeric = raw_profile()
        c = st.columns(4)
        c[0].metric("Rows × columns", f"{info['rows']:,} × {info['cols']}")
        c[1].metric("Fully complete rows", f"{info['complete_rows']:,}")
        c[2].metric("Hidden placeholders", f"{int(comp.hidden.sum()):,}")
        c[3].metric("Postcodes broken by default import", f"{info['plz_leading_zero']:,}")

    reg_path = PROFILING / "issue_register.csv"
    if reg_path.exists():
        st.subheader("Issue register from the B profiling notebook")
        reg = pd.read_csv(reg_path)
        st.dataframe(reg[["issue_id", "issue", "dimension", "affected_rows", "pct", "likely_cause", "impact", "c_handling"]],
                     hide_index=True, width="stretch")
        charts = sorted(PROFILING.glob("chart*.png"))
        for col, img in zip(st.columns(max(len(charts), 1)), charts):
            col.image(str(img), caption=img.stem)

        st.subheader("Completeness – empty cells vs hidden placeholders")
        top_n = st.slider("Columns shown", 5, 49, 20)
        top = comp.sort_values("effective_%", ascending=False).head(top_n)[["empty", "hidden"]]
        top = (top / info["rows"] * 100).iloc[::-1]
        fig = go.Figure([go.Bar(y=top.index, x=top["empty"], name="Empty cell (NaN)", orientation="h"),
                         go.Bar(y=top.index, x=top["hidden"], name="Hidden 'no_information'", orientation="h")])
        fig.update_layout(barmode="stack", height=max(350, 22 * top_n), xaxis_title="% of rows")
        fig.add_vline(x=50, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Validity – try the plausibility limits live")
        rs = raw_columns(("baseRent", "livingSpace"))
        a, b = st.columns(2)
        rent_lo, rent_hi = a.select_slider("Base rent limits (€)", options=[0, 1, 10, 50, 100, 5_000, 10_000, 20_000, 50_000, 10_000_000], value=(50, 20_000))
        size_lo, size_hi = b.select_slider("Living space limits (m²)", options=[0, 1, 5, 10, 20, 500, 1_000, 5_000, 200_000], value=(10, 1_000))
        ok = rs.baseRent.between(rent_lo, rent_hi) & rs.livingSpace.between(size_lo, size_hi)
        m_all, s_all, r_all = stat_row(rs.baseRent, rs.livingSpace)
        m_ok, s_ok, r_ok = stat_row(rs.baseRent[ok], rs.livingSpace[ok])
        c = st.columns(4)
        c[0].metric("Rows removed", f"{int((~ok).sum()):,}", f"{(~ok).mean():.2%} of data", delta_color="off")
        c[1].metric("Mean base rent", f"€{m_ok:,.0f}", f"€{m_ok - m_all:,.0f} vs all rows")
        c[2].metric("Std base rent", f"€{s_ok:,.0f}", f"was €{s_all:,.0f}", delta_color="off")
        c[3].metric("corr(rent, size)", f"{r_ok:.2f}", f"was {r_all:.2f}")

        st.subheader("Rule builder – edit, add or delete plausibility rules")
        rules = pd.DataFrame({"column": ["baseRent", "livingSpace", "noRooms", "floor", "yearConstructed"],
                              "min": [50.0, 10.0, 1.0, -1.0, 1500.0], "max": [20000.0, 1000.0, 20.0, 100.0, 2021.0]})
        edited = st.data_editor(rules, num_rows="dynamic", hide_index=True, width="stretch",
                                column_config={"column": st.column_config.SelectboxColumn(options=numeric, required=True)})
        out = []
        for r in edited.dropna(subset=["column"]).itertuples(index=False):
            v = raw_columns((r.column,))[r.column]
            bad = v.notna() & ~v.between(r.min, r.max)
            out.append({"rule": f"{r.column} in [{r.min:g}, {r.max:g}]", "violations": int(bad.sum()),
                        "share": f"{bad.mean():.3%}", "example bad values": ", ".join(f"{x:g}" for x in v[bad].head(3))})
        if out:
            res = pd.DataFrame(out)
            st.dataframe(res, hide_index=True, width="stretch")
            st.caption(f"Total violations: {res.violations.sum():,} (a row can break several rules).")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Consistency")
            st.dataframe(hide_index=True, width="stretch", data=pd.DataFrame({"Check": consistency.keys(), "Rows affected": consistency.values()}))
        with col2:
            st.subheader("Uniqueness & timeliness")
            w = waves.reset_index().melt(id_vars="date", var_name="type", value_name="listings")
            st.plotly_chart(px.bar(w, x="date", y="listings", color="type", title="Listings per scrape wave"),
                            width="stretch")

        st.subheader("Explore any numeric column")
        col = st.selectbox("Column", numeric, index=numeric.index("baseRent") if "baseRent" in numeric else 0)
        s = raw_columns((col,))[col].dropna()
        st.plotly_chart(px.box(s, x=col, log_x=bool((s > 0).all()), points="outliers",
                               title=f"{col}: min {s.min():,.1f} · median {s.median():,.1f} · max {s.max():,.1f}"),
                        width="stretch")

# ---------------------------------------------------------------------------
# C - Cleaning
# ---------------------------------------------------------------------------
with tabC:
    audit_path = CLEANING / "audit_log.csv"
    if not audit_path.exists():
        missing_file(audit_path)
    else:
        audit = pd.read_csv(audit_path)
        audit["label"] = "Step " + audit.step.astype(str) + " – " + audit.issue
        c = st.columns(4)
        c[0].metric("Rows", f"{audit.rows_after.iloc[-1]:,}", f"{audit.rows_after.iloc[-1] - audit.rows_before.iloc[0]:,}", delta_color="off")
        c[1].metric("Columns", int(audit.cols_after.iloc[-1]), f"{int(audit.cols_after.iloc[-1] - audit.cols_before.iloc[0])}", delta_color="off")
        c[2].metric("Mean base rent", f"€{audit.mean_baseRent_after.iloc[-1]:,.0f}", f"was €{audit.mean_baseRent_before.iloc[0]:,.0f}", delta_color="off")
        c[3].metric("Std base rent", f"€{audit.std_baseRent_after.iloc[-1]:,.0f}", f"was €{audit.std_baseRent_before.iloc[0]:,.0f}", delta_color="off")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.line(audit, x="label", y="rows_after", markers=True,
                                    title="Rows after each step"), width="stretch")
        with col2:
            st.plotly_chart(px.bar(audit, x="label", y="values_changed", log_y=True,
                                   title="Values changed per step (log scale)"), width="stretch")

        st.subheader("Walk through the pipeline")
        step = st.select_slider("Step", options=audit.label.tolist(), value=audit.label.iloc[0])
        i = audit.label.tolist().index(step)
        r = audit.iloc[i]
        c = st.columns(4)
        c[0].metric("Rows after this step", f"{r.rows_after:,}", f"{r.rows_after - r.rows_before:,}", delta_color="off")
        c[1].metric("Columns", int(r.cols_after), f"{int(r.cols_after - r.cols_before):+d}", delta_color="off")
        c[2].metric("Mean base rent", f"€{r.mean_baseRent_after:,.0f}", f"{r.mean_baseRent_after - r.mean_baseRent_before:+,.0f}", delta_color="off")
        c[3].metric("Values changed", f"{r.values_changed:,}")
        st.info(f"**{r.action}** ({r.issue})\n\n{r.detail}")
        prog = audit.assign(stage=["done" if k <= i else "to come" for k in range(len(audit))])
        st.plotly_chart(px.bar(prog, x="label", y="rows_after", color="stage", range_y=[audit.rows_after.min() * 0.98, audit.rows_before.max() * 1.002],
                               color_discrete_map={"done": BLUE, "to come": "#D3D3D3"}, title="Rows after each step"),
                        width="stretch")
        st.subheader("Audit log")
        st.dataframe(audit.drop(columns="label"), hide_index=True, width="stretch")

        if CLEAN.exists() and RAW.exists():
            st.subheader("Base rent before vs after cleaning")
            before = raw_columns(("baseRent",))["baseRent"]
            after = clean_data()["baseRent"]
            both = pd.concat([before[before > 0].rename("value").to_frame().assign(data="Raw"),
                              after.rename("value").to_frame().assign(data="Cleaned")])
            st.plotly_chart(px.box(both, x="data", y="value", log_y=True, points="outliers",
                                   labels={"value": "Base rent (€, log scale)", "data": ""}),
                            width="stretch")

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
with tabG:
    mpath = GRAPH / "district_metrics.csv"
    col1, col2 = st.columns(2)
    for col, name, cap in [(col1, "graph_schema.png", "Property graph model"),
                           (col2, "district_graph.png", "District network (colour = community)")]:
        p = GRAPH / name
        if p.exists():
            col.image(str(p), caption=cap)
    if not mpath.exists():
        missing_file(mpath)
    else:
        m = pd.read_csv(mpath)
        c = st.columns(4)
        c[0].metric("Districts", len(m))
        c[1].metric("Communities", m.community.nunique(), "modularity 0.80", delta_color="off")
        c[2].metric("Rent assortativity", "0.53", "similar rents cluster", delta_color="off")
        c[3].metric("corr(degree, rent)", f"{m.degree.corr(m.median_rent_m2, method='spearman'):.2f}")

        if (GRAPH / "district_graph.gexf").exists():
            st.subheader("Interactive district network")
            G, pos = network()
            a, b = st.columns(2)
            colour_by = a.segmented_control("Colour by", ["community", "state", "median_rent_m2"], default="community") or "community"
            focus = b.selectbox("Highlight a district", ["–"] + sorted(G.nodes()))
            keep = [n for n in G.nodes() if not sel_states or G.nodes[n].get("state") in sel_states]
            H = G.subgraph(keep)
            ex, ey = [], []
            for u, v in H.edges():
                ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
            nd = pd.DataFrame([{"district": n, "x": pos[n][0], "y": pos[n][1], **H.nodes[n]} for n in H.nodes()])
            if nd.empty:
                st.info("No districts in the selected states – adjust the state filter at the top.")
            else:
                nd["community"] = nd["community"].astype(str)
                fig = px.scatter(nd, x="x", y="y", color=colour_by, size="n_listings", size_max=28,
                                 hover_name="district", hover_data={"state": True, "median_rent_m2": True,
                                                                    "betweenness": True, "x": False, "y": False},
                                 color_continuous_scale=BLUE_SCALE)
                fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.4, color="rgba(150,150,150,0.35)"),
                                         hoverinfo="skip", showlegend=False))
                fig.data = fig.data[-1:] + fig.data[:-1]
                if focus != "–" and focus in H:
                    nb = list(H.neighbors(focus))
                    hx = [pos[n][0] for n in [focus] + nb]; hy = [pos[n][1] for n in [focus] + nb]
                    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers+text", text=[focus] + [""] * len(nb),
                                             textposition="top center", marker=dict(size=14, color="rgba(0,0,0,0)",
                                             line=dict(width=2, color="red")), name="selected + neighbours"))
                fig.update_layout(height=620, xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white")
                st.plotly_chart(fig, width="stretch")
            if focus != "–" and focus in G:
                nb = pd.DataFrame([{"neighbour": n, **{k: G.nodes[n].get(k) for k in ["state", "median_rent_m2"]},
                                    "shared regions": G[focus][n].get("weight", 1)} for n in G.neighbors(focus)])
                d = G.nodes[focus]
                st.markdown(f"**{focus}** ({d.get('state')}) · degree {G.degree(focus)} · betweenness "
                            f"{d.get('betweenness', 0):.3f} · median rent €{d.get('median_rent_m2', 0):.2f}/m² "
                            f"(neighbours: €{nb['median_rent_m2'].median():.2f}/m²)" if len(nb) else f"**{focus}** has no neighbours.")
                if len(nb):
                    st.dataframe(nb.sort_values("shared regions", ascending=False), hide_index=True, width="stretch")

        states = ["All"] + sorted(m.state.unique())
        pick = st.selectbox("Filter by state", states)
        mv = m if pick == "All" else m[m.state == pick]
        fig = px.scatter(mv, x="degree", y="betweenness", color=mv.community.astype(str),
                         size="n_listings", hover_name="district",
                         hover_data={"state": True, "median_rent_m2": True},
                         labels={"color": "community"},
                         title="Degree vs betweenness (hover a bubble; size = listings)")
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        cols = ["district", "state", "degree", "betweenness", "median_rent_m2"]
        a.markdown("**Top 10 by degree**")
        a.dataframe(mv.nlargest(10, "degree")[cols], hide_index=True, width="stretch")
        b.markdown("**Top 10 by betweenness (bridges)**")
        b.dataframe(mv.nlargest(10, "betweenness")[cols], hide_index=True, width="stretch")

        st.markdown("**Communities**")
        g = (m.groupby("community")
               .agg(districts=("district", "size"),
                    main_state=("state", lambda s: s.value_counts().index[0]),
                    share=("state", lambda s: f"{s.value_counts().iloc[0] / len(s):.0%}"),
                    states=("state", "nunique"), median_rent_m2=("median_rent_m2", "median"))
               .sort_values("districts", ascending=False))
        st.dataframe(g, width="stretch")


# ---------------------------------------------------------------------------
# Explore (cleaned data, driven by the sidebar filters)
# ---------------------------------------------------------------------------
with tabE:
    if not CLEAN.exists():
        missing_file(CLEAN)
    else:
        cd = clean_data()
        f = cd[cd["regio1"].isin(sel_states) & cd["date"].isin(sel_waves) & cd["baseRent"].between(*rent_range)]
        if f.empty:
            st.warning("No listings match the filters.")
        else:
            c = st.columns(4)
            c[0].metric("Listings", f"{len(f):,}", f"{len(f) / len(cd):.0%} of cleaned data", delta_color="off")
            c[1].metric("Median base rent", f"€{f.baseRent.median():,.0f}")
            c[2].metric("Median rent per m²", f"€{f.rent_per_m2.median():.2f}")
            c[3].metric("Median living space", f"{f.livingSpace.median():.0f} m²")

            groups = {"Federal state": "regio1", "Flat type": "typeOfFlat", "Condition": "condition",
                      "Interior quality": "interiorQual", "Heating type": "heatingType"}
            g_label = st.segmented_control("Compare rent per m² by", list(groups), default="Federal state") or "Federal state"
            g = groups[g_label]
            hide_unknown = st.toggle("Hide 'unknown' category", value=True)
            ff = f[f[g] != "unknown"] if hide_unknown else f
            agg = (ff.groupby(g).agg(listings=("baseRent", "size"), median_rent_m2=("rent_per_m2", "median"))
                     .query("listings >= 30").sort_values("median_rent_m2").reset_index())
            st.plotly_chart(px.bar(agg, x="median_rent_m2", y=g, orientation="h", text=agg.median_rent_m2.round(2),
                                   hover_data=["listings"], height=max(350, 28 * len(agg)),
                                   labels={"median_rent_m2": "Median rent per m² (€)", g: g_label},
                                   title=f"Median rent per m² by {g_label.lower()} (groups with ≥ 30 listings)"),
                            width="stretch")
            if len(agg) >= 2:
                hi, lo = agg.iloc[-1], agg.iloc[0]
                gap = hi.median_rent_m2 - lo.median_rent_m2
                callout(f"<b>Key insight:</b> {str(hi[g]).replace('_', ' ')} is the most expensive {g_label.lower()} at "
                        f"<b>€{hi.median_rent_m2:.2f}/m²</b>, {str(lo[g]).replace('_', ' ')} the cheapest at "
                        f"<b>€{lo.median_rent_m2:.2f}/m²</b>, a gap of €{gap:.2f} "
                        f"({gap / lo.median_rent_m2:.0%}). For a 70 m² flat that is about "
                        f"<b>€{gap * 70:,.0f} more per month</b>.")

            col1, col2 = st.columns(2)
            with col1:
                trend = (f.groupby(["date", "regio1"])["rent_per_m2"].median().reset_index())
                top_states = f["regio1"].value_counts().head(6).index
                st.plotly_chart(px.line(trend[trend.regio1.isin(top_states)], x="date", y="rent_per_m2", color="regio1",
                                        markers=True, title="Rent per m² across scrape waves (6 largest states)",
                                        labels={"rent_per_m2": "Median €/m²", "date": "Scrape wave"}),
                                width="stretch")
            with col2:
                n = st.slider("Points in scatter", 500, 10_000, 3_000, step=500)
                smp = f.sample(min(n, len(f)), random_state=42)
                st.plotly_chart(px.scatter(smp, x="livingSpace", y="baseRent", color=g, opacity=0.5,
                                           log_x=True, log_y=True, hover_data=["regio2"],
                                           title="Living space vs base rent (log scales)"),
                                width="stretch")

            st.download_button("⬇️ Download this summary (CSV)", agg.to_csv(index=False).encode("utf-8"),
                               file_name=f"rent_by_{g}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Ask the data: tools and loop (the AI model can only call these fixed pandas functions)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # any OpenAI chat model with tool calling
MAX_TOOL_ROUNDS = 8

# Columns the assistant may group or filter by (cleaned dataset).
AI_TEXT_COLS = {
    "regio1": "federal state",
    "regio2": "district or city",
    "typeOfFlat": "flat type",
    "condition": "condition of the flat",
    "interiorQual": "interior quality",
    "heatingType": "heating type",
    "date": "scrape wave (2018-09, 2019-05, 2019-10, 2020-02)",
    "petsAllowed": "pets allowed (yes / no / negotiable)",
}
BOOLEAN = ["balcony", "hasKitchen", "lift", "garden", "cellar", "newlyConst"]
NUMERIC = {
    "baseRent": "base rent in € per month (without service charges)",
    "totalRent": "total rent in € per month",
    "serviceCharge": "service charges in € per month",
    "livingSpace": "living space in m²",
    "rent_per_m2": "base rent per m² in €",
    "noRooms": "number of rooms",
    "yearConstructed": "year the building was built",
}
STATS = ["median", "mean", "min", "max", "count"]

FILTER_SCHEMA = {
    "type": "object",
    "description": "Optional filters. Text columns take a list of exact values; "
                   "yes/no columns take true or false; number columns take {min, max}.",
    "properties": {
        **{c: {"type": "array", "items": {"type": "string"}, "description": d}
           for c, d in AI_TEXT_COLS.items()},
        **{c: {"type": "boolean"} for c in BOOLEAN},
        **{c: {"type": "object", "description": d,
               "properties": {"min": {"type": "number"}, "max": {"type": "number"}}}
           for c, d in NUMERIC.items()},
    },
    "additionalProperties": False,
}

TOOLS = [
    {
        "name": "list_values",
        "description": "List the distinct values of a text column with their listing counts. "
                       "Call this first when you need an exact spelling (e.g. 'Berlin' or "
                       "'Nordrhein_Westfalen') before filtering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": list(AI_TEXT_COLS)},
                "contains": {"type": "string", "description": "Optional case-insensitive search text."},
                "limit": {"type": "integer", "description": "Max values to return (default 30)."},
            },
            "required": ["column"],
            "additionalProperties": False,
        },
    },
    {
        "name": "summarise",
        "description": "Summary statistics (count, median, mean, min, max) of number columns "
                       "for the listings that match the filters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string", "enum": list(NUMERIC)}},
                "filters": FILTER_SCHEMA,
            },
            "required": ["columns"],
            "additionalProperties": False,
        },
    },
    {
        "name": "group_compare",
        "description": "Group the matching listings by one column and compute one statistic of a "
                       "number column per group. Use it for rankings and comparisons such as "
                       "'most expensive cities per m²'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "enum": list(AI_TEXT_COLS) + BOOLEAN},
                "metric": {"type": "string", "enum": list(NUMERIC)},
                "stat": {"type": "string", "enum": STATS},
                "filters": FILTER_SCHEMA,
                "min_listings": {"type": "integer", "description": "Drop groups smaller than this (default 30)."},
                "sort": {"type": "string", "enum": ["desc", "asc"]},
                "top_n": {"type": "integer", "description": "Rows to return (default 15, max 50)."},
            },
            "required": ["group_by", "metric", "stat"],
            "additionalProperties": False,
        },
    },
    {
        "name": "project_files",
        "description": "Read one of the project's result tables: 'quality_issues' (Component B "
                       "issue register), 'cleaning_log' (Component C audit log, step by step), or "
                       "'district_graph' (graph metrics per district: degree, betweenness, community).",
        "input_schema": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": ["quality_issues", "cleaning_log", "district_graph"]},
                "sort_by": {"type": "string", "description": "Optional column to sort by (descending)."},
                "top_n": {"type": "integer", "description": "Rows to return (default 25)."},
            },
            "required": ["table"],
            "additionalProperties": False,
        },
    },
]

# Same tools in the format the OpenAI API expects.
OPENAI_TOOLS = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in TOOLS]

SYSTEM = """You answer questions about a dataset of German apartment rental listings \
(ImmoScout24, four scrape waves 2018-09 to 2020-02). The data is the CLEANED dataset \
from the team's pipeline, already limited by the dashboard filters the user chose.

Rules:
- Get every number from a tool. Never guess or use outside knowledge for figures.
- Values in regio1 and regio2 use underscores (e.g. Nordrhein_Westfalen, Frankfurt_am_Main). \
If you are unsure of a spelling, call list_values first.
- Rent per m² means baseRent / livingSpace. Prefer the median over the mean, because rents are skewed.
- Answer in 2 to 6 short sentences or a small markdown table. Say how many listings an answer \
is based on, and mention when a group is small.
- If a question cannot be answered with these columns, say so plainly."""


class DataTools:
    """The pandas functions behind the tools."""

    def __init__(self, df: pd.DataFrame, files: dict):
        self.df = df
        self.files = files  # name -> Path

    # -- helpers --------------------------------------------------------
    def _filter(self, filters):
        d = self.df
        for col, val in (filters or {}).items():
            if col in AI_TEXT_COLS:
                vals = [val] if isinstance(val, str) else list(val)
                d = d[d[col].isin(vals)]
            elif col in BOOLEAN:
                d = d[d[col] == bool(val)]
            elif col in NUMERIC and isinstance(val, dict):
                if val.get("min") is not None:
                    d = d[d[col] >= val["min"]]
                if val.get("max") is not None:
                    d = d[d[col] <= val["max"]]
            else:
                raise ValueError(f"Unknown filter column: {col}")
        return d

    @staticmethod
    def _table(frame: pd.DataFrame) -> str:
        return frame.round(2).to_json(orient="records", force_ascii=False)

    # -- tools ----------------------------------------------------------
    def list_values(self, column, contains=None, limit=30):
        vc = self.df[column].value_counts()
        if contains:
            vc = vc[vc.index.astype(str).str.contains(contains, case=False, regex=False)]
        out = vc.head(min(int(limit), 200)).rename_axis(column).reset_index(name="listings")
        return json.dumps({"distinct_values": int(self.df[column].nunique()),
                           "values": json.loads(self._table(out))}, ensure_ascii=False)

    def summarise(self, columns, filters=None):
        d = self._filter(filters)
        if d.empty:
            return json.dumps({"listings": 0, "note": "No listings match these filters."})
        s = d[list(columns)].agg(["count", "median", "mean", "min", "max"]).T
        return json.dumps({"listings": len(d), "stats": json.loads(self._table(s.rename_axis("column").reset_index()))},
                          ensure_ascii=False)

    def group_compare(self, group_by, metric, stat, filters=None, min_listings=30, sort="desc", top_n=15):
        d = self._filter(filters)
        g = d.groupby(group_by)[metric].agg([stat, "count"]).rename(columns={"count": "listings"})
        g = g[g["listings"] >= int(min_listings)].sort_values(stat, ascending=(sort == "asc"))
        return json.dumps({"listings_used": len(d), "groups": len(g),
                           "rows": json.loads(self._table(g.head(min(int(top_n), 50)).reset_index()))},
                          ensure_ascii=False)

    def project_files(self, table, sort_by=None, top_n=25):
        path = self.files.get(table)
        if path is None or not path.exists():
            return json.dumps({"error": f"{table} file not found. Run the matching notebook first."})
        t = pd.read_csv(path)
        if sort_by and sort_by in t.columns:
            t = t.sort_values(sort_by, ascending=False)
        return json.dumps({"rows_total": len(t), "rows": json.loads(self._table(t.head(min(int(top_n), 60))))},
                          ensure_ascii=False)

    def run(self, name, args):
        if name not in {t["name"] for t in TOOLS}:
            raise ValueError(f"Unknown tool: {name}")
        return getattr(self, name)(**args)


def ask(client, history, tools: DataTools):
    """Run one question through the OpenAI model, calling tools until it answers.

    `history` is the list of chat messages (it is extended in place, so the next
    question keeps the conversation). Returns (answer_text, tool_log).
    """
    log = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + history,
            tools=OPENAI_TOOLS,
        )
        msg = response.choices[0].message
        calls = msg.tool_calls or []
        entry = {"role": "assistant", "content": msg.content or ""}
        if calls:
            entry["tool_calls"] = [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                   for c in calls]
        history.append(entry)

        if getattr(msg, "refusal", None):
            return "Sorry, I can't help with that question.", log
        if not calls:
            return (msg.content or "(no answer)").strip(), log

        for call in calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                out, err = tools.run(call.function.name, args), False
            except Exception as e:  # bad column, bad value: tell the model so it can fix the call
                args, out, err = call.function.arguments, f"Error: {e}", True
            log.append({"tool": call.function.name, "input": args, "output": out[:3000], "error": err})
            history.append({"role": "tool", "tool_call_id": call.id, "content": out})

    return "I stopped after too many steps. Try a more specific question.", log


# ---------------------------------------------------------------------------
# Ask the data (AI assistant; answers only from fixed pandas tools defined above)
# ---------------------------------------------------------------------------
with tabAI:
    st.subheader("Ask the data")
    st.caption(":material/auto_awesome: **AI-powered insights** · ask anything, get instant answers straight from the data.")
    try:
        import openai
    except ImportError:
        openai = None
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets.toml
        secret_key = None
    api_key = os.environ.get("OPENAI_API_KEY") or secret_key

    if openai is None:
        st.info("Install the AI package first:  `pip install openai`  and restart the dashboard.")
    elif not CLEAN.exists():
        missing_file(CLEAN)
    else:
        if not api_key:
            st.info("No OpenAI key found. Put it in `.streamlit/secrets.toml` in the project folder as  "
                    "`OPENAI_API_KEY = \"sk-...\"`  and restart the dashboard.")
        cd = clean_data()
        f_ai = cd[cd["regio1"].isin(sel_states) & cd["date"].isin(sel_waves) & cd["baseRent"].between(*rent_range)]
        st.caption(f"Answering from **{len(f_ai):,}** cleaned listings (change the filters at the top to narrow this).")

        examples = ["Which 10 cities have the highest median rent per m²?",
                    "How much more does a flat with a lift cost per m² in Berlin?",
                    "Compare median rent per m² across the four scrape waves.",
                    "Which data quality issues affected the most rows?"]
        picked_q = st.pills("Try an example", examples, key="ai_example")

        if "ai_chat" not in st.session_state:
            st.session_state.ai_chat, st.session_state.ai_history = [], []
        if st.session_state.ai_chat and st.button("Clear conversation", icon=":material/delete:"):
            st.session_state.ai_chat, st.session_state.ai_history = [], []
            st.rerun()

        for turn in st.session_state.ai_chat:
            with st.chat_message(turn["role"], avatar=":material/person:" if turn["role"] == "user"
                                 else ":material/insights:"):
                st.markdown(turn["text"])
                if turn.get("log"):
                    with st.expander(f"How this was calculated ({len(turn['log'])} queries)"):
                        for step in turn["log"]:
                            st.markdown(f"**{step['tool']}** `{json.dumps(step['input'], ensure_ascii=False)}`")
                            st.code(step["output"], language="json")

        question = st.chat_input("Ask about rents, districts, data quality…", key="ai_input") or picked_q
        if question and question != st.session_state.get("ai_last_example_done"):
            if picked_q == question:
                st.session_state.ai_last_example_done = question
            if not api_key:
                st.warning("Add your OpenAI key to .streamlit/secrets.toml first (see the note above).")
            else:
                st.session_state.ai_chat.append({"role": "user", "text": question})
                files = {"quality_issues": PROFILING / "issue_register.csv",
                         "cleaning_log": CLEANING / "audit_log.csv",
                         "district_graph": GRAPH / "district_metrics.csv"}
                history = st.session_state.ai_history
                start = len(history)
                history.append({"role": "user", "content": question})
                with st.spinner("Querying the data…"):
                    try:
                        answer, log = ask(openai.OpenAI(api_key=api_key), history, DataTools(f_ai, files))
                        if history[-1]["role"] != "assistant" or history[-1].get("tool_calls"):  # stopped mid-way: keep history valid
                            del history[start:]
                    except openai.AuthenticationError:
                        del history[start:]
                        answer, log = "That API key was rejected. Check it and try again.", []
                    except openai.APIError as e:
                        del history[start:]
                        answer, log = f"The AI service returned an error: {e}", []
                st.session_state.ai_chat.append({"role": "assistant", "text": answer, "log": log})
                st.rerun()
