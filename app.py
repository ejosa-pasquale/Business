from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:  # reportlab not installed
    A4 = None


from common import fetch_csv
from parking_occupancy import parse_parking_csv, estimate_daily_arrivals
from trento_chargers import summarize_chargers, DEFAULT_TRENTO_DATASET_PAGE
from demand import DemandInputs, demand_from_parking, FunnelInputs, demand_from_funnel
from sizing import ChargerTech, SizingInputs, size_for_tech
from finance import FinanceInputs, evaluate_finance
from optimizer_multi import TechCost, OptimizationInputs, optimize_mix_4tech
from formatting import eur, pct, num


def kwh_capacity_year(
    n_chargers: int,
    power_kw: float,
    connectors_per_charger: int,
    uptime: float,
    target_util: float,
    hours_per_day: float = 24.0,
) -> float:
    """Energy throughput capacity at target utilization (kWh/year).

    Notes
    - `target_util` is interpreted as an average utilization of the *available operating window*.
    - `hours_per_day` allows modeling night charging / restricted access windows.
    """
    hours_per_day = float(max(0.0, min(24.0, hours_per_day)))
    return (
        float(n_chargers)
        * float(connectors_per_charger)
        * float(power_kw)
        * 365.0
        * hours_per_day
        * float(uptime)
        * float(target_util)
    )




def build_quick_roi_pdf_report(
    site_name: str,
    inputs: dict,
    results: dict,
    forecast_df: pd.DataFrame,
) -> bytes:
    """Build a simple PDF report for the Quick ROI screen."""
    if A4 is None:
        # Fallback: return a minimal text file as bytes
        lines = []
        lines.append(f"Quick ROI Report — {site_name}")
        lines.append("")
        lines.append("INPUTS")
        for k, v in inputs.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("RESULTS")
        for k, v in results.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("FORECAST (first rows)")
        lines.append(forecast_df.head(10).to_csv(index=False))
        return ("\n".join(lines)).encode("utf-8")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>Quick ROI Report</b> — {site_name}", styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Input principali</b>", styles["Heading2"]))
    in_table = [["Parametro", "Valore"]] + [[str(k), str(v)] for k, v in inputs.items()]
    t = Table(in_table, colWidths=[7*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Risultati</b>", styles["Heading2"]))
    res_table = [["KPI", "Valore"]] + [[str(k), str(v)] for k, v in results.items()]
    t2 = Table(res_table, colWidths=[7*cm, 8*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Previsione 5 anni</b>", styles["Heading2"]))
    cols = ["Anno", "Sessioni/giorno", "kWh richiesti", "kWh venduti", "Ricavi (€)", "EBITDA (€)"]
    show = forecast_df.copy()
    # ensure expected columns
    for c in cols:
        if c not in show.columns:
            pass
    data = [cols] + show[cols].values.tolist()
    t3 = Table(data, colWidths=[1.4*cm, 2.4*cm, 2.6*cm, 2.6*cm, 2.7*cm, 2.7*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t3)
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generato il {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


st.set_page_config(page_title="Business Center EV Charging — ROI & Sizing Tool", layout="wide", page_icon="⚡")

st.markdown("""
<style>
/* Titolo/label del metric */
div[data-testid="stMetricLabel"] p {
  font-size: 12px !important;
}

/* Valore grande del metric */
div[data-testid="stMetricValue"] {
  font-size: 22px !important;
}

/* (opzionale) delta del metric */
div[data-testid="stMetricDelta"] {
  font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    .hero {
        background: linear-gradient(90deg, #0F172A 0%, #0EA5E9 55%, #22C55E 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.8rem; }
    .hero p { margin: 0.2rem 0 0 0; opacity: 0.9; }
    .card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.06);
    }
    .muted { color: #475569; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>⚡ Business Center eV Charging — ROI, CAPEX/OPEX, Strategia & Sizing</h1>
  <p>Valuta quante colonnine AC (fino 22 kW) e DC (fino 120 kW) servono: domanda → sizing → business case → raccomandazione.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------
# Sidebar: scenario inputs
# -----------------------
with st.sidebar:
    st.header("📍 Sito & Vincoli")
    site_name = st.text_input("Nome parcheggio / sito", value="Parcheggio — Trento")
    total_spots = st.number_input("Posti totali parcheggio", min_value=10, value=200, step=10)
    avg_stay_hours = st.slider("Sosta media (ore)", min_value=0.5, max_value=12.0, value=4.0, step=0.5)

    st.subheader("⚡ Vincoli tecnici")
    power_available_kw = st.number_input("Potenza disponibile (kW)", min_value=10.0, value=200.0, step=10.0)
    capex_budget = st.number_input("Budget CAPEX max (€)", min_value=5_000.0, value=50_000.0, step=10_000.0)

    st.subheader("🧠 Modalità domanda")
    demand_mode = st.radio(
        "Come stimare la domanda?",
        ["Ho dati storici del parcheggio (consigliato)", "Non li ho: uso funnel BEV"],
        index=0,
    )

    st.subheader("🔌 Parametri EV")
    bev_share = st.slider("Quota BEV sul traffico (%)", 0.0, 40.0, 7.0, step=0.5) / 100.0
    share_bev_that_charge = st.slider("Quota BEV che ricarica nel sito (%)", 0.0, 60.0, 18.0, step=1.0) / 100.0

    st.subheader("⏱️ Durata sessione")
    avg_session_hours_ac = st.slider("Durata media sessione AC (h)", 0.5, 10.0, 3.5, step=0.5)
    avg_session_hours_dc = st.slider("Durata media sessione DC (h)", 0.1, 2.0, 0.45, step=0.05)

    st.subheader("🔀 Mix domanda AC/DC")
    share_sessions_dc = st.slider("% sessioni su DC", 0, 100, 35, step=5) / 100.0
    kwh_per_session_ac = st.number_input("kWh medi per sessione AC", min_value=2.0, value=20.0, step=1.0)
    kwh_per_session_dc = st.number_input("kWh medi per sessione DC", min_value=5.0, value=35.0, step=1.0)

    st.subheader("🛠️ Affidabilità & saturazione")
    uptime = st.slider("Uptime tecnico (%)", 85, 100, 97) / 100.0
    target_util = st.slider("Target utilizzo medio (anti-coda) (%)", 10, 90, 40) / 100.0
    hours_ac = st.number_input("Ore operative/giorno per colonnina AC", min_value=1.0, max_value=24.0, value=24.0, step=1.0)
    hours_dc = st.number_input("Ore operative/giorno per colonnina DC", min_value=1.0, max_value=24.0, value=24.0, step=1.0)

    st.subheader("💶 Prezzi & costi")
    sell_price_ac = st.number_input("Prezzo vendita AC (€/kWh)", min_value=0.20, value=0.50, step=0.01, format="%.2f")
    sell_price_dc = st.number_input("Prezzo vendita DC (€/kWh)", min_value=0.20, value=0.75, step=0.01, format="%.2f")
    buy_price = st.number_input("Costo energia (€/kWh)", min_value=0.05, value=0.28, step=0.01, format="%.2f")
    variable_fee = st.number_input("OPEX variabile extra (€/kWh) — roaming/acquiring", min_value=0.0, value=0.03, step=0.01, format="%.2f")

    st.subheader("📈 Orizzonte investimento")
    years = st.slider("Anni analisi", 5, 15, 10)
    discount_rate = st.slider("WACC / tasso sconto (%)", 2.0, 15.0, 8.0, step=0.5) / 100.0
    kwh_growth = st.slider("Crescita kWh YoY (%)", 0.0, 30.0, 10.0, step=1.0) / 100.0

    st.subheader("🏗️ Costi unitari (editabili)")
    with st.expander("AC 22 kW (per colonnina)", expanded=True):
        ac_power = st.number_input("Potenza nominale AC (kW)", min_value=3.0, value=11.0, step=1.0)
        ac_connectors = st.number_input("Connettori per colonnina AC", min_value=1, value=2, step=1)
        ac_hw = st.number_input("Hardware AC (€)", min_value=500.0, value=2_000.0, step=100.0)
        ac_install = st.number_input("Installazione + opere AC (€)", min_value=500.0, value=3_000.0, step=100.0)
        ac_opex_year = st.number_input("OPEX fisso AC (€/anno per colonnina)", min_value=0.0, value=100.0, step=50.0)
        ac_mnt = st.number_input("Manutenzione annua AC (€/a)", min_value=0.0, value=100.0, step=10.0)
        ac_backend = st.number_input("Backend/CSMS annuo per colonnina AC (€/a)", min_value=0.0, value=50.0, step=10.0)

    st.caption("Per confrontare tecnologie diverse, qui separiamo le DC in 3 taglie (30/60/90 kW).")

    with st.expander("DC 30 kW (per colonnina)", expanded=True):
        dc30_power = st.number_input("Potenza nominale DC30 (kW)", min_value=20.0, value=30.0, step=5.0)
        dc30_connectors = st.number_input("Connettori per colonnina DC30", min_value=1, value=1, step=1)
        dc30_hw = st.number_input("Hardware DC30 (€)", min_value=5_000.0, value=8_000.0, step=1_000.0)
        dc30_install = st.number_input("Installazione + opere DC30 (€)", min_value=2_000.0, value=6_000.0, step=1_000.0)
        dc30_opex_year = st.number_input("OPEX fisso DC30 (€/anno per colonnina)", min_value=0.0, value=600.0, step=50.0)
        dc30_mnt = st.number_input("Manutenzione annua DC30 (€/a)", min_value=0.0, value=900.0, step=50.0)
        dc30_backend = st.number_input("Backend/CSMS annuo per colonnina DC30 (€/a)", min_value=0.0, value=420.0, step=20.0)

    with st.expander("DC 60 kW (per colonnina)", expanded=True):
        dc60_power = st.number_input("Potenza nominale DC60 (kW)", min_value=40.0, value=60.0, step=5.0)
        dc60_connectors = st.number_input("Connettori per colonnina DC60", min_value=1, value=2, step=1)
        dc60_hw = st.number_input("Hardware DC60 (€)", min_value=10_000.0, value=20_000.0, step=1_000.0)
        dc60_install = st.number_input("Installazione + opere DC60 (€)", min_value=3_000.0, value=6_000.0, step=1_000.0)
        dc60_opex_year = st.number_input("OPEX fisso DC60 (€/anno per colonnina)", min_value=0.0, value=300.0, step=50.0)
        dc60_mnt = st.number_input("Manutenzione annua DC60 (€/a)", min_value=0.0, value=100.0, step=50.0)
        dc60_backend = st.number_input("Backend/CSMS annuo per colonnina DC60 (€/a)", min_value=0.0, value=100.0, step=20.0)

    with st.expander("DC 90 kW (per colonnina)", expanded=True):
        dc90_power = st.number_input("Potenza nominale DC90 (kW)", min_value=60.0, value=90.0, step=5.0)
        dc90_connectors = st.number_input("Connettori per colonnina DC90", min_value=1, value=2, step=1)
        dc90_hw = st.number_input("Hardware DC90 (€)", min_value=15_000.0, value=35_000.0, step=1_000.0)
        dc90_install = st.number_input("Installazione + opere DC90 (€)", min_value=4_000.0, value=8_000.0, step=1_000.0)
        dc90_opex_year = st.number_input("OPEX fisso DC90 (€/anno per colonnina)", min_value=0.0, value=300.0, step=50.0)
        dc90_mnt = st.number_input("Manutenzione annua DC90 (€/a)", min_value=0.0, value=100.0, step=50.0)
        dc90_backend = st.number_input("Backend/CSMS annuo per colonnina DC90 (€/a)", min_value=0.0, value=100.0, step=20.0)

    st.subheader("🧱 CAPEX extra sito")
    grid_connection_capex = st.number_input(
        "Connessione rete / upgrade / scavi (CAPEX extra)",
        min_value=0.0,
        value=0.0,
        step=5_000.0,
        help="Inserisci una stima unica (MVP). In v1 puoi passare a range + Monte Carlo.",
    )
    signage_capex = st.number_input("Segnaletica + stalli dedicati (CAPEX)", min_value=0.0, value=500.0, step=500.0)

    st.subheader("🧾 OPEX fissi di sito")
    overhead_opex = st.number_input(
        "Overhead annuo (assicurazioni, pulizia, call center, affitto/royalty)",
        min_value=0.0,
        value=1_000.0,
        step=100.0,
    )
    overhead_growth = st.slider("Crescita OPEX fissi YoY (%)", 0.0, 10.0, 2.0, step=0.5) / 100.0

    st.subheader("🔎 Ricerca combinazioni")
    max_ac = st.slider("Max colonnine AC da testare", 0, 60, 30)
    cdc1, cdc2, cdc3 = st.columns(3)
    with cdc1:
        max_dc30 = st.slider("Max DC30 da testare", 0, 20, 6)
    with cdc2:
        max_dc60 = st.slider("Max DC60 da testare", 0, 20, 6)
    with cdc3:
        max_dc90 = st.slider("Max DC90 da testare", 0, 20, 6)


# -----------------------
# Demand estimation
# -----------------------
vehicles_per_day_est = None
parking_series_df = None

if demand_mode.startswith("Ho dati"):
    st.markdown("### 1) Domanda — da dati storici del parcheggio")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown("**Carica un CSV con serie temporale** (timestamp + occupazione oppure posti liberi).")
        up = st.file_uploader("CSV parcheggio", type=["csv", "tsv"])
        sample = st.checkbox("Usa un esempio fittizio (sample_data)", value=(up is None))

        if up is not None:
            raw = up.read()
            df_raw = pd.read_csv(io.BytesIO(raw))
        else:
            df_raw = None
            if sample:
                from pathlib import Path
                sample_path = Path("sample_data/parking_sample.csv")
                if sample_path.exists():
                    df_raw = pd.read_csv(sample_path)
                else:
                    st.info("Esempio (sample_data) non disponibile su questo deploy. Carica un CSV oppure disattiva 'sample'.")
                    df_raw = None
                    sample = False

        if df_raw is not None:
            try:
                series = parse_parking_csv(df_raw)
                arrivals = estimate_daily_arrivals(series, total_spots=int(total_spots), avg_stay_hours=float(avg_stay_hours))
                vehicles_per_day_est = float(arrivals["vehicles_per_day"].mean())
                parking_series_df = series.df

                st.success(f"Serie caricata. Veicoli stimati/giorno (media): {num(vehicles_per_day_est, 0)}")

                fig = px.line(series.df, x="ts", y="value", title="Serie storica parcheggio (metrica grezza)")
                st.plotly_chart(fig, use_container_width=True)

                fig2 = px.bar(arrivals, x="date", y="vehicles_per_day", title="Stima veicoli/giorno")
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Errore parsing CSV: {e}")

    with c2:
        st.markdown("<div class='card'><b>Nota</b><br><span class='muted'>Il calcolo veicoli/giorno è semplice: (occupazione media × 24) / sosta media. Se hai dati di ingressi reali, puoi inserirli direttamente sotto.</span></div>", unsafe_allow_html=True)
        vehicles_override = st.number_input(
            "Override veicoli/giorno (se conosci il dato)",
            min_value=0.0,
            value=float(vehicles_per_day_est) if vehicles_per_day_est is not None else 900.0,
            step=50.0,
        )
        vehicles_per_day_est = float(vehicles_override)

    # Compute demand
    dinp = DemandInputs(
        vehicles_per_day=float(vehicles_per_day_est),
        bev_share=float(bev_share),
        share_bev_that_charge=float(share_bev_that_charge),
        kwh_per_session_ac=float(kwh_per_session_ac),
        kwh_per_session_dc=float(kwh_per_session_dc),
        share_sessions_dc=float(share_sessions_dc),
    )
    dres = demand_from_parking(dinp)
    demand_kwh_year1 = dres.kwh_per_day * 365.0

else:
    st.markdown("### 1) Domanda — funnel BEV (senza dati parcheggio)")
    f1, f2 = st.columns([1.1, 0.9])

    with f1:
        st.markdown("Inserisci un funnel semplice: BEV → kWh annui → quota pubblico → quota cattura sito.")
        bev_2030 = st.number_input("BEV target anno base", min_value=0, value=30_000, step=1_000)
        kwh_per_bev_year = st.number_input("Consumo medio annuo per BEV (kWh/anno)", min_value=500.0, value=3_000.0, step=100.0)
        public_share = st.slider("Quota ricarica pubblica (%)", 0, 100, 30, step=5) / 100.0
        capture_share = st.slider("Quota cattura sito (%)", 0.1, 20.0, 4.0, step=0.5) / 100.0

        finp = FunnelInputs(
            bev_2030=int(bev_2030),
            kwh_per_bev_year=float(kwh_per_bev_year),
            public_share=float(public_share),
            capture_share=float(capture_share),
        )
        demand_kwh_year1 = demand_from_funnel(finp)

    with f2:
        st.markdown("<div class='card'><b>Tip</b><br><span class='muted'>Per rendere il funnel più credibile, usa anche i dataset pubblici sulle colonnine a Trento per stimare competizione e gap di copertura.</span></div>", unsafe_allow_html=True)
        st.caption("Dataset Comune di Trento (pagina):")
        st.code(DEFAULT_TRENTO_DATASET_PAGE)

    # convert annual kWh to daily + sessions via assumptions
    # Derive sessions from blended kWh/session
    blended_kwh_per_session = (1 - share_sessions_dc) * kwh_per_session_ac + share_sessions_dc * kwh_per_session_dc
    sessions_per_day = (demand_kwh_year1 / 365.0) / max(blended_kwh_per_session, 1e-6)
    dres = None
    class _Tmp: pass
    dres = _Tmp()
    dres.kwh_per_day = demand_kwh_year1 / 365.0
    dres.sessions_per_day = sessions_per_day
    dres.sessions_ac_per_day = sessions_per_day * (1 - share_sessions_dc)
    dres.sessions_dc_per_day = sessions_per_day * share_sessions_dc
    dres.kwh_ac_per_day = dres.sessions_ac_per_day * kwh_per_session_ac
    dres.kwh_dc_per_day = dres.sessions_dc_per_day * kwh_per_session_dc


st.divider()

"""Main Streamlit app.

This repo is used by non-technical users. We therefore keep a "Quick ROI" screen
with minimal inputs, on top of the full workflow tabs.
"""

# -----------------------
# Tabs: Quick ROI, Sizing, Finance, Strategy, Data
# -----------------------

quick_tab, sizing_tab, finance_tab, strategy_tab, data_tab = st.tabs(
    ["🚀 Quick ROI", "📐 Sizing", "💼 Business Case", "🧭 Strategia", "🗂️ Dati pubblici"]
)


with quick_tab:
    st.markdown("### Quick ROI — input semplici → ritorno investimento")
    st.caption(
        "K.I.S.S - Keep it simple and stupid - Inserisci auto che ricaricano, investimento e (opzionale) un mix AC/DC; il tool calcola domanda, capacità, kWh venduti e ritorno."
    )

    # Reset dei widget Quick ROI (Streamlit mantiene lo stato dei widget per key)
    # Questo evita che i default derivati dai dati (es. veicoli/giorno → sessioni/giorno) restino "bloccati".
    if st.button("🔄 Reset Quick ROI inputs", type="secondary", use_container_width=False):
        for k in [
            "quick_sessions_day",
            "quick_share_dc",
            "quick_n_ac",
            "quick_n_dc30",
            "quick_n_dc60",
            "quick_n_dc90",
            "quick_use_calc_capex",
            "quick_capex_input",
            "quick_sessions_day_user_set",
            "quick_upstream_hash",
        ]:
            if k in st.session_state:
                st.session_state.pop(k)
        st.rerun()

    st.caption(
        "Suggerimento: se hai toccato un input, Streamlit ne conserva il valore. "
        "Usa il reset per riallineare i default derivati dai dati."
    )

    q1, q2 = st.columns([1.1, 0.9])

    # Sync automatico dei default Quick ROI con i dati "a monte" (es. Override veicoli/giorno)
    # Streamlit mantiene i valori in session_state: aggiorniamo solo se l'utente NON ha modificato manualmente.
    def _mark_quick_sessions_day_user_set():
        st.session_state["quick_sessions_day_user_set"] = True

    if dres is not None:
        _new_hash = (
            float(vehicles_per_day_est) if vehicles_per_day_est is not None else None,
            float(bev_share),
            float(share_bev_that_charge),
            float(kwh_per_session_ac),
            float(kwh_per_session_dc),
            float(share_sessions_dc),
        )
        _old_hash = st.session_state.get("quick_upstream_hash")
        if _old_hash != _new_hash and not st.session_state.get("quick_sessions_day_user_set", False):
            st.session_state["quick_sessions_day"] = float(dres.sessions_per_day)
        st.session_state["quick_upstream_hash"] = _new_hash

    with q1:
        st.markdown("#### 1) Domanda (semplice)")
        q_sessions_day = st.number_input(
            "Auto che ricaricano al giorno (≈ sessioni/giorno)",
            min_value=0.0,
            value=float(dres.sessions_per_day) if dres is not None else 10.0,
            step=1.0,
            key="quick_sessions_day",
            on_change=_mark_quick_sessions_day_user_set,
            help=(
                "Domanda giornaliera stimata. Se hai inserito i dati del parcheggio (veicoli/giorno, % BEV, % che ricarica), "
                "questo valore viene pre-compilato come stima di sessioni/giorno."
            ),
        )
        q_share_dc = st.slider(
            "Quota sessioni DC (%)",
            min_value=0,
            max_value=100,
            value=int(share_sessions_dc * 100),
            step=5,
            key="quick_share_dc",
            help=(
                "Percentuale di sessioni effettuate su colonnine DC (ricarica veloce). "
                "Il resto delle sessioni è considerato su AC. Influenza kWh/sessione e prezzo medio di vendita."
            ),
        ) / 100.0

        q_kwh_req_year1 = (
            (q_sessions_day * (1 - q_share_dc) * float(kwh_per_session_ac)
             + q_sessions_day * q_share_dc * float(kwh_per_session_dc))
            * 365.0
        )

        st.metric("kWh richiesti (Year 1)", num(q_kwh_req_year1, 0))

        st.markdown("#### 2) Infrastruttura (semplice)")
        st.caption("Modifica leggermente il mix per vedere l'impatto su capacità e ROI.")

        qa, qb, qc, qd = st.columns(4)
        with qa:
            q_n_ac = st.number_input(
                "AC22",
                min_value=0,
                value=4,
                step=1,
                key="quick_n_ac",
                help="Numero di colonnine AC fino a 22 kW. Aumenta capacità annua (kWh vendibili) e CAPEX/OPEX.",
            )
        with qb:
            q_n_dc30 = st.number_input(
                "DC30",
                min_value=0,
                value=0,
                step=1,
                key="quick_n_dc30",
                help="Numero di colonnine DC ~30 kW. Utile per soste medio-brevi; aumenta capacità e CAPEX/OPEX.",
            )
        with qc:
            q_n_dc60 = st.number_input(
                "DC60",
                min_value=0,
                value=1,
                step=1,
                key="quick_n_dc60",
                help="Numero di colonnine DC ~60 kW. Compromesso tra tempi di ricarica e costi.",
            )
        with qd:
            q_n_dc90 = st.number_input(
                "DC90",
                min_value=0,
                value=0,
                step=1,
                key="quick_n_dc90",
                help="Numero di colonnine DC ~90 kW. Aumenta molto la capacità ma con CAPEX/OPEX più alti.",
            )

        # CAPEX calcolato (colonnine + costi sito)
        q_capex_calc = (
            q_n_ac * (float(ac_hw) + float(ac_install))
            + q_n_dc30 * (float(dc30_hw) + float(dc30_install))
            + q_n_dc60 * (float(dc60_hw) + float(dc60_install))
            + q_n_dc90 * (float(dc90_hw) + float(dc90_install))
            + float(grid_connection_capex)
            + float(signage_capex)
        )

        q_use_calc_capex = st.checkbox(
            "Usa CAPEX calcolato dal mix (altrimenti inserisco io l'investimento)",
            value=True,
            key="quick_use_calc_capex",
            help=(
                "Se attivo, il CAPEX totale è calcolato automaticamente dal mix di colonnine + costi sito (allaccio, segnaletica, ecc.). "
                "Se disattivo, puoi inserire manualmente l'investimento totale."
            ),
        )
        q_capex_input = st.number_input(
            "Investimento totale (CAPEX) (€)",
            min_value=0.0,
            value=float(q_capex_calc),
            step=5_000.0,
            key="quick_capex_input",
            help="CAPEX totale del progetto (hardware + installazione + costi sito). Usato per NPV/IRR/payback.",
        )
        q_capex_total = float(q_capex_calc) if q_use_calc_capex else float(q_capex_input)

    with q2:
        st.markdown("#### 3) Risultati")

        # Capacity (supply) at target utilization
        q_cap_ac = kwh_capacity_year(
            n_chargers=int(q_n_ac),
            power_kw=float(ac_power),
            connectors_per_charger=int(ac_connectors),
            uptime=float(uptime),
            target_util=float(target_util),
            hours_per_day=float(hours_ac),
        )
        q_cap_dc = (
            kwh_capacity_year(int(q_n_dc30), float(dc30_power), int(dc30_connectors), float(uptime), float(target_util), float(hours_dc))
            + kwh_capacity_year(int(q_n_dc60), float(dc60_power), int(dc60_connectors), float(uptime), float(target_util), float(hours_dc))
            + kwh_capacity_year(int(q_n_dc90), float(dc90_power), int(dc90_connectors), float(uptime), float(target_util), float(hours_dc))
        )
        q_kwh_sellable = float(q_cap_ac + q_cap_dc)
        q_kwh_sold = float(min(q_kwh_req_year1, q_kwh_sellable))
        q_kwh_lost = float(max(0.0, q_kwh_req_year1 - q_kwh_sellable))

        # Blended sell price based on AC/DC split of *demand* (not limited by capacity)
        q_kwh_ac_req = float(q_sessions_day) * (1 - float(q_share_dc)) * float(kwh_per_session_ac) * 365.0
        q_kwh_dc_req = float(q_sessions_day) * float(q_share_dc) * float(kwh_per_session_dc) * 365.0
        q_tot_req = max(q_kwh_ac_req + q_kwh_dc_req, 1e-9)
        q_sell_price_blended = (
            float(sell_price_ac) * q_kwh_ac_req + float(sell_price_dc) * q_kwh_dc_req
        ) / q_tot_req

        # Fixed OPEX (site + per charger fixed)
        q_fixed_opex = (
            float(overhead_opex)
            + int(q_n_ac) * (float(ac_opex_year) + float(ac_mnt) + float(ac_backend))
            + int(q_n_dc30) * (float(dc30_opex_year) + float(dc30_mnt) + float(dc30_backend))
            + int(q_n_dc60) * (float(dc60_opex_year) + float(dc60_mnt) + float(dc60_backend))
            + int(q_n_dc90) * (float(dc90_opex_year) + float(dc90_mnt) + float(dc90_backend))
        )

        fin_quick = FinanceInputs(
            years=int(years),
            discount_rate=float(discount_rate),
            capex_total=float(q_capex_total),
            price_sell_eur_per_kwh=float(q_sell_price_blended),
            price_buy_eur_per_kwh=float(buy_price),
            kwh_sold_year1=float(q_kwh_sold),
            kwh_growth_yoy=float(kwh_growth),
            fixed_opex_year1=float(q_fixed_opex),
            fixed_opex_growth_yoy=float(overhead_growth),
            variable_opex_per_kwh=float(variable_fee),
        )
        fres, _ = evaluate_finance(fin_quick)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("CAPEX", eur(q_capex_total))
        r2.metric("kWh vendibili (Year 1)", num(q_kwh_sellable, 0))
        r3.metric("kWh venduti (Year 1)", num(q_kwh_sold, 0))
        r4.metric("kWh persi", num(q_kwh_lost, 0))

        st.markdown("##### Risultati finanziari")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("NPV", eur(fres.npv))
        f2.metric("IRR", pct(fres.irr))
        f3.metric("Payback (anni, scontato)", num(fres.payback_year, 1))
        f4.metric("EBITDA anno 1", eur(fres.ebitda_year1))

        if q_kwh_lost > 0:
            st.warning(
                "Domanda > capacità: una parte della domanda non viene servita. Aumenta colonnine, ore operative o target utilizzo."
            )
        
        # -----------------------
        # Executive summary (cliente)
        # -----------------------
        st.markdown("#### Sintesi (cliente)")

        # 1) Semaforo capacità
        if q_kwh_req_year1 <= 0:
            st.info("Inserisci una domanda > 0 per valutare capacità e ritorni.")
        else:
            sat_ratio = q_kwh_sellable / max(q_kwh_req_year1, 1e-9)
            if sat_ratio >= 1.0:
                st.success(f"✅ Capacità sufficiente — puoi servire ~{sat_ratio*100:.0f}% della domanda (Year 1).")
            elif sat_ratio >= 0.9:
                st.warning(f"⚠️ Quasi saturo — puoi servire ~{sat_ratio*100:.0f}% della domanda (Year 1).")
            else:
                st.error(f"⛔ Capacità insufficiente — puoi servire ~{sat_ratio*100:.0f}% della domanda (Year 1).")

        # Suggerimento rapido di upgrade se saturi (euristica)
        if q_kwh_lost > 0:
            gap = q_kwh_lost
            # capacità marginale annua per +1 colonnina, a target util
            marg = {
                "AC22": kwh_capacity_year(1, float(ac_power), int(ac_connectors), float(uptime), float(target_util), float(hours_ac)),
                "DC30": kwh_capacity_year(1, float(dc30_power), int(dc30_connectors), float(uptime), float(target_util), float(hours_dc)),
                "DC60": kwh_capacity_year(1, float(dc60_power), int(dc60_connectors), float(uptime), float(target_util), float(hours_dc)),
                "DC90": kwh_capacity_year(1, float(dc90_power), int(dc90_connectors), float(uptime), float(target_util), float(hours_dc)),
            }
            capex_marg = {
                "AC22": float(ac_hw) + float(ac_install),
                "DC30": float(dc30_hw) + float(dc30_install),
                "DC60": float(dc60_hw) + float(dc60_install),
                "DC90": float(dc90_hw) + float(dc90_install),
            }
            # costo per kWh di capacità aggiunta (semplificato)
            best = None
            for tech, add_kwh in marg.items():
                if add_kwh <= 0:
                    continue
                cpk = capex_marg[tech] / add_kwh
                if best is None or cpk < best[1]:
                    best = (tech, cpk, add_kwh)
            if best is not None:
                tech, _, add_kwh = best
                n_need = int(np.ceil(gap / add_kwh))
                st.info(
                    f"Suggerimento rapido: aggiungi **{n_need}× {tech}** per coprire ~{num(gap,0)} kWh/anno di gap (a target utilizzo)."
                )

        # 2) Scenario rapido (sensibilità domanda ±20%)
        st.markdown("##### Scenario rapido (sensibilità domanda)")
        def _finance_for_demand_multiplier(mult: float):
            d = float(q_kwh_req_year1) * float(mult)
            sold = float(min(d, q_kwh_sellable))
            fin = FinanceInputs(
                years=int(years),
                discount_rate=float(discount_rate),
                capex_total=float(q_capex_total),
                price_sell_eur_per_kwh=float(q_sell_price_blended),
                price_buy_eur_per_kwh=float(buy_price),
                kwh_sold_year1=float(sold),
                kwh_growth_yoy=float(kwh_growth),
                fixed_opex_year1=float(q_fixed_opex),
                fixed_opex_growth_yoy=float(overhead_growth),
                variable_opex_per_kwh=float(variable_fee),
            )
            r, _ = evaluate_finance(fin)
            return sold, r

        s_base_sold, s_base = _finance_for_demand_multiplier(1.0)
        s_low_sold, s_low = _finance_for_demand_multiplier(0.8)
        s_high_sold, s_high = _finance_for_demand_multiplier(1.2)

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**-20% domanda**")
            st.metric("kWh venduti (Y1)", num(s_low_sold, 0))
            st.metric("NPV", eur(s_low.npv))
            st.metric("Payback", num(s_low.payback_year, 1))
        with s2:
            st.markdown("**Base**")
            st.metric("kWh venduti (Y1)", num(s_base_sold, 0))
            st.metric("NPV", eur(s_base.npv))
            st.metric("Payback", num(s_base.payback_year, 1))
        with s3:
            st.markdown("**+20% domanda**")
            st.metric("kWh venduti (Y1)", num(s_high_sold, 0))
            st.metric("NPV", eur(s_high.npv))
            st.metric("Payback", num(s_high.payback_year, 1))

        # 3) Assunzioni chiave (una riga)
        st.caption(
            "Assunzioni chiave: "
            f"Prezzo AC/DC {float(sell_price_ac):.2f}/{float(sell_price_dc):.2f} €/kWh · "
            f"Costo energia {float(buy_price):.2f} €/kWh · "
            f"Fee variabile {float(variable_fee):.2f} €/kWh · "
            f"Ore operative AC/DC {float(hours_ac):.0f}/{float(hours_dc):.0f}h · "
            f"Target utilizzo {float(target_util)*100:.0f}% · Uptime {float(uptime)*100:.0f}% · "
            f"kWh/sessione AC/DC {float(kwh_per_session_ac):.0f}/{float(kwh_per_session_dc):.0f}"
        )

        # 4) Output “riassunto proposta”
        st.markdown("##### Proposta (riassunto)")
        st.info(
            f"Installare **{int(q_n_ac)}× AC22**, **{int(q_n_dc30)}× DC30**, **{int(q_n_dc60)}× DC60**, **{int(q_n_dc90)}× DC90** "
            f"con **CAPEX {eur(q_capex_total)}**. "
            f"Nel primo anno: **{num(q_kwh_sold,0)} kWh venduti**, **EBITDA {eur(fres.ebitda_year1)}**, "
            f"**Payback {num(fres.payback_year,1)} anni**, **NPV {eur(fres.npv)}**."
        )

        st.markdown("#### Previsione domanda & ritorni (5 anni)")
        q_growth_sessions = st.slider(
            "Crescita annua domanda (%)",
            0.0, 80.0,
            35.0,
            step=1.0,
            key="quick_growth_sessions",
        ) / 100.0

        horizon_years = 5

        # Demand forecast (sessions/day and kWh requested) grows YoY; capacity stays constant
        rows = []
        for y in range(1, horizon_years + 1):
            sessions_day_y = float(q_sessions_day) * ((1 + q_growth_sessions) ** (y - 1))
            kwh_req_y = (
                (sessions_day_y * (1 - q_share_dc) * float(kwh_per_session_ac)
                 + sessions_day_y * q_share_dc * float(kwh_per_session_dc))
                * 365.0
            )
            kwh_sold_y = min(kwh_req_y, q_kwh_sellable)
            revenue_y = kwh_sold_y * float(q_sell_price_blended)
            energy_cost_y = kwh_sold_y * float(buy_price)
            var_cost_y = kwh_sold_y * float(variable_fee)

            fixed_y = float(q_fixed_opex) * ((1 + float(overhead_growth)) ** (y - 1))
            ebitda_y = revenue_y - energy_cost_y - var_cost_y - fixed_y

            rows.append({
                "Anno": y,
                "Sessioni/giorno": sessions_day_y,
                "kWh richiesti": kwh_req_y,
                "kWh venduti": kwh_sold_y,
                "Ricavi (€)": revenue_y,
                "EBITDA (€)": ebitda_y,
            })

        forecast_df = pd.DataFrame(rows)
        # Pretty rounding for display + PDF table
        forecast_df_disp = forecast_df.copy()
        forecast_df_disp["Sessioni/giorno"] = forecast_df_disp["Sessioni/giorno"].round(1)
        for c in ["kWh richiesti", "kWh venduti", "Ricavi (€)", "EBITDA (€)"]:
            forecast_df_disp[c] = forecast_df_disp[c].round(0).astype(int)

        cA, cB = st.columns([1.0, 1.0])
        with cA:
            fig = px.line(forecast_df, x="Anno", y="Sessioni/giorno", markers=True, title="Previsione auto in ricarica (sessioni/giorno)")
            st.plotly_chart(fig, use_container_width=True)

        with cB:
            fig2 = px.line(
                forecast_df,
                x="Anno",
                y=["Ricavi (€)", "EBITDA (€)"],
                markers=True,
                title="Ricavi ed EBITDA (5 anni) — limitati dalla capacità",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            forecast_df_disp,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Scarica report")
        st.caption("Report PDF con input + KPI + tabella di previsione 5 anni (utile per condivisione con cliente).")
        report_inputs = {
            "Sito": site_name,
            "Sessioni/giorno (Y1)": f"{q_sessions_day:.1f}",
            "Quota DC": f"{q_share_dc*100:.0f}%",
            "Crescita annua domanda": f"{q_growth_sessions*100:.0f}%",
            "Mix (AC22/DC30/DC60/DC90)": f"{q_n_ac}/{q_n_dc30}/{q_n_dc60}/{q_n_dc90}",
            "Ore operative AC / DC": f"{float(hours_ac):.0f}h / {float(hours_dc):.0f}h",
            "CAPEX totale": eur(q_capex_total),
            "Prezzo AC / DC": f"{float(sell_price_ac):.2f} / {float(sell_price_dc):.2f} €/kWh",
        }
        report_results = {
            "kWh richiesti (Y1)": num(q_kwh_req_year1, 0),
            "kWh vendibili (Y1)": num(q_kwh_sellable, 0),
            "kWh venduti (Y1)": num(q_kwh_sold, 0),
            "NPV": eur(fres.npv),
            "IRR": pct(fres.irr),
            "Payback (anni, scontato)": num(fres.payback_year, 1),
            "EBITDA anno 1": eur(fres.ebitda_year1),
        }
        pdf_bytes = build_quick_roi_pdf_report(site_name, report_inputs, report_results, forecast_df_disp)
        st.download_button(
            "⬇️ Scarica report (PDF)",
            data=pdf_bytes,
            file_name=f"QuickROI_{site_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            key="quick_download_pdf",
        )


        # Payback in sessions (very simple)
        margin_per_kwh = max(float(q_sell_price_blended) - float(buy_price) - float(variable_fee), 0.0)
        blended_kwh_per_session = (1 - q_share_dc) * float(kwh_per_session_ac) + q_share_dc * float(kwh_per_session_dc)
        margin_per_session = margin_per_kwh * max(blended_kwh_per_session, 1e-6)
        if margin_per_session > 0:
            sessions_to_payback = float(q_capex_total) / margin_per_session
            st.caption(f"Sessioni stimate per ripagare il CAPEX (semplificato): **{num(sessions_to_payback,0)}**")

with sizing_tab:
    st.markdown("### 2) Sizing — quante colonnine servono?")

    cA, cB = st.columns(2)

    # AC sizing
    with cA:
        st.markdown("#### AC 22 kW")
        tech_ac = ChargerTech(name="AC", power_kw=float(ac_power), connectors=int(ac_connectors))
        s_inp_ac = SizingInputs(
            demand_kwh_per_day=float(dres.kwh_ac_per_day),
            demand_sessions_per_day=float(dres.sessions_ac_per_day),
            uptime=float(uptime),
            target_utilization=float(target_util),
            avg_session_hours=float(avg_session_hours_ac),
        )
        sres_ac = size_for_tech(tech_ac, s_inp_ac)

        st.metric("Connettori richiesti", sres_ac.required_connectors)
        st.metric("Colonnine richieste", sres_ac.required_chargers)
        st.metric("Utilizzo richiesto (energy-based)", pct(sres_ac.achieved_utilization))

    with cB:
        st.markdown("#### DC (scegli taglia per sizing)")
        dc_sizing_choice = st.selectbox("Taglia DC per sizing", ["30 kW", "60 kW", "90 kW"], index=1)
        if dc_sizing_choice.startswith("30"):
            _dc_power = dc30_power
            _dc_connectors = dc30_connectors
        elif dc_sizing_choice.startswith("60"):
            _dc_power = dc60_power
            _dc_connectors = dc60_connectors
        else:
            _dc_power = dc90_power
            _dc_connectors = dc90_connectors

        tech_dc = ChargerTech(name=f"DC {dc_sizing_choice}", power_kw=float(_dc_power), connectors=int(_dc_connectors))
        s_inp_dc = SizingInputs(
            demand_kwh_per_day=float(dres.kwh_dc_per_day),
            demand_sessions_per_day=float(dres.sessions_dc_per_day),
            uptime=float(uptime),
            target_utilization=float(target_util),
            avg_session_hours=float(avg_session_hours_dc),
        )
        sres_dc = size_for_tech(tech_dc, s_inp_dc)

        st.metric("Connettori richiesti", sres_dc.required_connectors)
        st.metric("Colonnine richieste", sres_dc.required_chargers)
        st.metric("Utilizzo richiesto (energy-based)", pct(sres_dc.achieved_utilization))

    st.markdown("#### Sintesi domanda")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("kWh anno (Year 1)", num(demand_kwh_year1, 0))
    k2.metric("kWh/giorno", num(dres.kwh_per_day, 1))
    k3.metric("Sessioni/giorno", num(dres.sessions_per_day, 1))
    k4.metric("Mix DC sessioni", pct(share_sessions_dc))

    # Power feasibility check
    installed_power_req = sres_ac.required_chargers * ac_power + sres_dc.required_chargers * tech_dc.power_kw
    st.info(
        f"Potenza installata (sizing minimo): {num(installed_power_req,0)} kW vs Potenza disponibile: {num(power_available_kw,0)} kW"
    )

    st.markdown("---")
    st.markdown("### Sizing suggerito (5 anni, crescita domanda)")
    with st.expander("Suggerisci configurazione da sessioni iniziali + crescita", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            start_sessions_day = st.number_input(
                "Sessioni/giorno iniziali (auto che ricaricano)",
                min_value=0.0,
                value=float(dres.sessions_per_day),
                step=1.0,
                key="sizing5_start_sessions",
            )
            share_dc_for_suggest = (
                st.slider(
                    "Quota sessioni DC (%)",
                    0,
                    100,
                    int(share_sessions_dc * 100),
                    step=5,
                    key="sizing5_share_dc",
                )
                / 100.0
            )
        with c2:
            growth_yoy_suggest = st.slider("Crescita annua domanda (%)", 0.0, 80.0, 35.0, step=1.0, key="sizing5_growth") / 100.0
            years_suggest = st.selectbox("Orizzonte (anni)", [3, 4, 5, 6, 7, 10], index=2, key="sizing5_years")
        with c3:
            objective = st.selectbox("Obiettivo", ["Massimizza NPV", "Massimizza NPV/Capex"], index=0, key="sizing5_objective")
            capex_budget_chargers = float(max(capex_budget - grid_connection_capex - signage_capex, 0.0))
            st.caption(f"Budget colonnine (CAPEX max - costi sito): {num(capex_budget_chargers,0)} €")

        # Domanda Year 1 derivata da sessioni iniziali
        sessions_ac_day_s = float(start_sessions_day) * (1.0 - float(share_dc_for_suggest))
        sessions_dc_day_s = float(start_sessions_day) * float(share_dc_for_suggest)
        kwh_ac_year1_s = sessions_ac_day_s * float(kwh_per_session_ac) * 365.0
        kwh_dc_year1_s = sessions_dc_day_s * float(kwh_per_session_dc) * 365.0
        tot_kwh_s = max(kwh_ac_year1_s + kwh_dc_year1_s, 1e-9)
        blended_sell_price_s = (float(sell_price_ac) * kwh_ac_year1_s + float(sell_price_dc) * kwh_dc_year1_s) / tot_kwh_s

        # Definizione tecnologie per ottimizzatore (include ore operative)
        ac_cost_s = TechCost(name="AC22", capex_per_charger=float(ac_hw + ac_install), fixed_opex_per_charger_year=float(ac_opex_year), connectors=int(ac_connectors), power_kw=float(ac_power), hours_per_day=float(hours_ac))
        dc30_cost_s = TechCost(name="DC30", capex_per_charger=float(dc30_hw + dc30_install), fixed_opex_per_charger_year=float(dc30_opex_year), connectors=int(dc30_connectors), power_kw=float(dc30_power), hours_per_day=float(hours_dc))
        dc60_cost_s = TechCost(name="DC60", capex_per_charger=float(dc60_hw + dc60_install), fixed_opex_per_charger_year=float(dc60_opex_year), connectors=int(dc60_connectors), power_kw=float(dc60_power), hours_per_day=float(hours_dc))
        dc90_cost_s = TechCost(name="DC90", capex_per_charger=float(dc90_hw + dc90_install), fixed_opex_per_charger_year=float(dc90_opex_year), connectors=int(dc90_connectors), power_kw=float(dc90_power), hours_per_day=float(hours_dc))

        opt_inp_s = OptimizationInputs(
            kwh_ac_year1=float(kwh_ac_year1_s),
            kwh_dc_year1=float(kwh_dc_year1_s),
            uptime=float(uptime),
            target_utilization=float(target_util),
            power_available_kw=float(power_available_kw),
            capex_budget=float(capex_budget_chargers),
            years=int(years_suggest),
            discount_rate=float(discount_rate),
            price_sell_eur_per_kwh=float(blended_sell_price_s),
            price_buy_eur_per_kwh=float(buy_price),
            kwh_growth_yoy=float(growth_yoy_suggest),
            variable_opex_per_kwh=float(variable_fee),
            fixed_opex_overhead_year1=float(overhead_opex),
            fixed_opex_overhead_growth_yoy=float(overhead_growth),
            max_ac=int(max_ac),
            max_dc30=int(max_dc30),
            max_dc60=int(max_dc60),
            max_dc90=int(max_dc90),
        )

        best_s, all_s = optimize_mix_4tech(opt_inp_s, ac22=ac_cost_s, dc30=dc30_cost_s, dc60=dc60_cost_s, dc90=dc90_cost_s)

        import pandas as pd
        df_all = pd.DataFrame([r.__dict__ for r in all_s])
        df_all["capex_total"] = df_all["capex"] + float(grid_connection_capex) + float(signage_capex)
        df_all["npv_per_capex"] = df_all["npv"] / df_all["capex_total"].replace({0: float('nan')})

        if objective == "Massimizza NPV/Capex":
            df_valid = df_all.dropna(subset=["npv_per_capex"])
            if len(df_valid) > 0:
                row = df_valid.sort_values("npv_per_capex", ascending=False).iloc[0]
                best_s = best_s.__class__(**{k: row[k] for k in best_s.__dict__.keys()})

        st.success("Configurazione suggerita (entro vincoli di potenza e budget)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("AC22", int(best_s.n_ac))
        m2.metric("DC30", int(best_s.n_dc30))
        m3.metric("DC60", int(best_s.n_dc60))
        m4.metric("DC90", int(best_s.n_dc90))
        m5.metric("Potenza installata", f"{num(best_s.power_installed_kw,0)} kW")
        st.write(f"**CAPEX colonnine:** {num(best_s.capex,0)} €  |  **CAPEX totale (incl. sito):** {num(best_s.capex + grid_connection_capex + signage_capex,0)} €")
        st.write(f"**kWh venduti Year 1:** {num(best_s.kwh_sold_year1,0)}  |  **NPV:** {num(best_s.npv,0)}  |  **IRR:** {pct(best_s.irr)}  |  **Payback:** {num(best_s.payback,1)} anni")
        if str(best_s.notes).strip():
            st.caption(f"Note: {best_s.notes}")

        st.markdown("#### Frontiera (NPV vs CAPEX)")
        st.scatter_chart(df_all, x="capex_total", y="npv")
        st.caption("Ogni punto è un mix (AC22/DC30/DC60/DC90) entro vincoli. Usa l'obiettivo per scegliere il compromesso.")




with finance_tab:
    st.markdown("### 3) Business Case — CAPEX/OPEX/ROI")

    # Default configuration: use sizing results, but allow overrides
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Configurazione (manuale)")

        # Suggerimento: usa lo sizing AC e un sizing DC "tipo" (vedi tab precedente) solo come riferimento
        n_ac = st.number_input("Colonnine AC 22 kW", min_value=0, value=int(sres_ac.required_chargers), step=1)

        st.markdown("**DC (mix di taglie)**")
        n_dc30 = st.number_input("Colonnine DC 30 kW", min_value=0, value=0, step=1)
        n_dc60 = st.number_input("Colonnine DC 60 kW", min_value=0, value=int(sres_dc.required_chargers), step=1)
        n_dc90 = st.number_input("Colonnine DC 90 kW", min_value=0, value=0, step=1)

        capex = (
            n_ac * (ac_hw + ac_install)
            + n_dc30 * (dc30_hw + dc30_install)
            + n_dc60 * (dc60_hw + dc60_install)
            + n_dc90 * (dc90_hw + dc90_install)
            + grid_connection_capex
            + signage_capex
        )

        fixed_opex_year1 = (
            overhead_opex
            + n_ac * (ac_mnt + ac_backend)
            + n_dc30 * (dc30_mnt + dc30_backend)
            + n_dc60 * (dc60_mnt + dc60_backend)
            + n_dc90 * (dc90_mnt + dc90_backend)
        )

        installed_power_kw = (
            n_ac * ac_power
            + n_dc30 * dc30_power
            + n_dc60 * dc60_power
            + n_dc90 * dc90_power
        )

        # Capacità kWh/anno a target_util (anti-coda)
        # NB: "vendibili" = capacità teorica dell'infrastruttura, non i kWh effettivamente venduti.
        capacity_kwh_year1 = (
            kwh_capacity_year(n_ac, ac_power, ac_connectors, uptime, target_util, hours_per_day=hours_ac)
            + kwh_capacity_year(n_dc30, dc30_power, dc30_connectors, uptime, target_util, hours_per_day=hours_dc)
            + kwh_capacity_year(n_dc60, dc60_power, dc60_connectors, uptime, target_util, hours_per_day=hours_dc)
            + kwh_capacity_year(n_dc90, dc90_power, dc90_connectors, uptime, target_util, hours_per_day=hours_dc)
        )

        # kWh effettivamente venduti: limitati da domanda e capacità
        kwh_sold_year1 = float(min(demand_kwh_year1, capacity_kwh_year1))
        lost_kwh_year1 = float(max(0.0, demand_kwh_year1 - capacity_kwh_year1))

        # --- Lato domanda (auto / sessioni) per rendere la stima "reale"
        # Nota: modello semplice: 1 sessione ~ 1 auto che ricarica.
        # In modalità "Ho dati parcheggio" abbiamo anche il numero totale auto/giorno.
        is_parking_mode = demand_mode.startswith("Ho dati")

        # Sessioni e kWh/giorno: sempre disponibili via dres (anche nel funnel)
        sessions_day = float(
            getattr(
                dres,
                "sessions_per_day",
                (demand_kwh_year1 / 365.0)
                / max(
                    ((1 - share_sessions_dc) * kwh_per_session_ac + share_sessions_dc * kwh_per_session_dc),
                    1e-6,
                ),
            )
        )
        sessions_ac_day = float(getattr(dres, "sessions_ac_per_day", sessions_day * (1 - float(share_sessions_dc))))
        sessions_dc_day = float(getattr(dres, "sessions_dc_per_day", sessions_day * float(share_sessions_dc)))

        # Auto che ricaricano ~ sessioni
        vehicles_charging_day = sessions_day
        vehicles_charging_year = sessions_day * 365.0

        # Auto totali e BEV/giorno: solo se abbiamo dati parcheggio
        vehicles_per_day_val = float(getattr(dinp, "vehicles_per_day", 0.0)) if is_parking_mode else None
        bev_vehicles_day = (vehicles_per_day_val * float(bev_share)) if vehicles_per_day_val is not None else None

        demand_vs_capacity = float(demand_kwh_year1) / max(float(capacity_kwh_year1), 1e-9)

        st.metric("CAPEX totale", eur(capex))
        st.metric("OPEX fisso anno 1", eur(fixed_opex_year1))
        st.metric("Potenza installata", f"{num(installed_power_kw, 0)} kW")
        st.metric("kWh vendibili (Year 1, a target_util)", num(capacity_kwh_year1, 0))
        st.metric("kWh venduti (Year 1)", num(kwh_sold_year1, 0))

        st.markdown("#### Domanda (Year 1)")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Auto/giorno (tot)", num(vehicles_per_day_val, 0) if vehicles_per_day_val is not None else "—")
        d2.metric("BEV/giorno", num(bev_vehicles_day, 1) if bev_vehicles_day is not None else "—")
        d3.metric("Auto che ricaricano/giorno", num(vehicles_charging_day, 1))
        d4.metric("Auto che ricaricano/anno", num(vehicles_charging_year, 0))

        d5, d6, d7, d8 = st.columns(4)
        d5.metric("Sessioni AC/giorno", num(sessions_ac_day, 1))
        d6.metric("Sessioni DC/giorno", num(sessions_dc_day, 1))
        d7.metric("kWh richiesti (Year 1)", num(demand_kwh_year1, 0))
        d8.metric("Domanda/Capacità (a target_util)", pct(demand_vs_capacity))

        if lost_kwh_year1 > 1:
            st.warning(f"Domanda > capacità: perdi ~{num(lost_kwh_year1,0)} kWh nel Year 1 (a target_util).")

        if capex > capex_budget:
            st.warning("CAPEX sopra il budget impostato in sidebar.")
        if installed_power_kw > power_available_kw:
            st.warning("Potenza installata sopra la potenza disponibile.")
            st.warning("Potenza installata sopra la potenza disponibile.")

    with right:
        st.markdown("#### Risultati finanziari")

        # Prezzo medio (blended) in base al mix domanda AC/DC (Year 1)
        kwh_ac_year1_dem = float(sessions_ac_day) * float(kwh_per_session_ac) * 365.0
        kwh_dc_year1_dem = float(sessions_dc_day) * float(kwh_per_session_dc) * 365.0
        tot_kwh_dem = max(kwh_ac_year1_dem + kwh_dc_year1_dem, 1e-9)
        blended_sell_price = (float(sell_price_ac) * kwh_ac_year1_dem + float(sell_price_dc) * kwh_dc_year1_dem) / tot_kwh_dem

        fin_inp = FinanceInputs(
            years=int(years),
            discount_rate=float(discount_rate),
            capex_total=float(capex),
            price_sell_eur_per_kwh=float(blended_sell_price),
            price_buy_eur_per_kwh=float(buy_price),
            kwh_sold_year1=float(kwh_sold_year1),
            kwh_growth_yoy=float(kwh_growth),
            fixed_opex_year1=float(fixed_opex_year1),
            fixed_opex_growth_yoy=float(overhead_growth),
            variable_opex_per_kwh=float(variable_fee),
        )

        fin_res, fin_details = evaluate_finance(fin_inp)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NPV", eur(fin_res.npv))
        m2.metric("IRR", pct(fin_res.irr) if np.isfinite(fin_res.irr) else "n/a")
        m3.metric("Payback (anni, scontato)", "∞" if not np.isfinite(fin_res.payback_year) else num(fin_res.payback_year, 1))
        m4.metric("EBITDA anno 1", eur(fin_res.ebitda_year1))

        st.markdown("#### Payback visto come sessioni")
        # Margine per sessione (AC/DC) e numero di sessioni necessarie per ripagare il CAPEX
        margin_per_kwh_ac = float(sell_price_ac) - float(buy_price) - float(variable_fee)
        margin_per_kwh_dc = float(sell_price_dc) - float(buy_price) - float(variable_fee)
        margin_per_session_ac = margin_per_kwh_ac * float(kwh_per_session_ac)
        margin_per_session_dc = margin_per_kwh_dc * float(kwh_per_session_dc)

        sess_total_day = max(float(sessions_ac_day) + float(sessions_dc_day), 0.0)
        if sess_total_day > 1e-9:
            share_dc_sess = float(sessions_dc_day) / sess_total_day
        else:
            share_dc_sess = float(share_sessions_dc) if 'share_sessions_dc' in locals() else 0.0

        blended_margin_per_session = (1.0 - share_dc_sess) * margin_per_session_ac + share_dc_sess * margin_per_session_dc

        if blended_margin_per_session <= 0:
            st.warning("Con i prezzi/costi attuali il margine per sessione è <= 0: il payback in sessioni non è definito.")
        else:
            sessions_needed = float(capex) / blended_margin_per_session
            st.metric("Sessioni necessarie (stima)", num(sessions_needed, 0))

            # Curva payback (anni) al variare delle sessioni/giorno
            x = np.linspace(1.0, max(5.0, sess_total_day * 3.0, 100.0), 100)
            y = float(capex) / (blended_margin_per_session * x * 365.0)
            df_pb = pd.DataFrame({"sessioni_giorno": x, "payback_anni": y})
            fig_pb = px.line(df_pb, x="sessioni_giorno", y="payback_anni", title="Payback (anni) vs sessioni/giorno")
            st.plotly_chart(fig_pb, use_container_width=True)

        df_cf = pd.DataFrame({
            "year": list(range(0, years + 1)),
            "cashflow": fin_res.cashflows,
        })
        fig_cf = px.bar(df_cf, x="year", y="cashflow", title="Cashflow (Year 0 = CAPEX)")
        st.plotly_chart(fig_cf, use_container_width=True)

        df_pnl = pd.DataFrame({
            "year": list(range(1, years + 1)),
            "kwh": fin_details["kwh"],
            "revenue": fin_details["revenue"],
            "energy_cost": fin_details["energy_cost"],
            "fixed_opex": fin_details["fixed_opex"],
            "var_cost": fin_details["var_cost"],
            "ebitda": fin_details["ebitda"],
        })
        fig_ebitda = px.line(df_pnl, x="year", y="ebitda", title="EBITDA per anno")
        st.plotly_chart(fig_ebitda, use_container_width=True)

        with st.expander("Scarica P&L (CSV)"):
            st.download_button(
                "Download CSV",
                data=df_pnl.to_csv(index=False).encode("utf-8"),
                file_name="pnl_trento_ev.csv",
                mime="text/csv",
            )


with strategy_tab:
    st.markdown("### 4) Strategia — mix AC/DC ottimale (vincoli potenza & budget)")

    st.markdown(
        """
- **AC 22 kW** tende a massimizzare copertura per sosta lunga (costo basso, più stalli).
- **DC 120 kW** massimizza throughput e visibilità (sosta breve, alta rotazione), ma pesa su potenza e CAPEX.

Qui facciamo una **ricerca brute-force** su combinazioni AC/DC entro i vincoli e scegliamo quella con **NPV massimo**.
"""
    )

    ac_cost = TechCost(
        name="AC22",
        capex_per_charger=float(ac_hw + ac_install),
        fixed_opex_per_charger_year=float(ac_mnt + ac_backend),
        connectors=int(ac_connectors),
        power_kw=float(ac_power),
        hours_per_day=float(hours_ac),
    )
    dc30_cost = TechCost(
        name="DC30",
        capex_per_charger=float(dc30_hw + dc30_install),
        fixed_opex_per_charger_year=float(dc30_mnt + dc30_backend),
        connectors=int(dc30_connectors),
        power_kw=float(dc30_power),
        hours_per_day=float(hours_dc),
    )
    dc60_cost = TechCost(
        name="DC60",
        capex_per_charger=float(dc60_hw + dc60_install),
        fixed_opex_per_charger_year=float(dc60_mnt + dc60_backend),
        connectors=int(dc60_connectors),
        power_kw=float(dc60_power),
        hours_per_day=float(hours_dc),
    )
    dc90_cost = TechCost(
        name="DC90",
        capex_per_charger=float(dc90_hw + dc90_install),
        fixed_opex_per_charger_year=float(dc90_mnt + dc90_backend),
        connectors=int(dc90_connectors),
        power_kw=float(dc90_power),
        hours_per_day=float(hours_dc),
    )

    # Demand split year 1
    kwh_ac_year1 = float(dres.kwh_ac_per_day * 365.0)
    kwh_dc_year1 = float(dres.kwh_dc_per_day * 365.0)

    # Prezzo medio (blended) in base al mix domanda AC/DC (Year 1)
    tot_kwh = max(kwh_ac_year1 + kwh_dc_year1, 1e-9)
    blended_sell_price = (float(sell_price_ac) * kwh_ac_year1 + float(sell_price_dc) * kwh_dc_year1) / tot_kwh

    opt_inp = OptimizationInputs(
        kwh_ac_year1=kwh_ac_year1,
        kwh_dc_year1=kwh_dc_year1,
        uptime=float(uptime),
        target_utilization=float(target_util),
        power_available_kw=float(power_available_kw),
        capex_budget=float(capex_budget - grid_connection_capex - signage_capex),
        years=int(years),
        discount_rate=float(discount_rate),
        price_sell_eur_per_kwh=float(blended_sell_price),
        price_buy_eur_per_kwh=float(buy_price),
        kwh_growth_yoy=float(kwh_growth),
        variable_opex_per_kwh=float(variable_fee),
        fixed_opex_overhead_year1=float(overhead_opex),
        fixed_opex_overhead_growth_yoy=float(overhead_growth),
        max_ac=int(max_ac),
        max_dc30=int(max_dc30),
        max_dc60=int(max_dc60),
        max_dc90=int(max_dc90),
    )

    best, allres = optimize_mix_4tech(opt_inp, ac22=ac_cost, dc30=dc30_cost, dc60=dc60_cost, dc90=dc90_cost)

    st.markdown("#### Raccomandazione")
    rec1, rec2, rec3, rec4, rec5 = st.columns(5)
    rec1.metric("AC22", best.n_ac)
    rec2.metric("DC30", best.n_dc30)
    rec3.metric("DC60", best.n_dc60)
    rec4.metric("DC90", best.n_dc90)
    rec5.metric("NPV", eur(best.npv))

    st.info(
        f"Potenza installata (hardware): {num(best.power_installed_kw,0)} kW | "
        f"CAPEX extra sito: {eur(grid_connection_capex + signage_capex)} | "
        f"kWh venduti anno 1: {num(best.kwh_sold_year1,0)} | "
        f"Note: {best.notes or '—'}"
    )

    # Show top 20
    top = pd.DataFrame([asdict(x) for x in allres[:20]])
    if len(top) > 0:
        top["capex_total"] = top["capex"] + grid_connection_capex + signage_capex
        st.dataframe(
            top[[
                "n_ac",
                "n_dc30",
                "n_dc60",
                "n_dc90",
                "kwh_sold_year1",
                "npv",
                "irr",
                "payback",
                "capex_total",
                "power_installed_kw",
                "notes",
            ]],
            use_container_width=True,
        )

        fig = px.scatter(
            top,
            x="capex",
            y="npv",
            size="power_installed_kw",
            hover_data=["n_ac","n_dc30","n_dc60","n_dc90","kwh_sold_year1","payback","irr","notes"],
            title="Frontiera (Top 20): NPV vs CAPEX (hardware+install)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Roadmap (regole pratiche)")
    st.markdown(
        """
- **Fase 1 (0–12 mesi)**: installa il mix consigliato ma con priorità a **AC** se la sosta è lunga (≥2–3h).
- **Trigger DC**: aggiungi DC se **utilizzo DC** supera ~20–25% medio (o code/saturazione oltre target).
- **Vincolo rete**: se la potenza è limitata, abilita **load management** e pianifica upgrade (MT/BT) a step.
"""
    )


with data_tab:
    st.markdown("### 5) Dati pubblici — competizione & contesto")

    st.markdown(
        """
Questa sezione serve a:
- vedere quante colonnine esistono già a Trento (competizione, coverage)
- importare CSV pubblici (quando hai l'URL diretto al file)

**Nota:** la pagina dataset del Comune di Trento è un landing; spesso il link al CSV è all'interno della pagina.
Puoi incollare qui l'URL *diretto* al CSV oppure caricare il file.
"""
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Colonnine esistenti (Comune di Trento)")
        url = st.text_input("URL diretto CSV (opzionale)", value="")
        up = st.file_uploader("Oppure carica CSV colonnine", type=["csv","tsv"], key="chargers")

        df_ch = None
        meta = ""
        if up is not None:
            df_ch = pd.read_csv(up)
            meta = "upload"
        elif url.strip():
            try:
                fr = fetch_csv(url.strip())
                df_ch = fr.df
                meta = fr.note
            except Exception as e:
                st.error(f"Fetch fallito: {e}")

        if df_ch is None:
            st.info("Nessun file caricato. Puoi usare la pagina dataset come riferimento:")
            st.code(DEFAULT_TRENTO_DATASET_PAGE)
        else:
            summ = summarize_chargers(df_ch)
            st.success(f"Caricate {summ.n_points} righe | locations stimate: {summ.n_locations} | meta: {meta}")
            st.dataframe(df_ch.head(50), use_container_width=True)

    with c2:
        st.markdown("#### Esporta scenario")
        scenario = {
            "site": site_name,
            "total_spots": total_spots,
            "avg_stay_hours": avg_stay_hours,
            "power_available_kw": power_available_kw,
            "capex_budget": capex_budget,
            "bev_share": bev_share,
            "share_bev_that_charge": share_bev_that_charge,
            "share_sessions_dc": share_sessions_dc,
            "kwh_per_session_ac": kwh_per_session_ac,
            "kwh_per_session_dc": kwh_per_session_dc,
            "uptime": uptime,
            "target_util": target_util,
            "sell_price_ac": sell_price_ac,
            "sell_price_dc": sell_price_dc,
            "buy_price": buy_price,
            "variable_fee": variable_fee,
            "years": years,
            "discount_rate": discount_rate,
            "kwh_growth": kwh_growth,
            "ac": {
                "power": ac_power,
                "connectors": ac_connectors,
                "hw": ac_hw,
                "install": ac_install,
                "mnt": ac_mnt,
                "backend": ac_backend,
            },
            "dc30": {
                "power": dc30_power,
                "connectors": dc30_connectors,
                "hw": dc30_hw,
                "install": dc30_install,
                "mnt": dc30_mnt,
                "backend": dc30_backend,
            },
            "dc60": {
                "power": dc60_power,
                "connectors": dc60_connectors,
                "hw": dc60_hw,
                "install": dc60_install,
                "mnt": dc60_mnt,
                "backend": dc60_backend,
            },
            "dc90": {
                "power": dc90_power,
                "connectors": dc90_connectors,
                "hw": dc90_hw,
                "install": dc90_install,
                "mnt": dc90_mnt,
                "backend": dc90_backend,
            },
            "capex_extra": {
                "grid_connection": grid_connection_capex,
                "signage": signage_capex,
            },
            "opex_overhead": {
                "overhead": overhead_opex,
                "growth": overhead_growth,
            },
        }
        st.download_button(
            "Download scenario JSON",
            data=pd.Series(scenario).to_json(),
            file_name="scenario_trento_ev.json",
            mime="application/json",
        )

st.caption("MVP — costruito per essere esteso: Monte Carlo su CAPEX connessione rete, API parsing dataset Comune, tariffazione per kW impegnati, e ottimizzazione più realistica su code e profili orari.")
