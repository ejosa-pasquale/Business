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

from common import fetch_csv
from parking_occupancy import parse_parking_csv, estimate_daily_arrivals
from trento_chargers import summarize_chargers, DEFAULT_TRENTO_DATASET_PAGE
from demand import DemandInputs, demand_from_parking, FunnelInputs, demand_from_funnel
from sizing import ChargerTech, SizingInputs, size_for_tech
from finance import FinanceInputs, evaluate_finance
from optimizer_multi import TechCost, OptimizationInputs, optimize_mix_4tech
from formatting import eur, pct, num


def kwh_capacity_year(n_chargers: int, power_kw: float, connectors_per_charger: int, uptime: float, target_util: float) -> float:
    """Energy throughput capacity at target utilization (kWh/year)."""
    return float(n_chargers) * float(connectors_per_charger) * float(power_kw) * 8760.0 * float(uptime) * float(target_util)



st.set_page_config(page_title="Trento EV Charging — ROI & Sizing Tool", layout="wide", page_icon="⚡")

st.markdown(
    """
<style>
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    .hero {
        background: linear-gradient(90deg, #0F172A 0%, #0EA5E9 55%, #22C55E 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.8rem; }
    .hero p { margin: 0.2rem 0 0 0; opacity: 0.9; }
    .card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
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
  <h1>⚡ Trento EV Charging — ROI, CAPEX/OPEX, Strategia & Sizing</h1>
  <p>Valuta quante colonnine AC (fino 22 kW) e DC (fino 120 kW) installare in un parcheggio a Trento: domanda → sizing → business case → raccomandazione.</p>
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
    total_spots = st.number_input("Posti totali parcheggio", min_value=10, value=300, step=10)
    avg_stay_hours = st.slider("Sosta media (ore)", min_value=0.5, max_value=12.0, value=3.0, step=0.5)

    st.subheader("⚡ Vincoli tecnici")
    power_available_kw = st.number_input("Potenza disponibile (kW)", min_value=10.0, value=250.0, step=10.0)
    capex_budget = st.number_input("Budget CAPEX max (€)", min_value=5_000.0, value=250_000.0, step=10_000.0)

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
    avg_session_hours_ac = st.slider("Durata media sessione AC (h)", 0.5, 10.0, 2.5, step=0.5)
    avg_session_hours_dc = st.slider("Durata media sessione DC (h)", 0.1, 2.0, 0.45, step=0.05)

    st.subheader("🔀 Mix domanda AC/DC")
    share_sessions_dc = st.slider("% sessioni su DC", 0, 100, 35, step=5) / 100.0
    kwh_per_session_ac = st.number_input("kWh medi per sessione AC", min_value=2.0, value=18.0, step=1.0)
    kwh_per_session_dc = st.number_input("kWh medi per sessione DC", min_value=5.0, value=32.0, step=1.0)

    st.subheader("🛠️ Affidabilità & saturazione")
    uptime = st.slider("Uptime tecnico (%)", 85, 100, 97) / 100.0
    target_util = st.slider("Target utilizzo medio (anti-coda) (%)", 10, 90, 40) / 100.0
    hours_ac = st.number_input("Ore operative/giorno per colonnina AC", min_value=1.0, max_value=24.0, value=24.0, step=1.0)
    hours_dc = st.number_input("Ore operative/giorno per colonnina DC", min_value=1.0, max_value=24.0, value=24.0, step=1.0)

    st.subheader("💶 Prezzi & costi")
    sell_price_ac = st.number_input("Prezzo vendita AC (€/kWh)", min_value=0.20, value=0.55, step=0.01, format="%.2f")
    sell_price_dc = st.number_input("Prezzo vendita DC (€/kWh)", min_value=0.20, value=0.75, step=0.01, format="%.2f")
    buy_price = st.number_input("Costo energia (€/kWh)", min_value=0.05, value=0.28, step=0.01, format="%.2f")
    variable_fee = st.number_input("OPEX variabile extra (€/kWh) — roaming/acquiring", min_value=0.0, value=0.03, step=0.01, format="%.2f")

    st.subheader("📈 Orizzonte investimento")
    years = st.slider("Anni analisi", 5, 15, 10)
    discount_rate = st.slider("WACC / tasso sconto (%)", 2.0, 15.0, 8.0, step=0.5) / 100.0
    kwh_growth = st.slider("Crescita kWh YoY (%)", 0.0, 30.0, 10.0, step=1.0) / 100.0

    st.subheader("🏗️ Costi unitari (editabili)")
    with st.expander("AC 22 kW (per colonnina)", expanded=True):
        ac_power = st.number_input("Potenza nominale AC (kW)", min_value=3.0, value=22.0, step=1.0)
        ac_connectors = st.number_input("Connettori per colonnina AC", min_value=1, value=2, step=1)
        ac_hw = st.number_input("Hardware AC (€)", min_value=500.0, value=2_000.0, step=100.0)
        ac_install = st.number_input("Installazione + opere AC (€)", min_value=500.0, value=2_500.0, step=100.0)
        ac_opex_year = st.number_input("OPEX fisso AC (€/anno per colonnina)", min_value=0.0, value=300.0, step=50.0)
        ac_mnt = st.number_input("Manutenzione annua AC (€/a)", min_value=0.0, value=120.0, step=10.0)
        ac_backend = st.number_input("Backend/CSMS annuo per colonnina AC (€/a)", min_value=0.0, value=180.0, step=10.0)

    st.caption("Per confrontare tecnologie diverse, qui separiamo le DC in 3 taglie (30/60/90 kW).")

    with st.expander("DC 30 kW (per colonnina)", expanded=True):
        dc30_power = st.number_input("Potenza nominale DC30 (kW)", min_value=20.0, value=30.0, step=5.0)
        dc30_connectors = st.number_input("Connettori per colonnina DC30", min_value=1, value=1, step=1)
        dc30_hw = st.number_input("Hardware DC30 (€)", min_value=5_000.0, value=22_000.0, step=1_000.0)
        dc30_install = st.number_input("Installazione + opere DC30 (€)", min_value=2_000.0, value=12_000.0, step=1_000.0)
        dc30_opex_year = st.number_input("OPEX fisso DC30 (€/anno per colonnina)", min_value=0.0, value=600.0, step=50.0)
        dc30_mnt = st.number_input("Manutenzione annua DC30 (€/a)", min_value=0.0, value=900.0, step=50.0)
        dc30_backend = st.number_input("Backend/CSMS annuo per colonnina DC30 (€/a)", min_value=0.0, value=420.0, step=20.0)

    with st.expander("DC 60 kW (per colonnina)", expanded=True):
        dc60_power = st.number_input("Potenza nominale DC60 (kW)", min_value=40.0, value=60.0, step=5.0)
        dc60_connectors = st.number_input("Connettori per colonnina DC60", min_value=1, value=2, step=1)
        dc60_hw = st.number_input("Hardware DC60 (€)", min_value=10_000.0, value=35_000.0, step=1_000.0)
        dc60_install = st.number_input("Installazione + opere DC60 (€)", min_value=3_000.0, value=16_000.0, step=1_000.0)
        dc60_opex_year = st.number_input("OPEX fisso DC60 (€/anno per colonnina)", min_value=0.0, value=700.0, step=50.0)
        dc60_mnt = st.number_input("Manutenzione annua DC60 (€/a)", min_value=0.0, value=1_100.0, step=50.0)
        dc60_backend = st.number_input("Backend/CSMS annuo per colonnina DC60 (€/a)", min_value=0.0, value=420.0, step=20.0)

    with st.expander("DC 90 kW (per colonnina)", expanded=True):
        dc90_power = st.number_input("Potenza nominale DC90 (kW)", min_value=60.0, value=90.0, step=5.0)
        dc90_connectors = st.number_input("Connettori per colonnina DC90", min_value=1, value=2, step=1)
        dc90_hw = st.number_input("Hardware DC90 (€)", min_value=15_000.0, value=45_000.0, step=1_000.0)
        dc90_install = st.number_input("Installazione + opere DC90 (€)", min_value=4_000.0, value=18_000.0, step=1_000.0)
        dc90_opex_year = st.number_input("OPEX fisso DC90 (€/anno per colonnina)", min_value=0.0, value=800.0, step=50.0)
        dc90_mnt = st.number_input("Manutenzione annua DC90 (€/a)", min_value=0.0, value=1_250.0, step=50.0)
        dc90_backend = st.number_input("Backend/CSMS annuo per colonnina DC90 (€/a)", min_value=0.0, value=420.0, step=20.0)

    st.subheader("🧱 CAPEX extra sito")
    grid_connection_capex = st.number_input(
        "Connessione rete / upgrade / scavi (CAPEX extra)",
        min_value=0.0,
        value=30_000.0,
        step=5_000.0,
        help="Inserisci una stima unica (MVP). In v1 puoi passare a range + Monte Carlo.",
    )
    signage_capex = st.number_input("Segnaletica + stalli dedicati (CAPEX)", min_value=0.0, value=6_000.0, step=500.0)

    st.subheader("🧾 OPEX fissi di sito")
    overhead_opex = st.number_input(
        "Overhead annuo (assicurazioni, pulizia, call center, affitto/royalty)",
        min_value=0.0,
        value=9_000.0,
        step=500.0,
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
        st.markdown("<div class='card'><b>Nota</b><br><span class='muted'>Lo stimatore veicoli/giorno è semplice: (occupazione media × 24) / sosta media. Se hai dati di ingressi reali, puoi inserirli direttamente sotto.</span></div>", unsafe_allow_html=True)
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
        bev_2030 = st.number_input("BEV (Provincia di Trento) target anno base", min_value=0, value=30_000, step=1_000)
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

# -----------------------
# Tabs: Sizing, Finance, Strategy, Data
# -----------------------

sizing_tab, finance_tab, strategy_tab, data_tab = st.tabs(["📐 Sizing", "💼 Business Case", "🧭 Strategia", "🗂️ Dati pubblici"])

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
            start_sessions_day = st.number_input("Sessioni/giorno iniziali (auto che ricaricano)",
        key="sizing5_start_sessions", min_value=0.0, value=float(dres.sessions_per_day), step=1.0)
            share_dc_for_suggest = st.slider("Quota sessioni DC (%)", key="sizing5_share_dc", 0, 100, int(share_sessions_dc*100), step=5) / 100.0
        with c2:
            growth_yoy_suggest = st.slider("Crescita annua domanda (%)", key="sizing5_growth", 0.0, 80.0, 35.0, step=1.0) / 100.0
            years_suggest = st.selectbox("Orizzonte (anni)", key="sizing5_years", [3, 4, 5, 6, 7, 10], index=2)
        with c3:
            objective = st.selectbox("Obiettivo", key="sizing5_objective", ["Massimizza NPV", "Massimizza NPV/Capex"], index=0)
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



st.markdown("---")
st.markdown("### Sizing suggerito (5 anni, crescita domanda)")
with st.expander("Suggerisci configurazione da sessioni iniziali + crescita", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        start_sessions_day = st.number_input(
            "Sessioni/giorno iniziali (auto che ricaricano)",
        key="bc_start_sessions",
            min_value=0.0,
            value=float(getattr(dres, "sessions_per_day", 0.0)),
            step=1.0,
        )
        share_dc_for_suggest = st.slider(
            "Quota sessioni DC (%)",
            0,
            100,
            int(float(share_sessions_dc) * 100),
            step=5,
        ) / 100.0
    with c2:
        growth_yoy_suggest = st.slider("Crescita annua domanda (%)", key="sizing5_growth", 0.0, 80.0, 35.0, step=1.0) / 100.0
        years_suggest = st.selectbox("Orizzonte (anni)", key="sizing5_years", [3, 4, 5, 6, 7, 10], index=2)
    with c3:
        objective = st.selectbox("Obiettivo", key="sizing5_objective", ["Massimizza NPV", "Massimizza NPV/Capex"], index=0)
        capex_budget_chargers = float(max(capex_budget - grid_connection_capex - signage_capex, 0.0))
        st.caption(f"Budget colonnine (CAPEX max - costi sito): {num(capex_budget_chargers,0)} €")

    # Domanda Year 1 derivata da sessioni iniziali (modello semplice: 1 sessione ~ 1 auto che ricarica)
    sessions_ac_day_s = float(start_sessions_day) * (1.0 - float(share_dc_for_suggest))
    sessions_dc_day_s = float(start_sessions_day) * float(share_dc_for_suggest)
    kwh_ac_year1_s = sessions_ac_day_s * float(kwh_per_session_ac) * 365.0
    kwh_dc_year1_s = sessions_dc_day_s * float(kwh_per_session_dc) * 365.0
    tot_kwh_s = max(kwh_ac_year1_s + kwh_dc_year1_s, 1e-9)
    blended_sell_price_s = (float(sell_price_ac) * kwh_ac_year1_s + float(sell_price_dc) * kwh_dc_year1_s) / tot_kwh_s

    # Definizione tecnologie per ottimizzatore (include ore operative)
    ac_cost_s = TechCost(
        name="AC22",
        capex_per_charger=float(ac_hw + ac_install),
        fixed_opex_per_charger_year=float(ac_mnt + ac_backend),
        connectors=int(ac_connectors),
        power_kw=float(ac_power),
        hours_per_day=float(hours_ac),
    )
    dc30_cost_s = TechCost(
        name="DC30",
        capex_per_charger=float(dc30_hw + dc30_install),
        fixed_opex_per_charger_year=float(dc30_mnt + dc30_backend),
        connectors=int(dc30_connectors),
        power_kw=float(dc30_power),
        hours_per_day=float(hours_dc),
    )
    dc60_cost_s = TechCost(
        name="DC60",
        capex_per_charger=float(dc60_hw + dc60_install),
        fixed_opex_per_charger_year=float(dc60_mnt + dc60_backend),
        connectors=int(dc60_connectors),
        power_kw=float(dc60_power),
        hours_per_day=float(hours_dc),
    )
    dc90_cost_s = TechCost(
        name="DC90",
        capex_per_charger=float(dc90_hw + dc90_install),
        fixed_opex_per_charger_year=float(dc90_mnt + dc90_backend),
        connectors=int(dc90_connectors),
        power_kw=float(dc90_power),
        hours_per_day=float(hours_dc),
    )

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

    best_s, all_s = optimize_mix_4tech(
        opt_inp_s,
        ac22=ac_cost_s,
        dc30=dc30_cost_s,
        dc60=dc60_cost_s,
        dc90=dc90_cost_s,
    )

    import pandas as pd
    df_all = pd.DataFrame([r.__dict__ for r in all_s])
    df_all["capex_total"] = df_all["capex"] + float(grid_connection_capex) + float(signage_capex)
    df_all["npv_per_capex"] = df_all["npv"] / df_all["capex_total"].replace({0: float("nan")})

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
    st.write(
        f"**CAPEX colonnine:** {num(best_s.capex,0)} €  |  **CAPEX totale (incl. sito):** {num(best_s.capex + grid_connection_capex + signage_capex,0)} €"
    )
    st.write(
        f"**kWh venduti Year 1:** {num(best_s.kwh_sold_year1,0)}  |  **NPV:** {num(best_s.npv,0)}  |  **IRR:** {pct(best_s.irr)}  |  **Payback:** {num(best_s.payback,1)} anni"
    )
    if str(getattr(best_s, "notes", "")).strip():
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
        cap_kwh_year = (
            kwh_capacity_year(n_ac, ac_power, ac_connectors, uptime, target_util)
            + kwh_capacity_year(n_dc30, dc30_power, dc30_connectors, uptime, target_util)
            + kwh_capacity_year(n_dc60, dc60_power, dc60_connectors, uptime, target_util)
            + kwh_capacity_year(n_dc90, dc90_power, dc90_connectors, uptime, target_util)
        )
        kwh_sold_year1_fin = float(min(demand_kwh_year1, cap_kwh_year))
        lost_kwh_year1 = float(max(0.0, demand_kwh_year1 - cap_kwh_year))

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

        demand_vs_capacity = float(demand_kwh_year1) / max(float(cap_kwh_year), 1e-9)

        st.metric("CAPEX totale", eur(capex))
        st.metric("OPEX fisso anno 1", eur(fixed_opex_year1))
        st.metric("Potenza installata", f"{num(installed_power_kw, 0)} kW")
        st.metric("kWh vendibili (Year 1, a target_util)", num(kwh_sold_year1_fin, 0))

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
            kwh_sold_year1=float(kwh_sold_year1_fin),
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