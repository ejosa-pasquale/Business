# app.py - Trento EV Charging Investment Tool (FLAT, Streamlit Cloud)
from __future__ import annotations

import sys
import pandas as pd
import streamlit as st

# Flat imports (same folder)
from common import fetch_csv, parse_parking_csv, format_eur, format_pct
from trento_chargers import summarize_trento_chargers
from parking_occupancy import estimate_daily_traffic_from_parking
from demand import demand_from_parking_model, demand_from_funnel_model
from sizing import suggest_mix_from_targets, compute_capacity_kwh_per_year
from finance import build_cashflows, npv, irr, payback_year
from optimizer import optimize_mix_bruteforce

st.set_page_config(page_title="Trento EV Charging Investment Tool", page_icon="⚡", layout="wide")

st.sidebar.title("⚡ Trento EV Tool")
st.sidebar.caption("Sizing + ROI + Strategy (AC 22 kW / DC 120 kW)")
st.sidebar.caption(f"Python: {sys.version.split()[0]}")

mode = st.sidebar.radio("Modalità domanda", ["Data-driven (CSV parcheggio)", "Funnel (stima macro)"], index=0)

st.sidebar.divider()
st.sidebar.subheader("Orizzonte & Finanza")
years = st.sidebar.slider("Orizzonte (anni)", 5, 15, 10, 1)
discount_rate = st.sidebar.slider("WACC / tasso sconto (%)", 0.0, 20.0, 8.0, 0.5) / 100.0

st.sidebar.divider()
st.sidebar.subheader("Prezzi energia & vendita")
sell_price_kwh = st.sidebar.number_input("Prezzo vendita (€/kWh)", min_value=0.0, value=0.62, step=0.01)
energy_cost_kwh = st.sidebar.number_input("Costo energia (€/kWh)", min_value=0.0, value=0.26, step=0.01)
roaming_fee_pct = st.sidebar.number_input("Fee roaming/payment (% ricavi)", min_value=0.0, value=6.0, step=0.5) / 100.0

st.sidebar.divider()
st.sidebar.subheader("Affidabilità & target utilizzo")
uptime = st.sidebar.slider("Uptime (%)", 80, 100, 97, 1) / 100.0
target_util = st.sidebar.slider("Target utilizzo medio (%)", 10, 90, 35, 1) / 100.0

st.sidebar.divider()
st.sidebar.subheader("Vincoli sito")
site_power_kw = st.sidebar.number_input("Potenza disponibile sito (kW)", min_value=1.0, value=200.0, step=10.0)
capex_budget = st.sidebar.number_input("CAPEX massimo (opzionale, €)", min_value=0.0, value=0.0, step=1000.0)

st.title("⚡ Tool investimento colonnine EV - Trento")
st.caption("Valuta ROI/NPV/IRR, CAPEX/OPEX e strategia (AC 22kW, DC 120kW).")

tabs = st.tabs(["1) Dati", "2) Domanda", "3) Sizing & Mix", "4) Business Case", "5) Ottimizzatore"])

with tabs[0]:
    st.subheader("Dati pubblici / input")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Colonnine esistenti (upload CSV o URL)")
        chargers_file = st.file_uploader("Upload CSV colonnine (opzionale)", type=["csv"], key="chargers_file")
        chargers_url = st.text_input("URL CSV colonnine (opzionale)", value="", key="chargers_url")

        chargers_df = None
        if chargers_file is not None:
            chargers_df = pd.read_csv(chargers_file)
        elif chargers_url.strip():
            chargers_df = fetch_csv(chargers_url.strip())

        if chargers_df is not None and not chargers_df.empty:
            st.success("CSV colonnine caricato.")
            st.dataframe(chargers_df.head(20), use_container_width=True)
            st.markdown("**Sintesi**")
            st.json(summarize_trento_chargers(chargers_df))

    with c2:
        st.markdown("### Dati parcheggio (CSV) - data-driven")
        parking_file = st.file_uploader("Upload CSV parcheggio (opzionale)", type=["csv"], key="parking_file")
        parking_df = None
        if parking_file is not None:
            raw = pd.read_csv(parking_file)
            parking_df = parse_parking_csv(raw)
            st.success("CSV parcheggio caricato e normalizzato.")
            st.dataframe(parking_df.head(30), use_container_width=True)

    st.session_state["parking_df"] = parking_df

with tabs[1]:
    st.subheader("Domanda stimata (kWh/anno)")
    if mode == "Data-driven (CSV parcheggio)":
        if st.session_state.get("parking_df") is None:
            st.info("Carica un CSV parcheggio nel tab **1) Dati**.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                total_stalls = st.number_input("Posti totali parcheggio", min_value=1, value=400, step=10)
                avg_dwell_h = st.number_input("Durata media sosta (h)", min_value=0.25, value=2.0, step=0.25)
            with c2:
                bev_share = st.number_input("% BEV sul traffico", min_value=0.0, value=6.0, step=0.5) / 100.0
                charge_take_rate = st.number_input("% BEV che ricaricano in sito", min_value=0.0, value=18.0, step=1.0) / 100.0
            with c3:
                kwh_per_session_ac = st.number_input("kWh per sessione AC (media)", min_value=1.0, value=12.0, step=1.0)
                kwh_per_session_dc = st.number_input("kWh per sessione DC (media)", min_value=1.0, value=22.0, step=1.0)

            daily_traffic = estimate_daily_traffic_from_parking(st.session_state["parking_df"], total_stalls, avg_dwell_h)
            demand = demand_from_parking_model(
                daily_traffic=daily_traffic,
                bev_share=bev_share,
                charge_take_rate=charge_take_rate,
                kwh_per_session_ac=kwh_per_session_ac,
                kwh_per_session_dc=kwh_per_session_dc,
                share_dc=0.25,
                days_per_year=365,
            )
            st.metric("Traffico stimato (veicoli/giorno)", f"{daily_traffic:,.0f}".replace(",", "."))
            st.metric("Domanda annua (kWh/anno)", f"{demand['kwh_year']:,.0f}".replace(",", "."))
            st.session_state["demand_kwh_year"] = float(demand["kwh_year"])
            st.json(demand)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            bev_count = st.number_input("BEV area di riferimento (n.)", min_value=0, value=3500, step=100)
        with c2:
            kwh_per_bev_year = st.number_input("kWh/BEV/anno (totali)", min_value=0.0, value=2200.0, step=50.0)
            public_share = st.number_input("% ricarica pubblica", min_value=0.0, value=35.0, step=1.0) / 100.0
        with c3:
            capture = st.number_input("% quota catturata dal sito", min_value=0.0, value=2.0, step=0.1) / 100.0

        demand = demand_from_funnel_model(bev_count, kwh_per_bev_year, public_share, capture)
        st.metric("Domanda annua (kWh/anno)", f"{demand['kwh_year']:,.0f}".replace(",", "."))
        st.session_state["demand_kwh_year"] = float(demand["kwh_year"])
        st.json(demand)

    st.session_state.setdefault("demand_kwh_year", 0.0)

with tabs[2]:
    st.subheader("Sizing & Mix AC/DC")
    demand_kwh_year = float(st.session_state.get("demand_kwh_year", 0.0) or 0.0)
    if demand_kwh_year <= 0:
        st.info("Prima stima la domanda nel tab **2) Domanda**.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            ac_power_kw = st.number_input("Potenza AC per punto (kW)", min_value=1.0, value=22.0, step=1.0)
            ac_capex = st.number_input("CAPEX per punto AC (€)", min_value=0.0, value=6500.0, step=500.0)
            ac_opex_year = st.number_input("OPEX annuo per punto AC (€)", min_value=0.0, value=450.0, step=50.0)
        with c2:
            dc_power_kw = st.number_input("Potenza DC per punto (kW)", min_value=10.0, value=120.0, step=10.0)
            dc_capex = st.number_input("CAPEX per punto DC (€)", min_value=0.0, value=65000.0, step=1000.0)
            dc_opex_year = st.number_input("OPEX annuo per punto DC (€)", min_value=0.0, value=2500.0, step=100.0)
        with c3:
            share_dc = st.slider("Quota domanda su DC (%)", 0, 80, 25, 1) / 100.0
            st.write(f"Target utilizzo medio: **{format_pct(target_util)}**")
            st.write(f"Uptime: **{format_pct(uptime)}**")

        sizing = suggest_mix_from_targets(
            demand_kwh_year=demand_kwh_year,
            share_dc=share_dc,
            ac_power_kw=ac_power_kw,
            dc_power_kw=dc_power_kw,
            uptime=uptime,
            target_util=target_util,
            site_power_kw=site_power_kw,
        )

        st.markdown("### Risultato sizing")
        colA, colB, colC = st.columns(3)
        colA.metric("Punti AC", f"{sizing['n_ac']}")
        colB.metric("Punti DC", f"{sizing['n_dc']}")
        colC.metric("Potenza richiesta (kW)", f"{sizing['power_required_kw']:.0f}")
        st.json(sizing)

        cap_ac = compute_capacity_kwh_per_year(ac_power_kw, uptime, target_util) * sizing["n_ac"]
        cap_dc = compute_capacity_kwh_per_year(dc_power_kw, uptime, target_util) * sizing["n_dc"]
        st.markdown("### Capacità annua a target utilizzo")
        st.write(f"- AC: **{cap_ac:,.0f} kWh/anno**
- DC: **{cap_dc:,.0f} kWh/anno**
- Totale: **{(cap_ac+cap_dc):,.0f} kWh/anno**")

        st.session_state["config"] = {
            "n_ac": sizing["n_ac"],
            "n_dc": sizing["n_dc"],
            "ac_power_kw": ac_power_kw,
            "dc_power_kw": dc_power_kw,
            "ac_capex": ac_capex,
            "dc_capex": dc_capex,
            "ac_opex_year": ac_opex_year,
            "dc_opex_year": dc_opex_year,
            "share_dc": share_dc,
        }

with tabs[3]:
    st.subheader("Business Case (CAPEX/OPEX/ROI)")
    demand_kwh_year = float(st.session_state.get("demand_kwh_year", 0.0) or 0.0)
    cfg = st.session_state.get("config")

    if demand_kwh_year <= 0 or not cfg:
        st.info("Completa **2) Domanda** e **3) Sizing & Mix**.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            fixed_capex = st.number_input("CAPEX fisso (opere civili/connessione) €", min_value=0.0, value=25000.0, step=1000.0)
        with c2:
            fixed_opex = st.number_input("OPEX fisso annuo (CSMS, connettività, assicurazione) €", min_value=0.0, value=6000.0, step=500.0)
        with c3:
            annual_growth_kwh = st.number_input("Crescita annua domanda (%)", min_value=-20.0, value=8.0, step=1.0) / 100.0

        capex_total = cfg["n_ac"] * cfg["ac_capex"] + cfg["n_dc"] * cfg["dc_capex"] + fixed_capex

        cash = build_cashflows(
            years=years,
            demand_kwh_year=demand_kwh_year,
            annual_growth_kwh=annual_growth_kwh,
            sell_price_kwh=sell_price_kwh,
            energy_cost_kwh=energy_cost_kwh,
            roaming_fee_pct=roaming_fee_pct,
            n_ac=cfg["n_ac"],
            n_dc=cfg["n_dc"],
            opex_ac_year=cfg["ac_opex_year"],
            opex_dc_year=cfg["dc_opex_year"],
            fixed_opex_year=fixed_opex,
            capex_total=capex_total,
        )

        npv_val = npv(cash["net_cashflow"], discount_rate)
        irr_val = irr(cash["net_cashflow"])
        pb = payback_year(cash["net_cashflow"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("CAPEX totale", format_eur(capex_total))
        k2.metric("NPV", format_eur(npv_val))
        k3.metric("IRR", f"{(irr_val*100):.1f}%" if irr_val is not None else "n/a")
        k4.metric("Payback", f"{pb:.1f} anni" if pb is not None else "n/a")

        df = cash["table"]
        st.dataframe(df, use_container_width=True)
        st.download_button("Scarica cashflow CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="cashflow.csv", mime="text/csv")

with tabs[4]:
    st.subheader("Ottimizzatore mix (NPV massimo)")
    demand_kwh_year = float(st.session_state.get("demand_kwh_year", 0.0) or 0.0)
    if demand_kwh_year <= 0:
        st.info("Stima prima la domanda nel tab **2) Domanda**.")
    else:
        st.write("Cerca il miglior mix AC/DC con vincoli di potenza e (opzionalmente) budget.")
        c1, c2, c3 = st.columns(3)
        with c1:
            ac_power_kw = st.number_input("AC kW", min_value=1.0, value=22.0, step=1.0, key="opt_ac_kw")
            ac_capex = st.number_input("CAPEX AC (€)", min_value=0.0, value=6500.0, step=500.0, key="opt_ac_capex")
            ac_opex = st.number_input("OPEX AC annuo (€)", min_value=0.0, value=450.0, step=50.0, key="opt_ac_opex")
        with c2:
            dc_power_kw = st.number_input("DC kW", min_value=10.0, value=120.0, step=10.0, key="opt_dc_kw")
            dc_capex = st.number_input("CAPEX DC (€)", min_value=0.0, value=65000.0, step=1000.0, key="opt_dc_capex")
            dc_opex = st.number_input("OPEX DC annuo (€)", min_value=0.0, value=2500.0, step=100.0, key="opt_dc_opex")
        with c3:
            fixed_capex = st.number_input("CAPEX fisso (€)", min_value=0.0, value=25000.0, step=1000.0, key="opt_fixed_capex")
            fixed_opex = st.number_input("OPEX fisso annuo (€)", min_value=0.0, value=6000.0, step=500.0, key="opt_fixed_opex")
            annual_growth_kwh = st.number_input("Crescita annua domanda (%)", min_value=-20.0, value=8.0, step=1.0, key="opt_growth") / 100.0

        c4, c5, c6 = st.columns(3)
        with c4:
            max_ac = st.slider("Max punti AC", 0, 40, 16, 1)
        with c5:
            max_dc = st.slider("Max punti DC", 0, 10, 3, 1)
        with c6:
            share_dc = st.slider("Quota domanda su DC (%)", 0, 80, 25, 1) / 100.0

        if st.button("Esegui ottimizzazione"):
            res = optimize_mix_bruteforce(
                years=years,
                discount_rate=discount_rate,
                demand_kwh_year=demand_kwh_year,
                annual_growth_kwh=annual_growth_kwh,
                sell_price_kwh=sell_price_kwh,
                energy_cost_kwh=energy_cost_kwh,
                roaming_fee_pct=roaming_fee_pct,
                ac_power_kw=ac_power_kw,
                dc_power_kw=dc_power_kw,
                ac_capex=ac_capex,
                dc_capex=dc_capex,
                ac_opex_year=ac_opex,
                dc_opex_year=dc_opex,
                fixed_capex=fixed_capex,
                fixed_opex_year=fixed_opex,
                uptime=uptime,
                target_util=target_util,
                share_dc=share_dc,
                site_power_kw=site_power_kw,
                capex_budget=(capex_budget if capex_budget > 0 else None),
                max_ac=max_ac,
                max_dc=max_dc,
            )
            if res is None:
                st.warning("Nessuna combinazione soddisfa i vincoli (potenza/budget) e la copertura domanda.")
            else:
                st.success("Miglior mix trovato.")
                st.json(res["best"])
                st.dataframe(res["top_table"], use_container_width=True)
                st.download_button("Scarica top risultati CSV", data=res["top_table"].to_csv(index=False).encode("utf-8"), file_name="optimizer_top_results.csv", mime="text/csv")
