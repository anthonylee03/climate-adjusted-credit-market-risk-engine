"""
Climate-Adjusted Credit & Market Risk Engine
---------------------------------------------
PIK/NGFS scenario data (temperature pathway, carbon price) as input to a
firm/portfolio-level PD, LGD, and Climate-VaR calculator.

Module map:
  1. NGFS/PIK Scenario Parser      -> SCENARIOS dict
  2. Structural Credit Risk Model  -> calculate_merton_pd, single_firm_shock
  3. Portfolio Climate-VaR Calc.   -> run_portfolio_montecarlo, portfolio_risk_metrics
  4. Interactive Dashboard         -> Streamlit UI below (sidebar + tabs)

Author: Seungmin Lee (HUFS, Public Administration & Economics)
License: MIT
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Climate-Adjusted Credit & Market Risk Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.25rem; }
        .sub-header { font-size: 1.05rem; color: #6c757d; margin-bottom: 1.5rem; }
        .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 15px;
                       border-left: 5px solid #0d6efd; }
        .footnote { color: #9aa1a9; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# NGFS / PIK SCENARIO LIBRARY
# -----------------------------------------------------------------------------
# Carbon price ($/tCO2) and temperature pathway are illustrative stand-ins for
# NGFS Phase IV scenario outputs, calibrated to be directionally consistent
# with published NGFS scenario explorer ranges (not a live data feed).
SCENARIOS = {
    "Net Zero 2050 (Orderly)": {"carbon_price": 150.0, "temp_rise": 1.5,
                                 "desc": "Early, coordinated transition. Carbon price rises smoothly; physical risk stays contained."},
    "Delayed Transition (Disorderly)": {"carbon_price": 280.0, "temp_rise": 1.8,
                                         "desc": "Action delayed to ~2030 then tightened abruptly. Highest transition-risk shock."},
    "Current Policies (Hot House World)": {"carbon_price": 10.0, "temp_rise": 3.5,
                                            "desc": "No new policy. Minimal transition risk, severe physical risk."},
    "Custom Setup": {"carbon_price": 80.0, "temp_rise": 2.0,
                      "desc": "Manually set carbon price and temperature pathway with the sliders below."},
}

# -----------------------------------------------------------------------------
# FIRM LIBRARY (illustrative preset companies, one per sector)
# -----------------------------------------------------------------------------
# Calibrated to be directionally representative of each sector's typical
# balance-sheet size, leverage, and emissions intensity -- not real company
# financials. Use "Custom" to enter your own numbers instead.
FIRM_LIBRARY = {
    "Custom": None,
    "Integrated Steel Producer": {"v0": 1000.0, "debt": 700.0, "sigma_v": 0.28,
                                   "emissions": 4200.0, "pass_through": 0.20, "base_lgd": 0.50},
    "Oil & Gas Major": {"v0": 5000.0, "debt": 2800.0, "sigma_v": 0.30,
                         "emissions": 9500.0, "pass_through": 0.35, "base_lgd": 0.45},
    "Fossil Power Utility": {"v0": 2200.0, "debt": 1500.0, "sigma_v": 0.20,
                              "emissions": 6800.0, "pass_through": 0.55, "base_lgd": 0.40},
    "Clean Energy Tech": {"v0": 800.0, "debt": 300.0, "sigma_v": 0.35,
                           "emissions": 120.0, "pass_through": 0.10, "base_lgd": 0.35},
    "Commercial Real Estate REIT": {"v0": 1500.0, "debt": 950.0, "sigma_v": 0.18,
                                     "emissions": 300.0, "pass_through": 0.15, "base_lgd": 0.55},
}

# -----------------------------------------------------------------------------
# CORE MODELS
# -----------------------------------------------------------------------------
def calculate_merton_pd(V, D, r, sigma_V, T=1.0):
    """Merton structural model: Distance to Default (DD) and Probability of Default (PD)."""
    if V <= 0 or D <= 0 or sigma_V <= 0:
        return 0.0, 1.0
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    dd = d2
    pd_ = norm.cdf(-dd)
    return float(dd), float(pd_)


def physical_damage_function(temp_increase):
    """PIK-style non-linear physical damage factor, capped at 50% of asset value."""
    damage_factor = 0.02 * (temp_increase**2) + 0.01 * temp_increase
    return min(damage_factor, 0.5)


def single_firm_shock(v0, debt, sigma_v, r, emissions, pass_through, base_lgd,
                       carbon_price, temp_rise):
    """Runs baseline vs. climate-shocked Merton metrics for one firm."""
    effective_carbon_cost = emissions * 1000 * carbon_price * (1.0 - pass_through) / 1e6  # $M
    v0_shocked = max(v0 - effective_carbon_cost, 1.0)

    dd_base, pd_base = calculate_merton_pd(v0, debt, r, sigma_v)
    dd_shocked, pd_shocked = calculate_merton_pd(v0_shocked, debt, r, sigma_v)

    phys_damage = physical_damage_function(temp_rise)
    lgd_adjusted = min(base_lgd + (1 - base_lgd) * phys_damage, 1.0)

    el_base = v0 * pd_base * base_lgd
    el_shocked = v0_shocked * pd_shocked * lgd_adjusted

    return {
        "v0_shocked": v0_shocked, "dd_base": dd_base, "pd_base": pd_base,
        "dd_shocked": dd_shocked, "pd_shocked": pd_shocked,
        "lgd_adjusted": lgd_adjusted, "el_base": el_base, "el_shocked": el_shocked,
    }


def nearest_psd_matrix(matrix):
    """Clips negative eigenvalues so a (possibly hand-edited) correlation /
    covariance matrix is usable by multivariate_normal."""
    matrix = (matrix + matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


@st.cache_data(show_spinner=False)
def run_portfolio_montecarlo(weights, base_vals, vols, c_scores, carbon_price,
                              temp_rise, corr_matrix, n_sims, seed):
    """Vectorized Monte Carlo of portfolio value under market + climate shocks.

    corr_matrix: flattened tuple of a full NxN correlation matrix (diag = 1),
    built either from the average-correlation slider or hand-edited by the user.
    """
    n_assets = len(weights)
    rng = np.random.default_rng(seed)

    corr = np.array(corr_matrix).reshape(n_assets, n_assets)
    vols_arr = np.array(vols)
    base_vals_arr = np.array(base_vals)
    c_scores_arr = np.array(c_scores)
    cov_matrix = corr * np.outer(vols_arr, vols_arr)
    # Guard against a non-positive-semidefinite matrix from hand-edited correlations.
    cov_matrix = nearest_psd_matrix(cov_matrix)

    market_shocks = rng.multivariate_normal(np.zeros(n_assets), cov_matrix, size=n_sims)

    transition_penalty = (carbon_price / 300.0) * c_scores_arr * 0.15
    physical_penalty = (temp_rise / 4.5) * 0.10
    total_climate_penalty = transition_penalty + physical_penalty

    simulated_returns = market_shocks - total_climate_penalty
    simulated_asset_values = base_vals_arr * (1 + simulated_returns)
    simulated_portfolio_values = np.sum(simulated_asset_values, axis=1)

    total_portfolio_value = np.sum(base_vals)
    portfolio_losses = total_portfolio_value - simulated_portfolio_values
    return portfolio_losses, total_portfolio_value


def portfolio_risk_metrics(portfolio_losses, total_portfolio_value):
    var_95 = np.percentile(portfolio_losses, 95)
    var_99 = np.percentile(portfolio_losses, 99)
    tail_95 = portfolio_losses[portfolio_losses >= var_95]
    es_95 = tail_95.mean() if len(tail_95) > 0 else var_95
    return var_95, var_99, es_95, total_portfolio_value


def generate_executive_report(scenario_choice, carbon_price, temp_rise, firm_choice,
                               firm_result, dd_base, pd_base, base_lgd,
                               portfolio_metrics, firm_scenario_df, portfolio_scenario_df):
    """Builds a consulting/central-bank-style Markdown summary from the current
    session's inputs and outputs, mirroring the auto-generated report pattern
    used in the companion CTVI project."""
    var_95, var_99, es_95, tpv = portfolio_metrics
    lines = [
        "# Climate-Adjusted Credit & Market Risk Engine — Executive Report",
        "",
        f"**Active scenario:** {scenario_choice}  ",
        f"**Carbon price:** ${carbon_price:.0f}/tCO2  |  **Temperature rise:** {temp_rise:.1f}°C",
        "",
        "## 1. Single-Firm Credit Risk",
        f"**Firm:** {firm_choice}",
        "",
        "| Metric | Baseline | Climate-Shocked | Change |",
        "|---|---:|---:|---:|",
        f"| Distance to Default | {dd_base:.2f} | {firm_result['dd_shocked']:.2f} | {firm_result['dd_shocked']-dd_base:+.2f} |",
        f"| Probability of Default | {pd_base*100:.2f}% | {firm_result['pd_shocked']*100:.2f}% | {(firm_result['pd_shocked']-pd_base)*100:+.2f}pp |",
        f"| LGD | {base_lgd*100:.1f}% | {firm_result['lgd_adjusted']*100:.1f}% | {(firm_result['lgd_adjusted']-base_lgd)*100:+.1f}pp |",
        f"| Expected Loss ($M) | {firm_result['el_base']:.2f} | {firm_result['el_shocked']:.2f} | {firm_result['el_shocked']-firm_result['el_base']:+.2f} |",
        "",
        "## 2. Portfolio Climate-VaR",
    ]
    if var_95 is not None:
        lines += [
            f"- Total portfolio value: ${tpv:.1f}M",
            f"- 95% Climate-VaR (1-year): ${var_95:.2f}M ({var_95/tpv*100:.1f}% of book)",
            f"- 99% Climate-VaR (1-year): ${var_99:.2f}M ({var_99/tpv*100:.1f}% of book)",
            f"- 95% Expected Shortfall: ${es_95:.2f}M",
            "",
        ]
    else:
        lines += ["- Not available: add at least one portfolio row with positive volatility in the "
                   "Portfolio Climate-VaR tab.", ""]

    lines += ["## 3. Scenario Comparison (Single Firm)", ""]
    if firm_scenario_df is not None and len(firm_scenario_df) > 0:
        lines.append(firm_scenario_df.to_markdown(index=False))
    else:
        lines.append("Not available.")
    lines += ["", "## 4. Scenario Comparison (Portfolio Climate-VaR)", ""]
    if portfolio_scenario_df is not None and len(portfolio_scenario_df) > 0:
        lines.append(portfolio_scenario_df.to_markdown(index=False))
    else:
        lines.append("Not available -- set up the portfolio tab first.")

    lines += [
        "",
        "## 5. Notes & Limitations",
        "- Carbon price / temperature pathways are illustrative, directionally aligned with NGFS "
        "scenario narratives; not a live NGFS Scenario Explorer feed.",
        "- Portfolio correlation reflects whatever matrix was set in the Portfolio Climate-VaR tab "
        "(default: single average-correlation assumption unless hand-edited).",
        "- Teaching / portfolio-demo tool for scenario-based stress testing; not investment or credit advice.",
        "",
        "*Generated by the Climate-Adjusted Credit & Market Risk Engine.*",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# SIDEBAR: SCENARIO & PARAMETER CONTROL
# -----------------------------------------------------------------------------
st.sidebar.title("🌿 PIK / NGFS Scenario Setup")

scenario_choice = st.sidebar.selectbox("Select Climate Scenario", list(SCENARIOS.keys()))
st.sidebar.caption(SCENARIOS[scenario_choice]["desc"])

default_carbon_price = SCENARIOS[scenario_choice]["carbon_price"]
default_temp = SCENARIOS[scenario_choice]["temp_rise"]

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Scenario Parameters")
carbon_price = st.sidebar.slider("Carbon Tax Rate ($/ton)", 0.0, 350.0, float(default_carbon_price), step=10.0)
temp_rise = st.sidebar.slider("Global Temp Rise (°C)", 0.5, 4.5, float(default_temp), step=0.1)
risk_free_rate = st.sidebar.slider("Risk-free Interest Rate (%)", 0.0, 10.0, 3.5, step=0.1) / 100.0

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Monte Carlo Setup")
mc_sims = st.sidebar.selectbox("Number of Simulations", [1000, 5000, 10000, 25000], index=1)
avg_corr = st.sidebar.slider("Default Cross-Asset Correlation", 0.0, 0.9, 0.30, step=0.05,
                              help="Fills the off-diagonal of the portfolio correlation matrix. "
                                   "Edit individual pairs directly in the Portfolio Climate-VaR tab.")
mc_seed = st.sidebar.number_input("Random Seed", value=42, step=1,
                                   help="Change this to see result stability across draws.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model notes: carbon prices and temperature pathways are illustrative "
    "and directionally aligned with NGFS scenario narratives, not a live "
    "data feed from the NGFS Scenario Explorer. Treat outputs as a stress-"
    "testing teaching tool, not investment or credit advice."
)

# -----------------------------------------------------------------------------
# DASHBOARD HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Climate-Adjusted Credit &amp; Market Risk Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">PIK/NGFS scenario-based portfolio climate stress testing and '
    'Climate-VaR measurement</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Single Firm Credit Stress Test",
    "💼 Portfolio Climate-VaR",
    "🔀 Scenario Comparison",
    "📖 Methodology",
    "📄 Executive Report",
])

# -----------------------------------------------------------------------------
# TAB 1: SINGLE FIRM MERTON MODEL
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Structural Credit Risk Analysis (Merton Model)")

    col_input, col_metrics = st.columns([1, 2])

    with col_input:
        firm_choice = st.selectbox("Load a preset company (or keep Custom)", list(FIRM_LIBRARY.keys()))
        preset = FIRM_LIBRARY[firm_choice]
        if preset:
            st.caption(f"Preset loaded: {firm_choice}. Sliders below start from its values -- adjust freely.")

        st.markdown("**Corporate Baseline Financials**")
        v0 = st.number_input("Firm Asset Value V0 ($M)", value=preset["v0"] if preset else 1000.0,
                              step=50.0, min_value=1.0, key=f"v0_{firm_choice}")
        debt = st.number_input("Total Debt / Default Barrier D ($M)", value=preset["debt"] if preset else 700.0,
                                step=50.0, min_value=1.0, key=f"debt_{firm_choice}")
        sigma_v = st.slider("Asset Volatility σ (%)", 5.0, 80.0,
                             (preset["sigma_v"] * 100) if preset else 25.0,
                             key=f"sigma_{firm_choice}") / 100.0

        st.markdown("**Emissions Profile**")
        emissions = st.number_input("Annual Scope 1+2 Emissions (k-tons CO2)",
                                     value=preset["emissions"] if preset else 500.0,
                                     step=50.0, min_value=0.0, key=f"em_{firm_choice}")
        pass_through = st.slider("Cost Pass-through to Customers (%)", 0.0, 100.0,
                                  (preset["pass_through"] * 100) if preset else 30.0,
                                  key=f"pt_{firm_choice}") / 100.0
        base_lgd = st.slider("Baseline LGD (%)", 10.0, 90.0,
                              (preset["base_lgd"] * 100) if preset else 45.0,
                              key=f"lgd_{firm_choice}") / 100.0

    result = single_firm_shock(v0, debt, sigma_v, risk_free_rate, emissions,
                                pass_through, base_lgd, carbon_price, temp_rise)
    dd_base, pd_base = calculate_merton_pd(v0, debt, risk_free_rate, sigma_v)

    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Distance to Default (DD)", f"{result['dd_shocked']:.2f}",
                   f"{result['dd_shocked'] - dd_base:.2f}", delta_color="normal")
        m2.metric("Probability of Default (PD)", f"{result['pd_shocked']*100:.2f}%",
                   f"{(result['pd_shocked'] - pd_base)*100:+.2f}%p", delta_color="inverse")
        m3.metric("Adjusted LGD", f"{result['lgd_adjusted']*100:.1f}%",
                   f"+{(result['lgd_adjusted'] - base_lgd)*100:.1f}%p", delta_color="inverse")
        m4.metric("Expected Loss (EL)", f"${result['el_shocked']:.2f}M",
                   f"${result['el_shocked'] - result['el_base']:+.2f}M", delta_color="inverse")

        st.markdown("---")

        v0_shocked = result["v0_shocked"]
        x_vals = np.linspace(min(v0_shocked * 0.4, debt * 0.5), max(v0 * 1.5, debt * 1.5), 500)
        pdf_base = norm.pdf(np.log(x_vals / v0), loc=(risk_free_rate - 0.5 * sigma_v**2), scale=sigma_v)
        pdf_shocked = norm.pdf(np.log(x_vals / v0_shocked), loc=(risk_free_rate - 0.5 * sigma_v**2), scale=sigma_v)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=pdf_base, mode="lines", name="Baseline Asset Distribution",
                                  line=dict(color="blue", dash="dash")))
        fig.add_trace(go.Scatter(x=x_vals, y=pdf_shocked, mode="lines", name="Climate-Shocked Asset Distribution",
                                  line=dict(color="red")))
        fig.add_vline(x=debt, line_width=2, line_dash="solid", line_color="black",
                       annotation_text="Default Barrier (Debt D)")
        fig.update_layout(title="Firm Asset Value Distribution vs Default Barrier",
                           xaxis_title="Asset Value V_T ($M)", yaxis_title="Probability Density",
                           template="plotly_white", height=350)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("**Sensitivity: PD response to carbon price and temperature rise**")
    st.caption("Holds all other inputs fixed at the values above; shows how PD moves as each driver is scanned independently.")

    cp_range = np.linspace(0, 350, 15)
    pd_vs_carbon = [
        single_firm_shock(v0, debt, sigma_v, risk_free_rate, emissions, pass_through,
                           base_lgd, cp, temp_rise)["pd_shocked"] * 100
        for cp in cp_range
    ]
    temp_range = np.linspace(0.5, 4.5, 15)
    pd_vs_temp = [
        single_firm_shock(v0, debt, sigma_v, risk_free_rate, emissions, pass_through,
                           base_lgd, carbon_price, tr)["pd_shocked"] * 100
        for tr in temp_range
    ]

    sens_col1, sens_col2 = st.columns(2)
    with sens_col1:
        fig_cp = px.line(x=cp_range, y=pd_vs_carbon, markers=True,
                          labels={"x": "Carbon Price ($/ton)", "y": "PD (%)"},
                          title="PD Sensitivity to Carbon Price")
        fig_cp.add_vline(x=carbon_price, line_dash="dot", line_color="orange")
        fig_cp.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_cp, width="stretch")
    with sens_col2:
        fig_tp = px.line(x=temp_range, y=pd_vs_temp, markers=True,
                          labels={"x": "Temp Rise (°C)", "y": "PD (%)"},
                          title="PD Sensitivity to Temperature Rise (via LGD channel is excluded here; PD itself is temp-invariant in this model)")
        fig_tp.add_vline(x=temp_rise, line_dash="dot", line_color="orange")
        fig_tp.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_tp, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: PORTFOLIO CLIMATE-VAR
# -----------------------------------------------------------------------------
corr_matrix = None  # populated inside tab2 below; reused by the scenario comparison and report tabs
var_95 = var_99 = es_95 = total_portfolio_value = None  # populated inside tab2 below; reused by the report tab

with tab2:
    st.subheader("Portfolio Climate Value at Risk (Climate-VaR) Simulator")

    if "portfolio_data" not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame({
            "Asset Sector": ["Clean Energy Tech", "Steel & Heavy Industry", "Fossil Energy Utility",
                              "Commercial Real Estate", "Sovereign/Tech"],
            "Weight (%)": [20.0, 25.0, 15.0, 20.0, 20.0],
            "Base Value ($M)": [200.0, 250.0, 150.0, 200.0, 200.0],
            "Annual Return Vol (%)": [22.0, 18.0, 25.0, 12.0, 15.0],
            "Carbon Intensity Score": [0.1, 0.9, 1.0, 0.4, 0.2],
        })

    st.markdown("**Corporate Portfolio Holdings Structure**")
    edited_df = st.data_editor(st.session_state.portfolio_data, num_rows="dynamic", key="portfolio_editor")

    weight_sum = edited_df["Weight (%)"].sum()
    if abs(weight_sum - 100.0) > 0.5:
        st.warning(f"Weights sum to {weight_sum:.1f}%, not 100%. Value-weighted results below use "
                   f"'Base Value ($M)' directly, but check your weights if they're meant to describe the book.")

    weights = edited_df["Weight (%)"].values / 100.0
    base_vals = edited_df["Base Value ($M)"].values.astype(float)
    vols = edited_df["Annual Return Vol (%)"].values.astype(float) / 100.0
    c_scores = edited_df["Carbon Intensity Score"].values.astype(float)
    sectors = edited_df["Asset Sector"].astype(str).tolist()

    if len(base_vals) == 0 or np.any(vols <= 0):
        st.error("Add at least one row with a positive volatility to run the simulation.")
    else:
        st.markdown("**Cross-Asset Correlation Matrix**")
        st.caption("Initialized from the sidebar's default correlation; edit individual pairs to model "
                   "sector-specific co-movement instead of a single average assumption.")

        corr_key = "corr_matrix_" + "|".join(sectors)
        if corr_key not in st.session_state:
            init_corr = np.full((len(sectors), len(sectors)), avg_corr)
            np.fill_diagonal(init_corr, 1.0)
            st.session_state[corr_key] = pd.DataFrame(init_corr, index=sectors, columns=sectors)

        corr_df = st.data_editor(
            st.session_state[corr_key], key=f"corr_editor_{corr_key}",
            column_config={s: st.column_config.NumberColumn(s, min_value=-1.0, max_value=1.0, step=0.05)
                           for s in sectors},
        )
        corr_matrix = corr_df.values.astype(float)
        np.fill_diagonal(corr_matrix, 1.0)  # diagonal is always 1, regardless of edits

        portfolio_losses, total_portfolio_value = run_portfolio_montecarlo(
            tuple(weights), tuple(base_vals), tuple(vols), tuple(c_scores),
            carbon_price, temp_rise, tuple(corr_matrix.flatten()), mc_sims, mc_seed,
        )
        var_95, var_99, es_95, total_portfolio_value = portfolio_risk_metrics(
            portfolio_losses, total_portfolio_value
        )

        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("Total Portfolio Value", f"${total_portfolio_value:.1f}M")
        col_v2.metric("95% Climate-VaR (1-Year)", f"${var_95:.2f}M",
                       f"{(var_95/total_portfolio_value)*100:.1f}% of book")
        col_v3.metric("95% Expected Shortfall (ES)", f"${es_95:.2f}M",
                       f"{(es_95/total_portfolio_value)*100:.1f}% of book")

        fig_hist = px.histogram(
            portfolio_losses, nbins=60,
            title="Monte Carlo Portfolio Loss Distribution under Climate Shock",
            labels={"value": "Loss Amount ($M)"}, color_discrete_sequence=["#457b9d"],
        )
        fig_hist.add_vline(x=var_95, line_color="orange", line_dash="dash", annotation_text=f"VaR 95%: ${var_95:.1f}M")
        fig_hist.add_vline(x=var_99, line_color="red", line_dash="solid", annotation_text=f"VaR 99%: ${var_99:.1f}M")
        fig_hist.update_layout(showlegend=False, template="plotly_white", height=400)
        st.plotly_chart(fig_hist, width="stretch")

        results_df = pd.DataFrame({"Simulated Portfolio Loss ($M)": portfolio_losses})
        st.download_button(
            "⬇️ Download simulation results (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="climate_var_simulation.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# TAB 3: SCENARIO COMPARISON (across the 3 NGFS presets, at once)
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("NGFS Scenario Comparison")
    st.caption("Re-runs the single-firm and portfolio models across all three NGFS presets, "
               "holding every other input fixed at the values set in the sidebar (except carbon "
               "price / temp rise, which are swapped per scenario).")

    compare_scenarios = {k: v for k, v in SCENARIOS.items() if k != "Custom Setup"}
    firm_rows = []
    var_rows = []
    var_df = None

    for name, params in compare_scenarios.items():
        r = single_firm_shock(v0, debt, sigma_v, risk_free_rate, emissions,
                               pass_through, base_lgd, params["carbon_price"], params["temp_rise"])
        firm_rows.append({
            "Scenario": name, "Carbon Price ($/t)": params["carbon_price"],
            "Temp Rise (°C)": params["temp_rise"], "PD (%)": round(r["pd_shocked"] * 100, 2),
            "Adjusted LGD (%)": round(r["lgd_adjusted"] * 100, 1),
            "Expected Loss ($M)": round(r["el_shocked"], 2),
        })

        if len(base_vals) > 0 and np.all(vols > 0) and corr_matrix is not None:
            losses, tpv = run_portfolio_montecarlo(
                tuple(weights), tuple(base_vals), tuple(vols), tuple(c_scores),
                params["carbon_price"], params["temp_rise"], tuple(corr_matrix.flatten()), mc_sims, mc_seed,
            )
            v95, v99, es95, tpv = portfolio_risk_metrics(losses, tpv)
            var_rows.append({"Scenario": name, "95% Climate-VaR ($M)": round(v95, 2),
                              "95% ES ($M)": round(es95, 2), "% of Book (VaR95)": round(v95 / tpv * 100, 1)})

    st.markdown("**Single-Firm Credit Risk Across Scenarios**")
    firm_df = pd.DataFrame(firm_rows)
    st.dataframe(firm_df, width="stretch", hide_index=True)
    fig_firm = px.bar(firm_df, x="Scenario", y="Expected Loss ($M)", color="Scenario",
                       title="Expected Loss by NGFS Scenario (Single Firm)")
    fig_firm.update_layout(template="plotly_white", height=350, showlegend=False)
    st.plotly_chart(fig_firm, width="stretch")

    if var_rows:
        st.markdown("**Portfolio Climate-VaR Across Scenarios**")
        var_df = pd.DataFrame(var_rows)
        st.dataframe(var_df, width="stretch", hide_index=True)
        fig_var = px.bar(var_df, x="Scenario", y="95% Climate-VaR ($M)", color="Scenario",
                          title="Portfolio 95% Climate-VaR by NGFS Scenario")
        fig_var.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig_var, width="stretch")

# -----------------------------------------------------------------------------
# TAB 4: METHODOLOGY & PIK FRAMEWORK
# -----------------------------------------------------------------------------
with tab4:
    st.markdown(
        r"""
### 📖 Academic Framework & Mathematical Foundation

This application bridges macroeconomic climate scenario narratives developed at the
**Potsdam Institute for Climate Impact Research (PIK)** / **NGFS** with quantitative
**Financial Risk Management (FRM)** tools.

---

#### 1. Transition Risk via Merton Structural Default Model
Equity is treated as a call option on firm assets $V$ with strike price equal to debt $D$.

* **Carbon cost shock to firm value:**
$$\Delta V = - \text{Carbon Price} \times \text{CO}_2 \text{ Emissions} \times (1 - \text{Pass-Through})$$
* **Distance to Default:**
$$DD = \frac{\ln(V_0 / D) + (r - 0.5\sigma_V^2)T}{\sigma_V \sqrt{T}}$$
* **Probability of Default:**
$$PD = N(-DD)$$

A higher carbon tax reduces $V_0$, shrinking DD and raising PD non-linearly.

---

#### 2. Physical Risk via Loss Given Default (LGD)
Physical impacts (extreme weather, capital destruction) scale LGD through a PIK-style
non-linear damage function:
$$\text{LGD}_{adjusted} = \text{LGD}_{base} + (1 - \text{LGD}_{base}) \times \Omega(T_{rise})$$

where $\Omega(T_{rise}) = \min(0.02\,T^2 + 0.01\,T,\ 0.5)$.

---

#### 3. Portfolio Climate Value at Risk (Climate-VaR)
Monte Carlo simulation draws correlated market shocks (mean-zero multivariate normal,
covariance built from per-asset volatility and an average correlation assumption), then
layers on a deterministic, sector-specific climate penalty:

$$\text{penalty}_i = \underbrace{\frac{\text{Carbon Price}}{300} \times \text{CarbonIntensity}_i \times 0.15}_{\text{transition}} + \underbrace{\frac{T_{rise}}{4.5} \times 0.10}_{\text{physical}}$$

95th/99th percentile losses (VaR) and the mean loss beyond VaR95 (Expected Shortfall) are
read off the simulated loss distribution.

---

#### Known limitations
- Carbon price / temperature pathways are illustrative stand-ins for NGFS scenario
  outputs, not pulled live from the NGFS Scenario Explorer.
- The correlation structure is a single average-correlation assumption, not an
  estimated covariance matrix.
- The model is a teaching/demo tool for scenario-based stress testing, not a
  production credit or market risk model.
        """
    )

# -----------------------------------------------------------------------------
# TAB 5: EXECUTIVE REPORT
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Auto-Generated Executive Report")
    st.caption("Assembles the current session's single-firm, portfolio, and scenario-comparison "
               "results into one downloadable Markdown report.")

    report_md = generate_executive_report(
        scenario_choice, carbon_price, temp_rise, firm_choice, result, dd_base, pd_base, base_lgd,
        (var_95, var_99, es_95, total_portfolio_value), firm_df, var_df,
    )
    st.download_button(
        "⬇️ Download Executive Report (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="climate_risk_executive_report.md",
        mime="text/markdown",
    )
    with st.expander("Preview report", expanded=True):
        st.markdown(report_md)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Climate-Adjusted Credit & Market Risk Engine | PIK/NGFS scenario-based credit & portfolio stress testing for academic demonstration.")
