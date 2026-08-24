# Climate-Adjusted Credit & Market Risk Engine

**PIK/NGFS Scenario-Based Portfolio Climate Stress Testing and Climate-VaR Measurement**


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14y4IMtyRo7oososN_oRRHZS26cGneZg4)
---

## Overview

This engine takes PIK/NGFS climate scenario data — temperature-rise pathways and carbon-price
trajectories — as input, and translates them into standard FRM risk metrics for a firm or portfolio:
Probability of Default (PD), Loss Given Default (LGD), and Climate Value at Risk (Climate-VaR). It is
a companion project to [CTVI](https://github.com/anthonylee03/Climate_Transition_Vulnerability_Index):
CTVI builds an index-level view of transition-risk exposure across companies; this engine makes the
underlying credit- and portfolio-risk mechanics interactive.

The project demonstrates how PIK's scenario work, via NGFS, functions as an input to central-bank and
supervisory climate stress tests:

- **Structural credit risk modeling** — Merton default model translating a carbon-price shock into
  Distance to Default and PD for a single firm, with a built-in firm library (Steel, Oil & Gas,
  Fossil Power Utility, Clean Energy Tech, Commercial Real Estate) as calibrated starting points
- **Physical-risk-adjusted LGD** — a non-linear, PIK-style damage function scaling Loss Given Default
  with global temperature rise
- **Portfolio Climate-VaR with an editable correlation matrix** — Monte Carlo simulation of a
  multi-sector portfolio with a hand-editable NxN correlation matrix (not just a single average-
  correlation assumption), plus a climate-weighted penalty, 95%/99% VaR, and Expected Shortfall
- **Scenario comparison** — all three NGFS-style presets (Orderly, Disorderly, Hot House World)
  re-run side by side for the same firm/portfolio
- **Auto-generated executive report** — a one-click, downloadable Markdown report assembling the
  current session's firm, portfolio, and scenario-comparison results
- **Interactive dashboard** — Streamlit app with live carbon-price/temperature sliders and an
  editable portfolio composition table
- **Vibe-coded** — the Merton model and Monte Carlo engine were scaffolded with AI assistance, then
  reviewed and adjusted by hand, including a covariance positive-semidefinite guard that keeps a
  hand-edited correlation matrix numerically valid (see Data & Limitations)

---

## Methodology at a Glance

| Risk channel | PIK/NGFS input | FRM translation |
|---|---|---|
| **Transition risk** | Carbon price pathway | Higher operating cost for carbon-intensive firms → Merton structural default model → shift in Distance to Default and PD |
| **Physical risk** | Temperature rise, extreme-weather frequency | Asset value write-down → higher LGD via a non-linear damage function |
| **Portfolio risk** | Both of the above, applied across sector weights | Climate-weighted Monte Carlo VaR / Expected Shortfall |

**Distance to Default:** $DD = \dfrac{\ln(V_0/D) + (r - 0.5\sigma_V^2)T}{\sigma_V\sqrt{T}}$, $PD = N(-DD)$

**LGD adjustment:** $\text{LGD}_{adj} = \text{LGD}_{base} + (1-\text{LGD}_{base}) \times \Omega(T_{rise})$,
where $\Omega(T) = \min(0.02T^2 + 0.01T,\ 0.5)$

Full derivations are in the in-app **Methodology** tab.

---

## Repository Structure

```
.
├── app.py                # Full Streamlit application: scenario parser, firm library,
│                          # Merton credit model, Monte Carlo Climate-VaR engine (editable
│                          # correlation matrix), scenario comparison, executive report
│                          # generator, and dashboard UI — single file
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Getting Started

```bash
git clone <this-repo>
cd climate-adjusted-credit-market-risk-engine
pip install -r requirements.txt
streamlit run app.py
```

No data download required — all scenario parameters and portfolio holdings are set interactively in
the app (sidebar sliders + editable data table).

---

## Technology Stack

- **App framework:** Streamlit
- **Modeling:** NumPy, SciPy (`norm.cdf`/`norm.pdf` for the Merton model)
- **Simulation:** NumPy (`default_rng`, `multivariate_normal`) for Monte Carlo Climate-VaR
- **Visualization:** Plotly (distribution plots, sensitivity charts, loss histograms)
- **Data handling:** pandas (portfolio table, CSV export)

---

## Key Results (example run, default inputs)

Single firm ($V_0=\$1{,}000$M, $D=\$700$M, $\sigma_V=25\%$, 500 kt CO₂/yr, 30% pass-through,
base LGD 45%) across the three NGFS presets:

| Scenario | PD (shocked) | Adjusted LGD | Expected Loss |
|---|---:|---:|---:|
| Net Zero 2050 (Orderly) | 11.0% | 48.3% | $50.4M |
| Delayed Transition (Disorderly) | 15.2% | 49.6% | $67.8M |
| Current Policies (Hot House World) | 7.7% | 60.4% | $46.2M |

(Baseline PD with no climate shock: 7.5%.)

Sample 5-sector portfolio ($1,000M book, 5,000 Monte Carlo draws, seed 42):

| Scenario | 95% Climate-VaR | 95% Expected Shortfall |
|---|---:|---:|
| Net Zero 2050 (Orderly) | $272.4M (27.2% of book) | $321.8M |
| Delayed Transition (Disorderly) | $308.4M (30.8% of book) | $359.9M |
| Current Policies (Hot House World) | $275.0M (27.5% of book) | $328.4M |

The Disorderly scenario produces the highest transition-driven PD and portfolio VaR; Hot House World
produces the lowest PD but the highest LGD, since its risk is physical rather than transition-driven —
consistent with the NGFS narrative that an abrupt, late transition is the most acute near-term credit
event, even though a no-policy world is worse on a multi-decade physical-damage basis.

---

## Data & Limitations

Carbon-price and temperature-rise pathways per scenario preset are illustrative and *directionally*
aligned with NGFS scenario narratives — they are not pulled live from the NGFS Scenario Explorer.
Swap in real scenario output (e.g. NGFS Phase IV data) before using this for anything beyond a demo.
Portfolio correlation is now hand-editable per asset pair, but it is still a user-supplied matrix
rather than one estimated from historical return data, and physical risk enters through LGD (single
firm) or a flat per-sector penalty (portfolio) rather than asset-level physical hazard exposure. The
Merton model and Monte Carlo engine logic were scaffolded with AI assistance and then checked by
hand — including a fix that keeps the simulated covariance matrix positive semi-definite even when a
hand-edited correlation matrix is internally inconsistent, and a fix for a tuple/array type bug in
the climate-penalty calculation caught by testing against `streamlit.testing.v1.AppTest` before this
was pushed. This is a teaching/portfolio tool for scenario-based stress testing, not a production
credit or market risk model, and nothing here is investment or credit advice.

---

## Author

Seungmin Lee (이승민) — HUFS, Public Administration & Economics · ACAMS (CAMS)
