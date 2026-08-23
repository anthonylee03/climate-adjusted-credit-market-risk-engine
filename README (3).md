# 🌍 Climate Transition Vulnerability Index (CTVI)

**A Machine Learning Framework for Quantifying Corporate Climate Transition Risk under Multiple Climate Policy Scenarios**

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anthonylee03/ctvi-climate-transition-risk/blob/main/CTVI_Climate_Transition_Vulnerability_Index.ipynb)

---

## Overview

CTVI is an **original**, transparent, data-driven composite index (0–100) quantifying how exposed a
company is to the financial risks of the global low-carbon transition. It is explicitly **not** a
prediction of an existing ESG score — it is built from first principles using financial-risk-modeling
conventions, in the style of a climate-stress-testing research note (à la NGFS / ECB / central bank
climate risk exercises).

The project demonstrates an end-to-end quantitative climate finance research pipeline:

- **Feature engineering** with explicit economic rationale (12+ engineered variables — Transition
  Leverage, Financial Resilience Index, Climate Financial Buffer, Policy Sensitivity Index, etc.)
- **An original index methodology** — 7 dimensions, weights derived via PCA rather than asserted by hand
- **Machine learning validation** — 6 model families (Linear/Logistic Regression, Random Forest, Extra
  Trees, Gradient Boosting, XGBoost, LightGBM) recovering CTVI from raw fundamentals, with full
  cross-validation, grid search, ROC/AUC, calibration, and learning curves
- **Explainable AI** — SHAP (global + per-company waterfall), permutation importance, partial dependence
- **Unsupervised clustering** — 5 transition-risk archetypes (K-Means, hierarchical, DBSCAN; PCA/t-SNE/UMAP)
- **Scenario analysis** — 7 NGFS-style climate policy scenarios (Current Policies → Net Zero 2050 →
  Rapid Fossil Fuel Phase-Out), with sector-level impact ranking and $-denominated earnings-at-risk
- **Monte Carlo simulation** — 2,000-iteration stochastic uncertainty quantification with sensitivity ranking
- **Interactive dashboard** — Streamlit app for live scenario exploration
- **Automated executive report** — consulting/central-bank-style Markdown research report generated
  programmatically from the pipeline's own outputs

---

## Score Interpretation

| CTVI Range | Tier                      |
|-----------:|---------------------------|
| 0–20       | Climate Leader             |
| 20–40      | Transition Ready           |
| 40–60      | Moderate Risk               |
| 60–80      | High Risk                   |
| 80–100     | Critical Transition Risk    |

---

## Methodology at a Glance

CTVI aggregates **7 dimensions** — Carbon Exposure, Financial Fragility, Policy Sensitivity, Operational
Resilience, Technology Readiness, Green Investment Capacity, and Market Adaptability — each built from
2–4 economically-motivated engineered sub-indicators.

Weights are **never asserted by hand**:
1. Sub-indicators are sign-aligned (higher = more vulnerable) and standardized.
2. Within each dimension, PCA is run on the sub-indicators; the first component's absolute loadings
   (normalized) become the sub-indicator weights.
3. A second, global PCA runs across the 7 resulting dimension scores; its loadings become the
   dimension-level weights in the final index.

This means every weight in CTVI traces back to *actual explained variance in the data*, not analyst
judgment — a defensible, auditable methodology consistent with how quantitative risk models are built
in practice.

---

## Repository Structure

```
.
├── CTVI_Climate_Transition_Vulnerability_Index.ipynb   # Main notebook (fully executed, run top-to-bottom in Colab)
├── CTVI_Executive_Report.md                            # Auto-generated executive research report
├── README.md
├── dashboard/
│   └── streamlit_app.py                                # Interactive dashboard (run: streamlit run dashboard/streamlit_app.py)
└── src/                                                 # Modular pipeline (source of truth; notebook inlines these)
    ├── data_sources.py            # Live-data attempts (World Bank, OWID, Yahoo Finance) + calibrated synthetic fallback
    ├── feature_engineering.py     # Economically-motivated engineered features
    ├── ctvi_index.py              # Original CTVI methodology (PCA-based data-driven weighting)
    ├── ml_models.py               # 6-model comparison, CV, grid search, evaluation
    ├── explainability.py          # SHAP, permutation importance, partial dependence
    ├── clustering.py              # K-Means / hierarchical / DBSCAN, PCA/t-SNE/UMAP
    ├── scenario_analysis.py       # 7 NGFS-style climate policy scenarios
    ├── monte_carlo.py             # 2,000-iteration stochastic simulation + sensitivity analysis
    ├── visualizations.py          # Plotly publication-quality figures
    └── report_generator.py        # Automated executive report assembly
```

---

## Getting Started

### Option 1 — Google Colab (recommended)
Click the **Open in Colab** badge above, or go directly to:
[colab.research.google.com/github/anthonylee03/ctvi-climate-transition-risk/blob/main/CTVI_Climate_Transition_Vulnerability_Index.ipynb](https://colab.research.google.com/github/anthonylee03/ctvi-climate-transition-risk/blob/main/CTVI_Climate_Transition_Vulnerability_Index.ipynb)

Then:
1. Uncomment the `!pip install` cell at the top and run all cells top-to-bottom (Runtime → Run all).
2. No manual data download required — the pipeline attempts live public-data pulls and falls back to
   calibrated synthetic data automatically.

*(The badge link only resolves once the notebook is pushed to the `main` branch of the GitHub repo —
if you rename the repo or use a different branch, update the badge URL in this README accordingly.)*

### Option 2 — Local environment
```bash
git clone <this-repo>
cd ctvi-climate-transition-risk
pip install -r requirements.txt   # or: pip install numpy pandas scikit-learn xgboost lightgbm shap plotly umap-learn tabulate streamlit requests
jupyter notebook CTVI_Climate_Transition_Vulnerability_Index.ipynb
```

### Interactive dashboard
```bash
streamlit run dashboard/streamlit_app.py
```
Select a company, then adjust carbon price, policy stringency, energy prices, and emission-reduction
effort sliders to see CTVI update live, along with its dimension decomposition.

---

## Technology Stack

- **Data / analysis:** pandas, numpy
- **Machine learning:** scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP, scikit-learn permutation importance / partial dependence
- **Clustering & dimensionality reduction:** scikit-learn (KMeans, Agglomerative, DBSCAN, PCA, t-SNE), UMAP
- **Visualization:** Plotly
- **Dashboard:** Streamlit
- **Notebook tooling:** nbformat, Jupyter / Google Colab

---

## Key Results (this run)

- **Best regression model:** LightGBM — Test R² = 0.88, 5-fold CV R² = 0.91 ± 0.02, recovering continuous
  CTVI from raw company fundamentals alone.
- **Best classification model:** XGBoost — Test AUC = 0.999 for the "High-Risk-or-worse" (CTVI ≥ 60) flag.
- **Most influential dimensions (data-driven PCA weights):** Carbon Exposure (19%), Green Investment
  Capacity (17%), Policy Sensitivity (15%).
- **Highest-exposure sectors:** Oil & Gas, Transportation, Materials — both in relative CTVI and in
  absolute estimated earnings-at-risk under aggressive transition scenarios.
- **Monte Carlo:** carbon-price uncertainty dominates portfolio-level CTVI variance among all simulated
  macro/policy drivers.

Full detail, methodology tables, and all supporting charts are in the executed notebook and
`CTVI_Executive_Report.md`.

---

## Data & Limitations

Company-level data is a **statistically calibrated synthetic panel** (320 companies across 12 sectors and
15 countries), anchored to approximate public reference figures (World Bank Carbon Pricing Dashboard, IEA,
sector/country risk profiles consistent with NGFS scenario design conventions) — used because live,
authenticated, company-level disclosure data (SEC climate disclosures, CDP, Bloomberg ESG) was not
available in this environment. The pipeline is explicitly designed to substitute live data sources
transparently (see `src/data_sources.py`) wherever network/API access permits; the *methodology* — CTVI
construction, ML validation, scenario design, Monte Carlo — is fully real and would apply unchanged to
real company data.

Other limitations, and a full future-research agenda, are documented in `CTVI_Executive_Report.md`
(Sections 10–11).

---

## Author

Built as a quantitative climate finance / financial risk portfolio project, in the style of research
produced at international financial institutions (World Bank, IMF, OECD, BIS, NGFS-affiliated central banks).

## License

MIT
