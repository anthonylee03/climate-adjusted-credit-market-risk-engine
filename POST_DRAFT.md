[Project Share] Building a Climate-Adjusted Credit & Market Risk Engine — PIK/NGFS Scenario Stress Testing for PD, LGD, and Climate-VaR

Seungmin Lee (https://www.linkedin.com/in/seungminlee030323/)
Compliance Specialist | CAMS Certified | Investment Asset Management | FRM Candidate
[date]

CTVI, my last project, scores companies on how exposed they are to the low-carbon
transition. It answers "how vulnerable." It doesn't answer the question a credit
desk actually asks next: if that exposure turns into a carbon-price shock, what
does it do to this firm's probability of default, and what does it do to a
portfolio's tail loss? That's the gap this project fills.

What I built

Climate-Adjusted Credit & Market Risk Engine takes PIK/NGFS scenario data —
temperature-rise pathways and carbon-price trajectories — and runs them through
two standard FRM tools: a Merton structural credit model for single-firm
transition risk, and a Monte Carlo simulation for portfolio Climate-VaR. Move a
carbon-price slider, and you watch Distance to Default shrink, PD rise, and LGD
widen through a non-linear physical-damage function, plus a portfolio-level
VaR/Expected Shortfall readout across sectors with different carbon intensity.

The logic I wanted to make concrete: PIK's scenario work, through NGFS, is
already the reference input for central bank and supervisory climate stress
tests. This project is the next translation step — turning a temperature path
and a carbon price into firm-level default risk and portfolio-level tail loss,
using the credit and market risk tools an FRM curriculum actually teaches.

Pipeline

* NGFS/PIK Scenario Parser: three preset pathways — Net Zero 2050 (Orderly),
  Delayed Transition (Disorderly), Current Policies (Hot House World) — each
  as a temperature-rise / carbon-price pair, plus a fully custom setup
* Structural Credit Risk Model: a built-in library of five calibrated
  companies (Steel, Oil & Gas, Fossil Power Utility, Clean Energy Tech,
  Commercial Real Estate) as starting points, or enter your own. Carbon tax
  → EBIT/asset-value shock → Distance to Default and PD via Merton; physical
  risk scales LGD through a PIK-style non-linear damage function
* Portfolio Climate-VaR Calculator: Monte Carlo simulation of a multi-sector,
  editable portfolio, now with a hand-editable NxN correlation matrix instead
  of a single average-correlation assumption — 95%/99% VaR and Expected
  Shortfall, with a downloadable CSV of the simulated loss distribution
* Scenario Comparison: re-runs both models across all three NGFS presets side
  by side for the same firm/portfolio, instead of one scenario at a time
* Auto-generated Executive Report: one click assembles the current session's
  firm, portfolio, and scenario-comparison results into a downloadable
  Markdown report
* Interactive dashboard: Streamlit app with live sliders for carbon price,
  temperature rise, correlation, and Monte Carlo seed
* Vibe-coded: the Merton model and Monte Carlo engine were scaffolded with AI
  assistance, then checked by hand and against `streamlit.testing.v1.AppTest`
  — which caught a real tuple/array type bug in the climate-penalty
  calculation before it shipped, plus a fix to keep the simulated covariance
  matrix valid when the hand-edited correlation matrix is internally
  inconsistent

Key finding

Running a sample firm and a 5-sector portfolio across the three scenarios: the
Delayed Transition (Disorderly) scenario produces both the highest PD (15.2%,
vs. a 7.5% no-shock baseline) and the highest portfolio Climate-VaR (30.8% of
book value). Current Policies (Hot House World) produces the lowest PD but the
highest LGD, since its risk channel is physical rather than transition-driven.
That split is the point of the exercise — a disorderly transition is the
sharper near-term credit event, even though a no-policy world compounds worse
over a multi-decade physical-damage horizon.

What's next

Carbon-price and temperature paths per scenario are currently illustrative,
directionally aligned with NGFS scenario narratives rather than pulled live
from the NGFS Scenario Explorer — swapping in real scenario output is the next
step, along with estimating the correlation matrix from historical return data
instead of setting it by hand. The Merton and Monte Carlo methodology carries
over unchanged.

Code, methodology, and the interactive dashboard are on GitHub:
[repo link]

#ClimateRisk #CreditRisk #FRM #NGFS #Streamlit
