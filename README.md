# CASCADIQ — SafeCart Fraud Decisioning

**TAG AI Team Development Program · Case 06**  
Prepared by **Abdelrahman Nabil**

### Portfolio summary

> **CASCADIQ** — Engineered an e-commerce fraud-decisioning system on 35K+ transactions; evaluated 8 candidate models (logistic regression champion, AUC 0.808) and optimized a capacity-constrained 4-action policy via linear programming, cutting expected fraud cost 67.9% (Monte Carlo-validated); operationalized in a Streamlit command-center dashboard.

End-to-end fraud decisioning system for SafeCart (e-commerce): a calibrated risk model plus a capacity-constrained optimization policy that assigns every operational event to one of four actions — **Approve**, **Step-up Authentication**, **Manual Review**, or **Temporary Hold**.

This repository includes the reproducible analysis notebook, champion model artifact, Streamlit command-center dashboard, final decision output CSV, and executive deliverables.

---

## Problem

SafeCart’s current process relies on manual review of suspicious events. Review capacity is limited, and the existing workflow misses roughly **26.5%** of later-confirmed fraud (catch rate ≈ **73.5%** on labeled historical data).

**Goal:** score every event and recommend an action automatically, without exceeding operational caps:

| Constraint | Cap |
|---|---|
| Manual Review | 1,800 events |
| Temporary Hold | 350 events |

---

## Solution (one sentence)

A **logistic regression** model estimates fraud probability; a **linear program (LP)** maps those probabilities to four actions under capacity limits; **Monte Carlo** stress-tests expected loss reduction under uncertain step-up deterrence.

---

## Key results

| Metric | Value |
|---|---|
| Historical events (clean) | 35,000 (70 duplicates dropped) |
| Base fraud rate | 3.44% |
| Current process recall | 73.5% |
| Champion model | Logistic Regression |
| Champion CV AUC | **0.808** |
| Champion CV AP | 0.183 (~5.3× base rate) |
| Do-nothing expected cost | ~715k |
| Naive 2-bucket cost | ~266k |
| **LP-optimal cost** | **~230k** |
| Monte Carlo mean loss prevented | **~67.9%** (P5–P95 ≈ 62–73%) |
| Zero-deterrence floor | ~57.6% |

### Action mix on the current operational file (2,500 events)

| Action | Count |
|---|---|
| Step-up Authentication | 1,591 |
| Manual Review | 508 |
| Temporary Hold | 349 |
| Approve | 52 |

---

## Repository structure

```
safecart-fraud-decisioning/
├── README.md
├── .gitignore
├── requirements.txt
├── dashboard/                          # Interactive Streamlit command center
│   ├── app.py                          # Dashboard UI (8 pages)
│   ├── precompute.py                   # Rebuilds summary / scored / MC artifacts
│   ├── final_champion.pkl              # Trained champion model + scaler + feature cols
│   ├── SafeCart_Reproducible_Analysis.ipynb
│   ├── .streamlit/config.toml
│   └── data/
│       ├── 02_SafeCart_Historical_Data.csv
│       ├── 03_SafeCart_Current_Operational_Cases.csv
│       ├── current_scored.csv
│       ├── decision_output.csv
│       ├── summary.json
│       ├── pr_curve.json
│       └── monte_carlo_loss_prevented_pct.npy
├── deliverables/                       # Case submission artifacts
│   ├── SafeCart_Executive_Presentation.pptx
│   ├── SafeCart_Executive_Recommendation.docx
│   ├── SafeCart_Mini_Implementation_Plan.docx
│   ├── 05_SafeCart_Decision_Output_FINAL.csv
│   └── SafeCart_Reproducible_Analysis.ipynb
└── docs/                               # Defense / walkthrough notes
    ├── CASCADIQ_Project_Walkthrough_Arabic.md
    ├── CASCADIQ_Project_Walkthrough_Arabic_Fixed.pdf
    └── CASCADIQ_Presentation_Script_Arabic.md
```

---

## Quick start (dashboard)

```bash
# 1. Clone
git clone https://github.com/<your-username>/safecart-fraud-decisioning.git
cd safecart-fraud-decisioning

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the command center
cd dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) if the browser does not open automatically.

> Precomputed artifacts already live under `dashboard/data/`, so the app loads instantly without re-training.

---

## Regenerating numbers

To rebuild `summary.json`, `current_scored.csv`, Monte Carlo outputs, etc. from the champion model and source CSVs:

```bash
cd dashboard
python precompute.py
```

This script mirrors the logic in `SafeCart_Reproducible_Analysis.ipynb` so dashboard KPIs stay aligned with the notebook, deck, and reports.

---

## Dashboard pages

| Page | What it shows |
|---|---|
| **Command Center** | Miss rate, champion AUC, expected loss reduction, action Sankey, capacity bars, cost comparison, 4-week shadow-mode ask |
| **Data & Quality** | Duplicate drop, monthly volume vs fraud rate, missingness, leakage wall, decision-time features |
| **Risk Intelligence** | Flag lifts, stacked-flag heatmaps, spend / login / tenure signals, hour & weekday (and why they are *not* a policy) |
| **Model Lab** | Eight model families on identical time-respecting folds; why logistic regression is champion |
| **Decision Policy** | LP vs do-nothing vs naive 2-bucket, unit economics, action mix, score-by-action |
| **Monte Carlo** | 20,000-run loss-prevented distribution, empirical CDF, percentiles, zero-deterrence floor |
| **Live Operations** | Filter the 2,500-event queue, open a dossier, nearest confirmed-fraud neighbors, download slice |
| **Responsible AI** | Equal Hold allocation vs unequal Hold recall by nationality / gender; named-owner pilot monitor |

---

## Method overview

### 1. Data
- **Historical:** ~35k labeled events (`Chargeback_Confirmed_30D`), Jan 2025 – Aug 2026
- **Current ops:** 2,500 unlabeled events (24–26 Aug 2026) to score and assign

### 2. Features (decision-time only)
Numeric / flag features include tenure, device known, new-device age, IP distance, failed logins, password reset, amount, amount-vs-avg, shipping-address-new, plus missingness indicators, hour, day-of-week, and one-hot categories (`Account_Value_Band`, `Payment_Method`).

Leakage-prone post-outcome fields are excluded from scoring.

### 3. Model
Eight families compared under time-respecting folds. **Logistic Regression** wins on AUC/AP tradeoff and interpretability (tied AUC with calibrated Linear SVM; better AP and ops clarity).

### 4. Decision policy (LP)
Unit costs (illustrative economics used in the case):

| Component | Value |
|---|---|
| Missed fraud cost | 8,000 |
| Manual review cost | 60 |
| Step-up cost | 8 |
| False hold cost | 220 |
| Step-up deterrence (point estimate) | 0.5 |
| Review catch rate (from history) | ~0.735 |

Each event is assigned exactly one action; Review ≤ 1,800 and Hold ≤ 350.

### 5. Monte Carlo
Deterrence sampled Uniform(0.2, 0.8) over **20,000** runs to produce the loss-prevented distribution (mean ~68%, floor with zero deterrence ~58%).

### 6. Fairness
Hold-tier allocation is near-equal across gender and nationality groups; conditional fraud recall in Hold can still differ — tracked as a pilot monitor with a named owner.

---

## Deliverables

| File | Purpose |
|---|---|
| `deliverables/SafeCart_Executive_Presentation.pptx` | Executive deck |
| `deliverables/SafeCart_Executive_Recommendation.docx` | Written recommendation |
| `deliverables/SafeCart_Mini_Implementation_Plan.docx` | 4-week shadow-mode / rollout plan |
| `deliverables/05_SafeCart_Decision_Output_FINAL.csv` | Scored + actioned operational file |
| `deliverables/SafeCart_Reproducible_Analysis.ipynb` | Full reproducible analysis |
| `dashboard/` | Supplementary interactive demo for defense / walkthrough |

The Streamlit app is a **supplementary demo**, not a required case deliverable. Lead with the PPTX, notebook, CSV, and Word docs; use **Live Operations** when an examiner asks to pull a specific event.

---

## Tech stack

- Python 3.10+
- pandas, numpy, scikit-learn, scipy
- streamlit, plotly
- Jupyter notebook for the full analysis

---

## Requirements

See [`requirements.txt`](requirements.txt):

```
streamlit>=1.30
pandas>=2.0
numpy>=1.24
plotly>=5.18
scikit-learn>=1.3
scipy>=1.11
```

Optional for the notebook: `jupyter` / `ipykernel`.

---

## Defense notes

Arabic walkthrough and presentation script live in `docs/`:

- Full project explanation: `docs/CASCADIQ_Project_Walkthrough_Arabic.md` (PDF also included)
- Spoken script: `docs/CASCADIQ_Presentation_Script_Arabic.md`

Every headline number in the dashboard traces to `dashboard/data/summary.json`, which is produced by the same logic as the notebook.

---

## License / usage

Prepared for the **TAG AI Team Development Program (Case 06)**. Dataset and case materials are for educational / case-evaluation use as provided by the program. Update this section if you publish under a different license.
