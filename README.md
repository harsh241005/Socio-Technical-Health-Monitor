# Socio-Technical Health Monitor

> Predicting software task stalling risk from the emotional and communicative footprint of developer teams — using JIRA issue data and Apache mailing list archives.

**MScIDS Semester VI Capstone Project**
Goa Business School, Goa University · Guide: Dr. Swapnil Fadte

---

## Team

| Name | Roll No. |
|---|---|
| Rudresh Achari | 2330 |
| Unnat Umarye | 2303 |
| Sarvadhnya Patil | 2321 |
| Samuel Bhandari | 2308 |
| Harsh Palyekar | 2329 |

---

## Overview

This project builds a machine learning system that monitors the socio-technical health of software development teams by analysing developer communication patterns in JIRA issue tracker data and Apache mailing list archives. The core hypothesis is that **the emotional tone, variance, and volume of developer communication alone are sufficient to predict project stalling** — without relying on structural metadata.

The system produces two actionable outputs:

1. **Task Stalling Risk** — classifies whether an open ticket is likely to remain stalled, using only pure socio-technical signals derived from email sentiment and communication volume.
2. **SHAP-Based Explainability** — an interactive dashboard that shows *why* the model raised or lowered a stall alert for any given ticket, enabling project managers to take targeted action.

---

## Dataset

| Property | Value |
|---|---|
| Source | Apache Software Foundation — JIRA Archive + Developer Mailing Lists |
| Projects Covered | Hadoop (Common), HDFS, YARN, MapReduce |
| Time Period | January 2023 – December 2024 |
| Raw Emails Processed | 7,051 |
| Bot Emails Removed | 6,039 (85% of corpus — sender-domain verified) |
| Human-Verified Records | 1,012 |
| Stalled Tickets | 190 |
| Active Tickets | 822 |
| Class Ratio | 4.3 : 1 (Active : Stalled) |

### Why Sender Verification Matters

In ISA II, a regex subject-line filter was used to remove automated bot emails (Jenkins, JIRA notifications). Subsequent audits revealed two critical failures: the regex was **deleting human developer emails** that discussed Jenkins, and **missing bots** that used MIME-encoded subject lines (e.g., `=?utf-8?Q?[jira]_[Commented]_..?=`). The ISA III pipeline fixes this by dropping any email sent from `jira@`, `jenkins@`, or `qa@` sender domains — a mathematically verifiable ground-truth filter that reduced the working corpus from ~6,950 contaminated rows to **1,012 confirmed human communications**.

---

## Methodology

### Pipeline Architecture (Sequential, MLOps Standard)

The repository is structured as a numbered sequential pipeline. Each script name encodes its position in the data flow:

```
01_data_acquisition.py    →  Download mbox archives from Apache Lists API
02_entity_linking.py      →  Parse emails, link to JIRA tickets, score sentiment
03_dataset_merger.py      →  Merge parsed chunks, deduplicate on UTC timestamps
04_feature_engineering.py →  Bot filter, MIME decoding, feature engineering, target definition
05_model_evaluation.py    →  5-fold stratified cross-validation across three architectures
06_shap_analysis.py       →  Final model training, PR curve, SHAP summary, .pkl export
07_eda_visualizations.py  →  EDA charts for report and presentation
```

### Sentiment Extraction

Email text is processed using **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, selected for its suitability with short-form professional communication. Each email produces a compound score in the range \[-1.0, +1.0\].

VADER is extended with a **44-term custom IT-domain lexicon** (authored by Unnat Umarye) that recalibrates technical vocabulary. For example, `kill`, `dead`, and `abort` are set to neutral (0), as they are routine programming commands — not hostile language. `fixed` and `resolved` are set positive; `crash`, `vulnerability`, and `outage` are set appropriately negative.

### The 7 "Honest" Socio-Technical Features

These features were deliberately chosen to contain **zero target leakage**. Features like `is_assigned`, `has_votes`, and `watches.watchcount` were identified as structural proxies — properties that stalled tickets already possess by definition — and excluded from the final model.

| Feature | Correlation to Stalling | Directional Meaning |
|---|---|---|
| `avg_sentiment` | +0.1865 | High positivity often masks a lack of urgency |
| `email_count_per_ticket` | −0.1771 | Active communication volume prevents stalling |
| `sentiment_variance` | −0.1636 | Emotional fluctuation signals active problem-solving; flat variance = dead ticket |
| `sentiment_trend` | +0.1070 | Tickets that grow happier over time stall more often — declining push to close |
| `priority_numeric` | +0.1026 | Higher-priority tickets stall slightly more (complexity effect) |
| `subject_length` | +0.0917 | Over-explaining vs. acting |
| `has_enough_emails` | −0.0365 | Multiple emails slightly reduces stall risk |

### Model Selection

Three architectures were evaluated using 5-fold stratified cross-validation on the 1,012 human-verified records:

| Algorithm | Mean Recall (Stalled) | Mean Precision | ROC-AUC |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.632 (± 0.153) | 0.330 | 0.717 |
| Random Forest (Ensemble) | 0.947 (± 0.074) | 0.733 | 0.982 |
| **XGBoost (Tuned — Final)** | **0.926 (± 0.091)** | **0.710** | **0.980** |

**XGBoost was selected as the final model** for the production dashboard due to its stable, well-calibrated probability distributions — critical for the SHAP force plot rendering in the Streamlit app. Random Forest achieved marginally higher cross-validation recall (0.947 vs 0.926), which is consistent with ensemble methods handling the fuzziness of human sentiment data slightly better than gradient-boosted error correction. Both results validate the socio-technical hypothesis.

### Final Model Performance (Unseen Test Set, Threshold = 0.50)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Active (0) | 0.97 | 0.95 | 0.96 | 165 |
| **Stalled (1)** | **0.79** | **0.89** | **0.84** | 38 |
| Accuracy | — | — | 0.94 | 203 |
| Macro Avg | 0.88 | 0.92 | 0.90 | 203 |

**PR-AUC: 0.93** — confirms model stability across decision thresholds.

The 0.79 precision on the stalled class is a deliberate trade-off. In an early warning radar context, a recall of 89% (catching 9 of 10 failing tasks) is more operationally valuable than high precision. The 21% of false alarms still represent tickets with poor communication patterns — flagging them prompts manager review that is likely to be productive regardless.

---

## Key Findings

- Developer sentiment remained broadly positive (mean ≈ 0.28) across the 24-month observation window.
- Sentiment dropped to **0.10** during the Hadoop 3.4.1 release candidate crunch (November 2024), recovering to **0.42** post-release — confirming the pipeline's ability to detect real engineering events in the data.
- Critical tickets carry the **lowest median sentiment (0.23)**; Blocker tickets show an elevated score (0.38), attributed to relief following rapid resolution.
- Stalled tasks exhibit a **bimodal sentiment distribution** — a high upper quartile (frustration release) combined with an extended negative tail (sustained stress).
- `sentiment_variance` is the **top SHAP predictor**: low variance (flat emotional signal) strongly pushes toward a stall prediction, confirming that communication silence is the clearest early warning of task abandonment.
- Communication-derived features account for the majority of total model feature importance, confirming that the NLP pipeline contributes substantially more predictive signal than structured JIRA metadata alone.

---

## Project Structure

```
Socio-Technical-Health-Monitor/
│
├── data/
│   ├── raw/                          # Original .mbox files and issues.csv
│   ├── interim/                      # Parsed chunks and master_project_dataset.csv
│   └── processed/                    # isa3_enriched_dataset.csv (model-ready)
│
├── scripts/
│   ├── 01_data_acquisition.py        # Download mbox archives
│   ├── 02_entity_linking.py          # Parse emails, VADER scoring, JIRA linking
│   ├── 03_dataset_merger.py          # Merge + UTC-dedup across chunks
│   ├── 04_feature_engineering.py     # Bot filter, MIME decode, feature engineering
│   ├── 05_model_evaluation.py        # 5-fold CV across LR, RF, XGBoost
│   ├── 06_shap_analysis.py           # Final model, SHAP plots, .pkl export
│   ├── 07_eda_visualizations.py      # EDA charts
│   └── archive_and_audits/           # ISA II scripts and diagnostic files
│
├── models/
│   ├── isa3_xgboost_honest.pkl       # Final trained XGBoost model
│   └── isa3_shap_explainer_honest.pkl
│
├── visuals/
│   ├── eda_plots/                    # EDA charts (output of 07_eda_visualizations.py)
│   │   ├── 1_sentiment_by_priority.png
│   │   ├── 2_stalled_vs_active.png
│   │   ├── 3_correlation_heatmap.png
│   │   ├── 5_monthly_sentiment_trend.png
│   │   ├── 6_email_volume_distribution.png
│   │   └── 7_task_status_by_priority.png
│   ├── isa3_honest_shap_summary.png  # Output of 06_shap_analysis.py
│   └── isa3_honest_pr_curve.png      # Output of 06_shap_analysis.py
│
├── tests/
│   └── test_parse.py                 # Unit tests for email extraction logic
│
├── reports/                          # ISA II and ISA III academic reports
├── app.py                            # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/rudresh33/Socio-Technical-Health-Monitor.git
cd Socio-Technical-Health-Monitor

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
vaderSentiment
streamlit
joblib
matplotlib
seaborn
jupyter
```

---

## Running the Pipeline

Run these scripts in order from the repository root. Each step produces the input for the next.

```bash
# Step 1 — Download mailing list archives (requires internet access)
python scripts/01_data_acquisition.py

# Step 2 — Parse emails, link to JIRA tickets, extract sender and body
python scripts/02_entity_linking.py

# Step 3 — Merge parsed chunks, deduplicate on UTC timestamps
python scripts/03_dataset_merger.py

# Step 4 — Apply bot filter, MIME decoding, engineer all features
python scripts/04_feature_engineering.py

# Step 5 — Cross-validate three model architectures, print metrics
python scripts/05_model_evaluation.py

# Step 6 — Train final XGBoost, generate SHAP plots, export .pkl files
python scripts/06_shap_analysis.py

# Step 7 — Generate all EDA visualisations (saved to visuals/eda_plots/)
python scripts/07_eda_visualizations.py

# Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## Running Tests

The `tests/` directory contains a unit-testing framework that validates the email extraction logic against mock data — without requiring any real mbox files.

```bash
python tests/test_parse.py
```

This will: generate a synthetic JIRA CSV and a 3-email mock mbox file (containing one human email, one Jenkins bot, and one JIRA bot), run the parser, merge against mock JIRA data, and print the extracted sender, subject, body snippet, and sentiment score for each matched ticket. Successful output confirms the bot filter and sentiment pipeline are working correctly.

---

## ISA Milestone History

| Milestone | Key Deliverable | Status |
|---|---|---|
| ISA I | Problem definition, project proposal, dataset identification | Complete |
| ISA II | EDA, Random Forest baseline, initial feature engineering | Complete |
| ISA III | Bot filter overhaul, target leakage removal, XGBoost + SHAP, Streamlit dashboard | Complete |

---

## Academic Context

This repository is the codebase for a Semester VI MScIDS capstone project submitted under the Internal Semester Assessment (ISA) framework at Goa Business School, Goa University. The project is evaluated across three ISA milestones covering problem definition, data analysis and modelling, and final delivery.

All data sourced from the Apache Software Foundation public archives. No private or proprietary data was used.

---

## License

Submitted for academic assessment. All rights reserved by the authors. Please contact the team before reusing any part of this work.