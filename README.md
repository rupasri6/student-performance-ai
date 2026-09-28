# Intelligent Student Performance Prediction and Personalized Learning Recommendation System

Advanced / final-year AI-ML project following the Trekverse Edutech assignment specification.

## What this system does

1. Cleans a public educational dataset (UCI Student Performance: Math + Portuguese).
2. Explores patterns with 10 charts.
3. Predicts a **final score (G3, 0–20)** and a **support group** (high / medium / low).
4. Clusters engagement profiles **without using the grade**.
5. Explains predictions (permutation importance + local feature deltas; SHAP when available).
6. Recommends learning actions from weak signals.
7. Serves everything in a **Streamlit** web app.

Predictions are **decision support**, not judgements of ability. The project does **not** claim 100% accuracy.

## Dataset

| Item | Detail |
| --- | --- |
| Source | [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) |
| License | UCI repository terms; Cortez & Silva (2008). Credit the original paper in reports. |
| Files | `data/raw/student-mat.csv`, `data/raw/student-por.csv` |
| Rows after merge | ~1044 course observations |
| Target (regression) | `G3` final grade |
| Target (classification) | `support_need` from G3: `<10` high, `10–13` medium, `≥14` low support |
| Leakage controls | `G1` and `G2` are **never** used as features |
| Fairness | `sex` is **never** used as a feature |

## Setup (Windows)

```powershell
cd C:\Users\lenovo\student-performance-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run the project in order (matches the assignment weeks)

```powershell
# Week 2–3: exploratory analysis (writes artifacts/figures/*.png)
python -m src.eda

# Week 3–5: train 5+ models, tune, explain, save pipelines
python -m src.models.train

# Week 7: unit tests
python -m pytest -q

# Week 6–7: web application
streamlit run src/app/streamlit_app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Repository layout

```
src/config.py              shared paths and feature lists
src/data/                  load, validate, feature engineering, preprocess
src/models/train.py        model zoo, tuning, clustering, metrics.json
src/models/inference.py    safe prediction API used by the app and tests
src/models/recommend.py    learning-resource rules
src/app/streamlit_app.py   dashboards
tests/                     preprocessing, inference, app smoke tests
docs/                      proposal, data dictionary, ethics, viva notes
notebooks/01_eda.ipynb     notebook view of EDA
```

## Models compared

**Regression:** mean baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR, XGBoost (if installed), tuned Random Forest.

**Classification:** majority baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVC, tuned Gradient Boosting.

**Clustering:** K-Means (k=3) on preprocessed features with **no G3**.

The deployed model is **not** chosen by training score. `artifacts/models/metrics.json` records held-out test metrics and the selection rationale.

## Tests

```powershell
python -m pytest -q
```

Coverage includes missing-value/range validation, leakage-feature exclusion, recommendation triggers, and a tiny end-to-end inference pipeline.

## Ethics (short)

- No student names or IDs are collected.
- Report real test MAE/RMSE/R² and macro-F1; never round them up to 100%.
- Use the explanation panel with a teacher before any high-stakes action.
- See `docs/ethics_and_limitations.md`.

## Citation

P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
