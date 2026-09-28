# Explainability report

This file describes the **method**. After you run `python -m src.models.train`, copy numbers from `artifacts/models/metrics.json` into your presentation.

## Global explanations

1. **Permutation importance** on the held-out test set (regression: R² drop; classification: macro-F1 drop). This answers: “If I shuffle this raw feature, how much does the model degrade?”
2. **SHAP** (when the SHAP library can explain the chosen estimator). Mean absolute SHAP values rank transformed features.

Typical strong drivers on this dataset, in past runs of similar pipelines, are **past failures**, **study time**, **wanting higher education**, **absences**, and **parent education**. Confirm with the current `metrics.json`; do not quote this paragraph as a result.

## Local explanations (at least three)

Training stores three SHAP local examples when SHAP succeeds. The app also computes a fast local explanation:

> How much does this student’s value on each raw feature move the predicted score versus a typical training student?

The Predict page table is that local explanation. Use **three different form submissions** in the demo (high absences, zero failures + high study time, no higher-education plan) so viva examiners see three individual explanations.

## How recommendations link to explanations

`src/models/recommend.py` maps weak raw signals (failures, absences, low studytime, no internet, …) to concrete resources. If local deltas are negative, those features are appended as “model-sensitive factors.”

## Caveat

Feature importance is **not causation**. A student with extra school support may have a lower predicted score because support was given *after* struggle, not because support harms learning.
