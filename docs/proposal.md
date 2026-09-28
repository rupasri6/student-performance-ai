# Project proposal and problem definition

**Title:** Intelligent Student Performance Prediction and Personalized Learning Recommendation System  
**Role context:** Project Manager – Trekverse Edutech (assignment framing)  
**Level:** Advanced / final-year / industry-oriented

## Problem

Schools already store attendance, prior failures, study time, family academic background, and engagement flags. Those tables rarely become **timely support**. Staff see a final grade when it is too late to help.

This project turns permitted academic and engagement fields into:

1. A predicted final score (0–20).
2. A support-need group (high / medium / low).
3. An explanation of which factors moved the estimate.
4. Concrete learning-resource recommendations.

## Why this dataset

The [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) set is public, well documented, and contains both academic process features (study time, failures, absences) and context features (parent education, extra support). It does **not** contain names or IDs.

Math and Portuguese files are stacked with a `subject` column so the model can learn subject differences without pretending every course is the same.

## Target definitions

| Task | Target | Justification |
| --- | --- | --- |
| Regression | `G3` | Official final grade on the original 0–20 scale. |
| Classification | `support_need` | `G3 < 10` → high support (below typical pass); `10 ≤ G3 < 14` → medium; `G3 ≥ 14` → low. Bands are policy-like, not a claim about intelligence. |
| Clustering | none | Groups students by engagement-style features only. |

## Leakage and fairness choices

- **Dropped `G2`:** second-period grade is nearly the final exam; using it would look accurate and be useless as an early warning.
- **Dropped `G1`:** still a prior exam on the same subject; excluded so the default model is an *early-warning* model.
- **Dropped `sex`:** demographic sex is not treated as academic ability.

## Success criteria (from the assignment)

- Reproducible pipeline, 5+ algorithms, baselines, held-out test metrics.
- ≥8 visualizations, explainability, recommendations, Streamlit app, tests, ethics write-up.
- Honest error reporting. No 100% accuracy claim.

## Out of scope

Deep learning, production SSO, and live school information systems. Those are listed as future work in the ethics document.
