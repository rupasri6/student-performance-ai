# Viva notes (assignment section 13)

Use this as spoken answers. Replace metric numbers with values from `artifacts/models/metrics.json` after training.

1. **Why this problem and dataset?** Schools collect engagement data but intervene late. UCI Student Performance is open, documented, and has no PII.
2. **Target?** Regression: G3. Classification: support bands from G3 (<10 / 10–13 / ≥14).
3. **Leakage?** G1 and G2 dropped. Labels created only from G3. Clustering ignores G3. Test split is held out before tuning conclusions (RandomizedSearchCV uses training folds only).
4. **Metrics?** RMSE/MAE/R² for scores. Macro-F1 for support groups because classes are imbalanced. Accuracy alone would hide missed high-support students.
5. **Imbalance?** We report class counts, use `class_weight="balanced"` on several classifiers, and select by macro-F1 not accuracy.
6. **Influential features?** Permutation importance globally; local deltas or SHAP for one student. Failures and studytime usually dominate — verify on the dashboard.
7. **Uncertainty?** Show class probabilities, residual RMSE, and a robustness run with missing studytime. Teachers can override.
8. **Ethics?** No PII; sex excluded; socioeconomic proxies exist; outputs are support suggestions.
9. **Drift?** Each term, check input shift and predicted-vs-actual G3; retrain if high-support recall drops.
10. **Improvements?** Larger multi-school data, TabNet, LLM-written support letters with a human reviewer.

# Test-results template

Run `python -m pytest -q` and paste the output below for the submission zip.

```
(paste pytest output here)
```

Training metrics live in `artifacts/models/metrics.json`. Do not edit them by hand.
