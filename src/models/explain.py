"""Global and local explanations. SHAP is preferred; permutation importance is the fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_table(estimator, X, y, feature_names, scoring, n_repeats=8, random_state=42) -> pd.DataFrame:
    result = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return table


def shap_global_and_local(pipeline, X_sample: pd.DataFrame, n_local: int = 3) -> dict[str, Any]:
    """Tree SHAP when the estimator supports it; otherwise skip (permutation importance still runs)."""
    try:
        import shap
    except ImportError:
        return {"available": False, "reason": "shap is not installed"}

    try:
        model = pipeline.named_steps["model"]
        pre = pipeline.named_steps["preprocess"]
        X_trans = np.asarray(pre.transform(X_sample))
        names = list(pre.get_feature_names_out())
        if type(model).__name__ not in {
            "RandomForestRegressor",
            "RandomForestClassifier",
            "GradientBoostingRegressor",
            "GradientBoostingClassifier",
            "DecisionTreeRegressor",
            "DecisionTreeClassifier",
            "XGBRegressor",
            "XGBClassifier",
        }:
            return {"available": False, "reason": f"Tree SHAP not configured for {type(model).__name__}"}

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        values = np.abs(shap_values).mean(axis=0)
        global_imp = (
            pd.DataFrame({"feature": names, "mean_abs_shap": values.tolist()})
            .sort_values("mean_abs_shap", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
        locals_out = []
        for i in range(min(n_local, len(X_sample))):
            row_vals = shap_values[i]
            top_idx = np.argsort(np.abs(row_vals))[::-1][:8]
            locals_out.append(
                {
                    "row_index": int(X_sample.index[i]),
                    "contributions": [
                        {"feature": names[j], "shap": float(row_vals[j])} for j in top_idx
                    ],
                }
            )
        return {"available": True, "global": global_imp, "local": locals_out}
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def simple_local_contributions(pipeline, X_row: pd.DataFrame, background: pd.DataFrame) -> list[dict]:
    """Fast fallback: how much does each raw feature move the prediction vs the background mean?"""
    base = float(pipeline.predict(background).mean())
    pred = float(pipeline.predict(X_row)[0])
    contributions = []
    for col in X_row.columns:
        swapped = X_row.copy()
        swapped[col] = background[col].iloc[0] if col in background else swapped[col]
        # replace with column mean / mode of background
        if pd.api.types.is_numeric_dtype(background[col]):
            swapped[col] = background[col].mean()
        else:
            swapped[col] = background[col].mode().iloc[0]
        new_pred = float(pipeline.predict(swapped)[0])
        contributions.append({"feature": col, "delta_if_typical": float(pred - new_pred)})
    contributions.sort(key=lambda d: abs(d["delta_if_typical"]), reverse=True)
    return [{"prediction": pred, "background_mean": base, "top": contributions[:10]}]
