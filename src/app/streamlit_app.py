"""Streamlit app: predict, explain, recommend, review metrics, and admin upload."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURE_DIR, MODEL_DIR  # noqa: E402
from src.models.inference import load_artifacts, predict_student  # noqa: E402

st.set_page_config(page_title="Student Performance Advisor", layout="wide")

DEFAULTS = {
    "school": "GP",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 2,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 2,
    "health": 3,
    "absences": 4,
    "subject": "math",
}


def _form_values() -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        school = st.selectbox("School", ["GP", "MS"], index=0)
        age = st.number_input("Age", min_value=15, max_value=22, value=DEFAULTS["age"])
        address = st.selectbox("Address type", ["U", "R"], help="U = urban, R = rural")
        famsize = st.selectbox("Family size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parent cohabitation", ["T", "A"], help="T = together, A = apart")
        Medu = st.slider("Mother education (0-4)", 0, 4, DEFAULTS["Medu"])
        Fedu = st.slider("Father education (0-4)", 0, 4, DEFAULTS["Fedu"])
        Mjob = st.selectbox("Mother job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father job", ["teacher", "health", "services", "at_home", "other"])
    with c2:
        reason = st.selectbox("Reason for school choice", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.slider("Travel time band (1-4)", 1, 4, DEFAULTS["traveltime"])
        studytime = st.slider("Weekly study time band (1-4)", 1, 4, DEFAULTS["studytime"])
        failures = st.slider("Past failures", 0, 4, DEFAULTS["failures"])
        absences = st.number_input("Absences", min_value=0, max_value=93, value=DEFAULTS["absences"])
        famrel = st.slider("Family relationship quality", 1, 5, DEFAULTS["famrel"])
        freetime = st.slider("Free time", 1, 5, DEFAULTS["freetime"])
        goout = st.slider("Going out", 1, 5, DEFAULTS["goout"])
    with c3:
        Dalc = st.slider("Weekday alcohol", 1, 5, DEFAULTS["Dalc"])
        Walc = st.slider("Weekend alcohol", 1, 5, DEFAULTS["Walc"])
        health = st.slider("Health", 1, 5, DEFAULTS["health"])
        schoolsup = st.selectbox("Extra school support", ["no", "yes"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Paid extra classes", ["no", "yes"])
        activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
        nursery = st.selectbox("Attended nursery", ["yes", "no"])
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Home internet", ["yes", "no"])
        romantic = st.selectbox("Romantic relationship", ["no", "yes"])
        subject = st.selectbox("Subject", ["math", "portuguese"])
    return {
        "school": school,
        "age": int(age),
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": int(Medu),
        "Fedu": int(Fedu),
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": int(traveltime),
        "studytime": int(studytime),
        "failures": int(failures),
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": int(famrel),
        "freetime": int(freetime),
        "goout": int(goout),
        "Dalc": int(Dalc),
        "Walc": int(Walc),
        "health": int(health),
        "absences": int(absences),
        "subject": subject,
    }


def page_home():
    st.title("Intelligent Student Performance Advisor")
    st.write(
        "This application estimates a **final course score** (0–20 scale) and a **support-need group**, "
        "explains the main drivers, and suggests learning actions. It is a decision-support tool, "
        "not a verdict on a student’s ability."
    )
    st.info(
        "Sex is never used as a predictor. Prior period grades G1 and G2 are excluded so the model "
        "cannot copy the final exam."
    )
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Best regressor", metrics.get("best_regressor", "—"))
        c2.metric("Best classifier", metrics.get("best_classifier", "—"))
        c3.metric("Test rows", metrics.get("n_test", "—"))
        st.caption(metrics.get("selection_rationale", ""))
    else:
        st.warning("Models are not trained yet. Run `python -m src.models.train` then refresh.")


def page_predict():
    st.title("Student / faculty prediction form")
    st.caption("Enter permitted academic and engagement fields only. Do not enter names, IDs, or contact details.")
    payload = _form_values()
    if st.button("Predict", type="primary"):
        try:
            result = predict_student(payload)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        if not result["ok"]:
            st.error("Please fix the following:")
            for err in result["errors"]:
                st.write(f"- {err}")
            return
        st.session_state["last_result"] = result
        st.session_state["last_payload"] = payload

    result = st.session_state.get("last_result")
    if not result:
        return

    st.subheader("Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted final score", result["predicted_score"])
    c2.metric("Support group", result["predicted_support"].replace("_", " "))
    c3.metric("Engagement cluster", result["engagement_cluster"])
    st.write("Class probabilities", result["class_probabilities"])
    st.warning(result["disclaimer"])

    st.subheader("Why this prediction?")
    top = result.get("explanation", {}).get("top", [])
    if top:
        st.dataframe(pd.DataFrame(top), use_container_width=True)
        st.caption(
            "Positive `delta_if_typical` means this student’s value on that feature raised the predicted "
            "score relative to a typical training student."
        )

    st.subheader("Personalized recommendations")
    for rec in result["recommendations"]:
        st.markdown(f"**{rec['gap']}** — {rec['action']}  \nResource: *{rec['resource']}*")

    summary = {
        "predicted_score": result["predicted_score"],
        "support_group": result["predicted_support"],
        "probabilities": result["class_probabilities"],
        "recommendations": [r["gap"] for r in result["recommendations"]],
        "disclaimer": result["disclaimer"],
    }
    st.download_button(
        "Export prediction summary (JSON)",
        data=json.dumps(summary, indent=2),
        file_name="prediction_summary.json",
        mime="application/json",
    )


def page_metrics():
    st.title("Model performance dashboard")
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        st.warning("Train the models first.")
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Regression (test set)")
    reg_rows = []
    for name, vals in metrics["regression"].items():
        if isinstance(vals, dict) and "rmse" in vals:
            reg_rows.append({"model": name, **{k: vals[k] for k in ("mae", "rmse", "r2") if k in vals}})
    st.dataframe(pd.DataFrame(reg_rows).sort_values("rmse"), use_container_width=True)

    st.subheader("Classification (test set)")
    cls_rows = []
    for name, vals in metrics["classification"].items():
        if isinstance(vals, dict) and "f1_macro" in vals:
            cls_rows.append(
                {
                    "model": name,
                    "accuracy": vals.get("accuracy"),
                    "precision_macro": vals.get("precision_macro"),
                    "recall_macro": vals.get("recall_macro"),
                    "f1_macro": vals.get("f1_macro"),
                    "roc_auc_ovr": vals.get("roc_auc_ovr"),
                }
            )
    st.dataframe(pd.DataFrame(cls_rows).sort_values("f1_macro", ascending=False), use_container_width=True)

    st.subheader("Class balance")
    st.json({"train": metrics.get("class_balance_train"), "test": metrics.get("class_balance_test")})

    st.subheader("Permutation importance (regression)")
    st.dataframe(pd.DataFrame(metrics.get("permutation_importance_regression", [])), use_container_width=True)

    st.subheader("EDA charts")
    for path in sorted(FIGURE_DIR.glob("*.png")):
        st.image(str(path), caption=path.name, use_container_width=True)


def page_admin():
    st.title("Admin — dataset upload and model status")
    metrics_path = MODEL_DIR / "metrics.json"
    st.write("Model files present:" if (MODEL_DIR / "regression_pipeline.joblib").exists() else "No trained model on disk.")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.success(f"Regressor: {metrics['best_regressor']} | Classifier: {metrics['best_classifier']}")
        st.json(metrics.get("data_quality", {}))
    uploaded = st.file_uploader("Upload a CSV with the UCI student schema (semicolon or comma separated)", type=["csv"])
    if uploaded is not None:
        try:
            from io import StringIO

            text = uploaded.getvalue().decode("utf-8")
            sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
            df = pd.read_csv(StringIO(text), sep=sep)
            required = {"age", "studytime", "failures", "absences"}
            missing = required - set(df.columns)
            if missing:
                st.error(f"Upload rejected. Missing columns: {sorted(missing)}")
            else:
                st.success(f"Accepted {len(df)} rows, {df.shape[1]} columns.")
                st.dataframe(df.head(), use_container_width=True)
                st.caption("Uploads are validated in the session only. They are not written to disk and are not used to retrain unless you run the training script.")
        except Exception as exc:
            st.error(f"Could not read file: {exc}")


def page_ethics():
    st.title("Limitations, privacy, and intended use")
    st.markdown(
        """
- Predictions are **probabilistic**. The project never claims 100% accuracy.
- Do not upload names, emails, student IDs, or other personal identifiers.
- Sex was excluded from the model to avoid encoding gender stereotypes as academic ability.
- G1 and G2 were excluded to prevent target leakage.
- Alcohol, health, and family fields are sensitive; treat explanations as prompts for support, not labels.
- After deployment, monitor for **model drift**: grade distributions, absence patterns, and error rates should be reviewed each term.
"""
    )


pages = {
    "Home": page_home,
    "Predict": page_predict,
    "Performance": page_metrics,
    "Admin": page_admin,
    "Ethics": page_ethics,
}

choice = st.sidebar.radio("Navigate", list(pages.keys()))
pages[choice]()
