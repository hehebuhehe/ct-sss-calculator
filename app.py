from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"


st.set_page_config(
    page_title="SSS Web Calculator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_assets():
    with (MODEL_DIR / "final_model_c_random_forest_preprocessor.pkl").open("rb") as f:
        preprocessor = pickle.load(f)
    with (MODEL_DIR / "final_model_c_random_forest_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (MODEL_DIR / "final_model_c_random_forest_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return preprocessor, model, metadata


def risk_category(probability: float, thresholds: dict[str, float]) -> str:
    if probability < thresholds["low_upper_exclusive"]:
        return "Low exploratory risk"
    if probability < thresholds["intermediate_upper_exclusive"]:
        return "Intermediate exploratory risk"
    return "High exploratory risk"


def build_input_frame(values: dict[str, float | int], feature_order: list[str]) -> pd.DataFrame:
    row = {feature: values[feature] for feature in feature_order}
    return pd.DataFrame([row], columns=feature_order)


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


preprocessor, model, metadata = load_assets()
feature_order = metadata["feature_order"]
thresholds = metadata["exploratory_risk_thresholds"]

st.title("CT-defined spine-specific sarcopenia (SSS) calculator")
st.caption("Final deployed model: Model C Random Forest")
st.warning("For research use only. Not for standalone clinical decision-making.")


defaults = {
    "age": 59.3,
    "BMI": 24.9,
    "albumin": 44.4,
    "cally_index": 3.1,
    "L3_PSMI": 10.3,
    "L3_PSMD": 36.5,
    "L3_FI": 0.13,
    "L3_PMD": 41.8,
    "normal_muscle_area_percent": 72.0,
    "central_adiposity_to_muscle": 0.80,
}


with st.form("sss_calculator_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=float(defaults["age"]), step=1.0)
        sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=float(defaults["BMI"]), step=0.1)
        diabetes_label = st.radio("Diabetes", options=["No", "Yes"], horizontal=True)
        hypertension_label = st.radio("Hypertension", options=["No", "Yes"], horizontal=True)
        smoke_label = st.radio("Current smoking", options=["No", "Yes"], horizontal=True)

    with col2:
        st.subheader("Laboratory")
        albumin = st.number_input("Albumin (g/L)", min_value=20.0, max_value=60.0, value=float(defaults["albumin"]), step=0.1)
        cally_index = st.number_input("CALLY index", min_value=0.0, max_value=20.0, value=float(defaults["cally_index"]), step=0.1)

    with col3:
        st.subheader("L3 CT-derived features")
        l3_psmi = st.number_input("L3 PSMI (cm²/m²)", min_value=0.0, max_value=30.0, value=float(defaults["L3_PSMI"]), step=0.1)
        l3_psmd = st.number_input("L3 PSMD (HU)", min_value=0.0, max_value=80.0, value=float(defaults["L3_PSMD"]), step=0.1)
        l3_fi = st.number_input("L3 FI", min_value=0.0, max_value=1.0, value=float(defaults["L3_FI"]), step=0.01, format="%.2f")
        l3_pmd = st.number_input("L3 PMD (HU)", min_value=0.0, max_value=80.0, value=float(defaults["L3_PMD"]), step=0.1)
        normal_muscle_pct = st.number_input(
            "Normal muscle area percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults["normal_muscle_area_percent"]),
            step=0.1,
        )
        central_ratio = st.number_input(
            "Central adiposity-to-muscle ratio",
            min_value=0.0,
            max_value=10.0,
            value=float(defaults["central_adiposity_to_muscle"]),
            step=0.01,
            format="%.2f",
            help="Calculated as visceral adipose tissue area divided by total muscle area at the center/L3 level.",
        )

    submitted = st.form_submit_button("Calculate SSS probability", use_container_width=True)


if submitted:
    coding = metadata["categorical_coding"]
    values = {
        "age": float(age),
        "sex": int(coding["sex"][sex_label]),
        "BMI": float(bmi),
        "L3_PSMI": float(l3_psmi),
        "L3_PSMD": float(l3_psmd),
        "L3_FI": float(l3_fi),
        "L3_PMD": float(l3_pmd),
        "normal_muscle_area_percent": float(normal_muscle_pct),
        "central_adiposity_to_muscle": float(central_ratio),
        "diabetes": int(coding["diabetes"][diabetes_label]),
        "hypertension": int(coding["hypertension"][hypertension_label]),
        "smoke": int(coding["smoke"][smoke_label]),
        "albumin": float(albumin),
        "cally_index": float(cally_index),
    }
    X = build_input_frame(values, feature_order)
    X_imp = preprocessor.transform(X)
    prob = float(model.predict_proba(X_imp)[:, 1][0])
    category = risk_category(prob, thresholds)
    model_pred = "Predicted SSS" if prob >= 0.50 else "Predicted non-SSS"

    st.markdown("## Prediction")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Predicted probability of CT-defined SSS", f"{prob:.1%}")
    with m2:
        metric_card("Exploratory risk category", category)
    with m3:
        metric_card("Model decision at 0.50 threshold", model_pred)

    st.progress(min(max(prob, 0.0), 1.0))

    with st.expander("Input record used for prediction", expanded=False):
        st.dataframe(X, use_container_width=True)

    with st.expander("Manuscript-ready text snippets", expanded=False):
        st.markdown(
            """
**Methods subsection**  
An anonymous Streamlit-based web calculator was developed for the final Model C random forest. The calculator loads the saved preprocessing object and trained model from disk and applies the exact predictor set and feature order used in the final deployed model. User inputs include demographic, laboratory, and L3 CT-derived variables retained in the final model after preprocessing and collinearity screening. The calculator outputs the predicted probability of CT-defined spine-specific sarcopenia (SSS) together with an exploratory risk category. The web interface is intended for research use only and not for standalone clinical decision-making.

**Results sentence**  
To facilitate transparent research deployment, we implemented an anonymous web calculator for the final Model C random forest that returns the predicted probability of CT-defined SSS using the exact deployed predictor set and preprocessing workflow.

**Figure caption**  
Screenshot of the anonymous Streamlit-based web calculator for the final Model C random forest predicting CT-defined spine-specific sarcopenia (SSS). The interface accepts demographic, laboratory, and L3 CT-derived inputs and returns the predicted probability of SSS together with an exploratory risk category.
            """.strip()
        )


st.markdown("---")
st.caption("For research use only. Not for standalone clinical decision-making.")
