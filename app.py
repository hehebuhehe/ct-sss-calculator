from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"
ASSET_STEM = "final_model_c_primary_schemeA_seed2036"


st.set_page_config(
    page_title="SSS Web Calculator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_assets():
    with (MODEL_DIR / f"{ASSET_STEM}_pipeline.pkl").open("rb") as f:
        pipeline = pickle.load(f)
    with (MODEL_DIR / f"{ASSET_STEM}_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (MODEL_DIR / f"{ASSET_STEM}_metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, model, metadata


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


pipeline, model, metadata = load_assets()
feature_order = metadata["feature_order"]
thresholds = metadata["exploratory_risk_thresholds"]
defaults = metadata["defaults"]
coding = metadata["categorical_coding"]
feature_help = metadata.get("feature_help", {})

st.title("CT-defined spine-specific sarcopenia (SSS) calculator")
st.caption(f"Primary deployed model: {metadata['model_name']}")
st.warning("For research use only. Not for standalone clinical decision-making.")


with st.form("sss_calculator_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age (years)", min_value=18.0, max_value=100.0, value=float(defaults.get("age", 59.0)), step=1.0)
        sex_default = "Male" if int(defaults.get("sex", 0)) == 1 else "Female"
        sex_label = st.radio("Sex", options=["Female", "Male"], index=0 if sex_default == "Female" else 1, horizontal=True)
        bmi = st.number_input("BMI (kg/m2)", min_value=10.0, max_value=50.0, value=float(defaults.get("BMI", 25.0)), step=0.1)
        diabetes_default = "Yes" if int(defaults.get("diabetes", 0)) == 1 else "No"
        diabetes_label = st.radio("Diabetes", options=["No", "Yes"], index=1 if diabetes_default == "Yes" else 0, horizontal=True)
        hypertension_default = "Yes" if int(defaults.get("hypertension", 0)) == 1 else "No"
        hypertension_label = st.radio("Hypertension", options=["No", "Yes"], index=1 if hypertension_default == "Yes" else 0, horizontal=True)
        smoke_default = "Yes" if int(defaults.get("smoke", 0)) == 1 else "No"
        smoke_label = st.radio("Current smoking", options=["No", "Yes"], index=1 if smoke_default == "Yes" else 0, horizontal=True)

    with col2:
        st.subheader("Laboratory")
        albumin = st.number_input("Albumin (g/L)", min_value=20.0, max_value=60.0, value=float(defaults.get("albumin", 44.0)), step=0.1)
        crp = None
        if "CRP" in feature_order:
            crp = st.number_input("C-reactive protein (mg/L)", min_value=0.0, max_value=30.0, value=float(defaults.get("CRP", 2.5)), step=0.1)
        cally_index = st.number_input("CALLY index", min_value=0.0, max_value=20.0, value=float(defaults.get("cally_index", 3.0)), step=0.1)

    with col3:
        st.subheader("L3 CT-derived features")
        l3_psmi = st.number_input("L3 PSMI (cm2/m2)", min_value=0.0, max_value=30.0, value=float(defaults.get("L3_PSMI", 10.0)), step=0.1)
        l3_psmd = st.number_input("L3 PSMD (HU)", min_value=0.0, max_value=80.0, value=float(defaults.get("L3_PSMD", 36.0)), step=0.1)
        l3_fi = st.number_input("L3 FI", min_value=0.0, max_value=1.0, value=float(defaults.get("L3_FI", 0.13)), step=0.01, format="%.2f", help=feature_help.get("L3_FI"))
        l3_pmd = st.number_input("L3 PMD (HU)", min_value=0.0, max_value=80.0, value=float(defaults.get("L3_PMD", 41.0)), step=0.1)
        normal_muscle_pct = st.number_input(
            "Normal muscle area percentage (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(defaults.get("normal_muscle_area_percent", 72.0)),
            step=0.1,
        )
        central_ratio = st.number_input(
            "Central adiposity-to-muscle ratio",
            min_value=0.0,
            max_value=10.0,
            value=float(defaults.get("central_adiposity_to_muscle", 0.80)),
            step=0.01,
            format="%.2f",
            help=feature_help.get("central_adiposity_to_muscle"),
        )

    submitted = st.form_submit_button("Calculate SSS probability", use_container_width=True)


if submitted:
    values: dict[str, float | int] = {
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
    if "CRP" in feature_order:
        values["CRP"] = float(crp)

    X = build_input_frame(values, feature_order)
    prob = float(pipeline.predict_proba(X)[:, 1][0])
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

    with st.expander("Model summary", expanded=False):
        vm = metadata["validation_metrics"]
        st.markdown(
            f"""
**Primary model**: {metadata['model_name']}  
**Selection strategy**: {metadata['selection_strategy']}  
**Validation AUROC**: {vm['auroc']:.3f} ({vm['auroc_ci_low']:.3f}-{vm['auroc_ci_high']:.3f})  
**Validation AUPRC**: {vm['auprc']:.3f}  
**Validation Brier score**: {vm['brier_score']:.3f}
            """.strip()
        )

    with st.expander("Manuscript-ready text snippets", expanded=False):
        st.markdown(
            """
**Methods subsection**  
An anonymous Streamlit-based web calculator was developed for the primary Model C logistic regression selected under the scheme A workflow. The calculator loads the saved preprocessing object and trained model from disk and applies the exact predictor set and feature order retained after missingness screening, collinearity filtering, and training-only model selection. User inputs include demographic, laboratory, and L3 CT-derived variables used in the deployed primary model. The calculator returns the predicted probability of CT-defined spine-specific sarcopenia (SSS) together with an exploratory risk category. The interface is intended for research use only and not for standalone clinical decision-making.

**Results sentence**  
To facilitate transparent research deployment, we implemented an anonymous web calculator for the primary Model C logistic regression that returns the predicted probability of CT-defined SSS using the exact deployed preprocessing workflow and predictor set.

**Figure caption**  
Screenshot of the anonymous Streamlit-based web calculator for the primary Model C logistic regression predicting CT-defined spine-specific sarcopenia (SSS). The interface accepts demographic, laboratory, and L3 CT-derived inputs and returns the predicted probability of SSS together with an exploratory risk category.
            """.strip()
        )


st.markdown("---")
st.caption("For research use only. Not for standalone clinical decision-making.")
