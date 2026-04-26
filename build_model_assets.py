from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SOURCE_SCRIPT = Path(r"C:\wonder\keti2\data\study\paraspinal_sarcopenia_ml_paper.py")
TRAIN_CSV = Path(r"C:\Users\Wonder\paraspinal_lprp_ml\train_630.csv")
VALID_CSV = Path(r"C:\Users\Wonder\paraspinal_lprp_ml\valid_100.csv")
MISSING_CSV = Path(r"C:\Users\Wonder\paraspinal_lprp_ml\tables\missing_rate_table.csv")


def load_source_module():
    spec = importlib.util.spec_from_file_location("paraspinal_pipeline", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_source_module()
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    missing_df = pd.read_csv(MISSING_CSV)

    base_features = mod.MODEL_GROUPS["Model C"]
    features_after_missing, dropped_missing = mod.filter_missing_features(base_features, missing_df, threshold=0.20)
    features_final, dropped_corr_df = mod.filter_collinearity(
        train_df, features_after_missing, mod.FEATURE_PRIORITY, threshold=0.80
    )

    rf_spec = next(spec for spec in mod.get_model_specs() if spec.name == "Random Forest")
    cv = StratifiedKFold(n_splits=mod.CV_SPLITS, shuffle=True, random_state=2025)
    fit_out = mod.fit_single_model(
        rf_spec,
        train_df[features_final].copy(),
        train_df["LPRP"].astype(int),
        valid_df[features_final].copy(),
        valid_df["LPRP"].astype(int),
        cv,
    )
    pipeline = fit_out["best_estimator"]
    preprocessor = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    row = fit_out["row"]

    with (MODEL_DIR / "final_model_c_random_forest_pipeline.pkl").open("wb") as f:
        pickle.dump(pipeline, f)
    with (MODEL_DIR / "final_model_c_random_forest_preprocessor.pkl").open("wb") as f:
        pickle.dump(preprocessor, f)
    with (MODEL_DIR / "final_model_c_random_forest_model.pkl").open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_name": "Model C Random Forest",
        "outcome_name": "CT-defined spine-specific sarcopenia (SSS)",
        "training_cohort_n": int(train_df.shape[0]),
        "validation_cohort_n": int(valid_df.shape[0]),
        "positive_class_definition": "SSS = 1",
        "feature_names": features_final,
        "feature_order": features_final,
        "dropped_missing": dropped_missing,
        "dropped_collinearity": dropped_corr_df.to_dict(orient="records"),
        "removed_candidate_features": ["CRP"] if "CRP" not in features_final else [],
        "best_params": json.loads(row["BestParams"]),
        "validation_metrics": {
            "auroc": float(row["Valid_AUROC"]),
            "auprc": float(row["Valid_AUPRC"]),
            "sensitivity": float(row["Valid_Sensitivity"]),
            "specificity": float(row["Valid_Specificity"]),
            "accuracy": float(row["Valid_Accuracy"]),
            "f1_score": float(row["Valid_F1"]),
            "brier_score": float(row["Valid_Brier"]),
            "auroc_ci_low": float(row["Valid_AUROC_CI_low"]),
            "auroc_ci_high": float(row["Valid_AUROC_CI_high"]),
        },
        "exploratory_risk_thresholds": {
            "low_upper_exclusive": 0.25,
            "intermediate_upper_exclusive": 0.50,
        },
        "categorical_coding": {
            "sex": {"Female": 0, "Male": 1},
            "diabetes": {"No": 0, "Yes": 1},
            "hypertension": {"No": 0, "Yes": 1},
            "smoke": {"No": 0, "Yes": 1},
        },
        "notes": [
            "CRP was a candidate predictor in the original Model C specification but was removed after collinearity screening in favor of cally_index.",
            "The deployed web calculator loads saved preprocessing and model objects from disk and does not retrain within the application runtime.",
        ],
    }
    with (MODEL_DIR / "final_model_c_random_forest_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved model assets to", MODEL_DIR)
    print("Feature order:", features_final)
    print("Validation AUROC:", row["Valid_AUROC"])


if __name__ == "__main__":
    main()
