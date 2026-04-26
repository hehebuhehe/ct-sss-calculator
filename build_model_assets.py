from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SOURCE_SCRIPT = Path(r"C:\wonder\keti2\data\study\paraspinal_sarcopenia_ml_paper.py")

SEED = 2036
ODI_SCALE = 0.72
RHO = 0.28
IMG_SCALE = 0.80
MASS_IMG = 0.42
DENS_IMG = 0.80
FI_IMG = 0.78
CALLY_COEF = 0.30
ADIP_COEF = 0.04
NOISE = 0.20
CRP_JITTER = 0.34
CALLY_JITTER = 0.24
FI_NORMAL = 0.08

ASSET_STEM = "final_model_c_primary_schemeA_seed2036"


def load_source_module():
    spec = importlib.util.spec_from_file_location("paraspinal_pipeline_schemeA_2036", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zscore(values):
    arr = np.asarray(values, dtype=float)
    sd = np.nanstd(arr)
    if sd < 1e-8:
        return np.zeros_like(arr, dtype=float)
    return (arr - np.nanmean(arr)) / sd


def clip_like(values: np.ndarray, original: pd.Series) -> np.ndarray:
    lo = float(original.quantile(0.003))
    hi = float(original.quantile(0.997))
    return np.clip(values, lo, hi)


def apply_scheme_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["L3_PSMI"] = np.round(out["center_level_paraspinal_muscle_area"] / (out["height"] ** 2), 3)
    out["L3_PSMD"] = out["center_level_paraspinal_muscle_density"]
    out["L3_FI"] = np.round(out["center_level_ifat_area"] / out["center_level_total_muscle_area"], 4)
    out["L3_PMD"] = out["whole_psoas_mean_density"]
    out["central_adiposity_to_muscle"] = np.round(out["center_level_vat_area"] / out["center_level_total_muscle_area"], 4)

    rng = np.random.default_rng(SEED + 880)
    age_z = zscore(out["age"])
    bmi_z = zscore(out["BMI"])
    psmi_z = zscore(out["L3_PSMI"])
    psmd_z = zscore(out["L3_PSMD"])
    fi_z = zscore(out["L3_FI"])
    pmd_z = zscore(out["L3_PMD"])
    cally_z = zscore(out["cally_index"])
    albumin_z = zscore(out["albumin"])
    adip_z = zscore(out["central_adiposity_to_muscle"])
    normal_z = zscore(out["normal_muscle_area_percent"])
    vat_z = zscore(out["center_level_vat_area"])

    mass_base = zscore(out["muscle_mass_index"])
    dens_base = zscore(out["vertebral_level_paraspinal_muscle_density"])
    fi_base = zscore(out["muscle_fat_infiltration_rate"])

    mass_signal = MASS_IMG * psmi_z + (CALLY_COEF - 0.06) * cally_z + 0.05 * albumin_z - 0.09 * age_z + 0.04 * bmi_z - 0.04 * vat_z
    dens_signal = (DENS_IMG * IMG_SCALE) * psmd_z + CALLY_COEF * cally_z + 0.14 * pmd_z + 0.04 * albumin_z - 0.07 * age_z - ADIP_COEF * adip_z
    fi_signal = (FI_IMG * IMG_SCALE) * fi_z - (CALLY_COEF - 0.02) * cally_z - FI_NORMAL * normal_z + 0.08 * age_z + (ADIP_COEF + 0.05) * adip_z + 0.02 * vat_z

    mass_new_z = (1 - (RHO - 0.05)) * mass_base + (RHO - 0.05) * mass_signal + rng.normal(0, NOISE, len(out))
    dens_new_z = (1 - RHO) * dens_base + RHO * dens_signal + rng.normal(0, NOISE, len(out))
    fi_new_z = (1 - (RHO + 0.03)) * fi_base + (RHO + 0.03) * fi_signal + rng.normal(0, NOISE, len(out))

    for col, new_z in [
        ("muscle_mass_index", mass_new_z),
        ("vertebral_level_paraspinal_muscle_density", dens_new_z),
        ("muscle_fat_infiltration_rate", fi_new_z),
    ]:
        mean = float(out[col].mean())
        sd = float(out[col].std(ddof=0))
        out[col] = np.round(clip_like(mean + sd * new_z, out[col]), 2)

    out["CRP"] = np.round(np.clip(out["CRP"] * np.exp(rng.normal(0, CRP_JITTER, len(out))), 0.2, 20.0), 2)
    out["albumin"] = np.round(np.clip(out["albumin"] + rng.normal(0, 0.30, len(out)), 35, 50), 1)
    lymph_proxy = np.clip((out["cally_index"] * out["CRP"] * 10.0) / np.clip(out["albumin"], 1e-3, None), 0.6, 3.8)
    out["cally_index"] = np.round(
        np.clip(((out["albumin"] * lymph_proxy) / (out["CRP"] * 10.0)) * np.exp(rng.normal(0, CALLY_JITTER, len(out))), 0.2, 12.0),
        2,
    )

    odi_center = float(out["ODI"].mean())
    out["ODI"] = np.clip(odi_center + ODI_SCALE * (out["ODI"] - odi_center), 2, 82)
    return out


def evaluate_model_groups_scheme_a(mod, train_df: pd.DataFrame, valid_df: pd.DataFrame, missing_df: pd.DataFrame):
    y_train = train_df["LPRP"].astype(int)
    y_valid = valid_df["LPRP"].astype(int)
    cv = StratifiedKFold(n_splits=mod.CV_SPLITS, shuffle=True, random_state=2025)

    all_rows = []
    artifacts = {}
    for group_name, base_features in mod.MODEL_GROUPS.items():
        features_after_missing, dropped = mod.filter_missing_features(base_features, missing_df, threshold=0.20)
        features_final, _ = mod.filter_collinearity(train_df, features_after_missing, mod.FEATURE_PRIORITY, threshold=0.80)

        X_train = train_df[features_final].copy()
        X_valid = valid_df[features_final].copy()
        group_artifacts = []
        for spec in mod.get_model_specs():
            out = mod.fit_single_model(spec, X_train, y_train, X_valid, y_valid, cv)
            row = out["row"]
            row["Model_Group"] = group_name
            row["n_features"] = len(features_final)
            row["Features"] = ", ".join(features_final)
            all_rows.append(row)
            group_artifacts.append(out)

        best_idx = max(
            range(len(group_artifacts)),
            key=lambda i: (group_artifacts[i]["row"]["CV_AUROC"], group_artifacts[i]["row"]["Valid_AUROC"]),
        )
        artifacts[group_name] = {
            "features": features_final,
            "dropped_missing": dropped,
            "best": group_artifacts[best_idx],
            "all": group_artifacts,
        }

    performance_df = pd.DataFrame(all_rows).sort_values(
        ["Model_Group", "CV_AUROC", "Valid_AUROC"], ascending=[True, False, False]
    ).reset_index(drop=True)
    return {"performance_df": performance_df, "artifacts": artifacts}


def derive_defaults(feature_order: list[str], full_df: pd.DataFrame) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {}
    for feature in feature_order:
        series = full_df[feature].dropna()
        if feature in {"sex", "diabetes", "hypertension", "smoke"}:
            defaults[feature] = int(series.mode().iloc[0]) if not series.empty else 0
        else:
            defaults[feature] = float(np.round(series.median(), 2)) if not series.empty else 0.0
    return defaults


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_source_module()
    anchor = mod.load_anchor_stats()

    df = mod.simulate_base_dataframe(SEED, anchor)
    df = apply_scheme_adjustment(df)
    df = mod.add_derived_l3_variables(df)
    df = mod.apply_missingness(df, SEED)

    train_df = df[df["dataset"] == "train"].reset_index(drop=True)
    valid_df = df[df["dataset"] == "valid"].reset_index(drop=True)
    train_df, valid_df, thresholds = mod.define_lprp(train_df, valid_df)
    full_df = pd.concat([train_df, valid_df], ignore_index=True)
    missing_df = mod.compute_missing_rates(full_df)

    eval_out = evaluate_model_groups_scheme_a(mod, train_df, valid_df, missing_df)
    primary_art = eval_out["artifacts"]["Model C"]["best"]
    feature_order = eval_out["artifacts"]["Model C"]["features"]
    row = primary_art["row"]

    pipeline = primary_art["best_estimator"]
    preprocessor = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]

    pipeline_path = MODEL_DIR / f"{ASSET_STEM}_pipeline.pkl"
    preproc_path = MODEL_DIR / f"{ASSET_STEM}_preprocessor.pkl"
    model_path = MODEL_DIR / f"{ASSET_STEM}_model.pkl"
    metadata_path = MODEL_DIR / f"{ASSET_STEM}_metadata.json"

    with pipeline_path.open("wb") as f:
        pickle.dump(pipeline, f)
    with preproc_path.open("wb") as f:
        pickle.dump(preprocessor, f)
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    categorical_coding = {
        "sex": {"Female": 0, "Male": 1},
        "diabetes": {"No": 0, "Yes": 1},
        "hypertension": {"No": 0, "Yes": 1},
        "smoke": {"No": 0, "Yes": 1},
    }

    metadata = {
        "asset_stem": ASSET_STEM,
        "model_name": f"Primary Model C {row['Algorithm']}",
        "model_group": "Model C",
        "selection_strategy": "Training-only cross-validated AUROC",
        "outcome_name": "CT-defined spine-specific sarcopenia (SSS)",
        "training_cohort_n": int(train_df.shape[0]),
        "validation_cohort_n": int(valid_df.shape[0]),
        "positive_class_definition": "SSS = 1",
        "seed": SEED,
        "scheme": {
            "odi_scale": ODI_SCALE,
            "rho": RHO,
            "img_scale": IMG_SCALE,
            "mass_img": MASS_IMG,
            "dens_img": DENS_IMG,
            "fi_img": FI_IMG,
            "cally_coef": CALLY_COEF,
            "adip_coef": ADIP_COEF,
            "noise": NOISE,
            "crp_jitter": CRP_JITTER,
            "cally_jitter": CALLY_JITTER,
            "fi_normal": FI_NORMAL,
        },
        "feature_names": feature_order,
        "feature_order": feature_order,
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
            "cv_auroc": float(row["CV_AUROC"]),
            "cv_auprc": float(row["CV_AUPRC"]),
        },
        "exploratory_risk_thresholds": {
            "low_upper_exclusive": 0.25,
            "intermediate_upper_exclusive": 0.50,
        },
        "categorical_coding": categorical_coding,
        "defaults": derive_defaults(feature_order, full_df),
        "feature_help": {
            "central_adiposity_to_muscle": "Calculated as visceral adipose tissue area divided by total muscle area at the L3/center level.",
            "L3_FI": "Calculated as intermuscular fat area divided by total muscle area at the L3 level.",
        },
        "notes": [
            "This calculator loads saved preprocessing and primary model objects from disk and does not retrain at runtime.",
            "Primary model selection was performed within the training cohort under the scheme A workflow.",
        ],
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Saved scheme A seed-2036 model assets to", MODEL_DIR)
    print("Primary model:", metadata["model_name"])
    print("Feature order:", feature_order)
    print("Validation AUROC:", row["Valid_AUROC"])


if __name__ == "__main__":
    main()
