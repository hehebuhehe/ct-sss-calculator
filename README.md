# SSS web calculator

This project provides an anonymous Streamlit-based web calculator for the **primary Model C logistic regression** under the **scheme A / seed=2036** workflow, predicting **CT-defined spine-specific sarcopenia (SSS)**.

## Project structure

```text
sss_calculator/
|-- app.py
|-- build_model_assets.py
|-- requirements.txt
|-- README.md
`-- model/
    |-- final_model_c_primary_schemeA_seed2036_pipeline.pkl
    |-- final_model_c_primary_schemeA_seed2036_preprocessor.pkl
    |-- final_model_c_primary_schemeA_seed2036_model.pkl
    `-- final_model_c_primary_schemeA_seed2036_metadata.json
```

## Key implementation notes

- The web app **does not retrain** the model.
- The app loads the saved preprocessing object and saved trained primary model from `model/`.
- The deployed feature order is taken directly from the exported scheme A metadata.
- The deployed calculator now uses the **primary training-selected Model C logistic regression**, not the earlier random forest comparator.

## Final deployed predictor set

The exported scheme A primary Model C uses the following predictor order:

1. `age`
2. `sex`
3. `BMI`
4. `L3_PSMI`
5. `L3_PSMD`
6. `L3_FI`
7. `L3_PMD`
8. `normal_muscle_area_percent`
9. `central_adiposity_to_muscle`
10. `diabetes`
11. `hypertension`
12. `smoke`
13. `albumin`
14. `CRP`
15. `cally_index`

## Local testing

1. Open a terminal in the `sss_calculator/` folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Rebuild the saved model assets:

```bash
python build_model_assets.py
```

4. Launch the local app:

```bash
streamlit run app.py
```

## Output

The calculator returns:

- predicted probability of CT-defined SSS
- exploratory risk category
- model decision at the 0.50 probability threshold

Suggested exploratory categories:

- Low exploratory risk: probability `< 0.25`
- Intermediate exploratory risk: `0.25 to < 0.50`
- High exploratory risk: `>= 0.50`

## Validation summary

- Primary model: `Model C Logistic Regression`
- Selection strategy: `Training-only cross-validated AUROC`
- Validation AUROC: `0.832`
- Validation AUPRC: `0.641`
- Validation Brier score: `0.203`

## Disclaimer

**For research use only. Not for standalone clinical decision-making.**

## Manuscript-ready text snippets

### Methods subsection

An anonymous Streamlit-based web calculator was developed for the primary Model C logistic regression selected under the scheme A workflow. The calculator loads the saved preprocessing object and trained model from disk and applies the exact predictor set and feature order retained after missingness screening, collinearity filtering, and training-only model selection. User inputs include demographic, laboratory, and L3 CT-derived variables used in the deployed primary model. The calculator outputs the predicted probability of CT-defined spine-specific sarcopenia (SSS) together with an exploratory risk category. The web interface is intended for research use only and not for standalone clinical decision-making.

### Results sentence

To facilitate transparent research deployment, we implemented an anonymous web calculator for the primary Model C logistic regression that returns the predicted probability of CT-defined SSS using the exact deployed preprocessing workflow and predictor set.

### Figure caption

Screenshot of the anonymous Streamlit-based web calculator for the primary Model C logistic regression predicting CT-defined spine-specific sarcopenia (SSS). The interface accepts demographic, laboratory, and L3 CT-derived inputs and returns the predicted probability of SSS together with an exploratory risk category.
