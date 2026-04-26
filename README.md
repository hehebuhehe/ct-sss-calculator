# SSS web calculator

This project provides an anonymous Streamlit-based web calculator for the **final Model C Random Forest** predicting **CT-defined spine-specific sarcopenia (SSS)**.

## Project structure

```text
sss_calculator/
├── app.py
├── build_model_assets.py
├── requirements.txt
├── README.md
└── model/
    ├── final_model_c_random_forest_pipeline.pkl
    ├── final_model_c_random_forest_preprocessor.pkl
    ├── final_model_c_random_forest_model.pkl
    └── final_model_c_random_forest_metadata.json
```

## Key implementation notes

- The web app **does not retrain** the model.
- The app loads the saved preprocessing object and saved trained model from `model/`.
- The deployed feature order is taken directly from the final exported Model C Random Forest metadata.
- The original candidate variable `CRP` is **not part of the deployed calculator**, because it was removed during collinearity screening in favor of `cally_index`.

## Final deployed predictor set

The final exported Model C Random Forest uses the following predictor order:

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
14. `cally_index`

## Local testing

1. Open a terminal in the `sss_calculator/` folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. If the `model/` folder is still empty, build the saved model assets once:

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

## Disclaimer

**For research use only. Not for standalone clinical decision-making.**

## Manuscript-ready text snippets

### Methods subsection

An anonymous Streamlit-based web calculator was developed for the final Model C random forest. The calculator loads the saved preprocessing object and trained model from disk and applies the exact predictor set and feature order used in the final deployed model. User inputs include demographic, laboratory, and L3 CT-derived variables retained in the final model after preprocessing and collinearity screening. The calculator outputs the predicted probability of CT-defined spine-specific sarcopenia (SSS) together with an exploratory risk category. The web interface is intended for research use only and not for standalone clinical decision-making.

### Results sentence

To facilitate transparent research deployment, we implemented an anonymous web calculator for the final Model C random forest that returns the predicted probability of CT-defined SSS using the exact deployed predictor set and preprocessing workflow.

### Figure caption

Screenshot of the anonymous Streamlit-based web calculator for the final Model C random forest predicting CT-defined spine-specific sarcopenia (SSS). The interface accepts demographic, laboratory, and L3 CT-derived inputs and returns the predicted probability of SSS together with an exploratory risk category.
