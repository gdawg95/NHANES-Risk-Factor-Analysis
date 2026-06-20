# NHANES Risk Factor Analysis

Cardiovascular Disease & Asthma prediction analysis based on NHANES dataset (2000–2018).  
This analysis examines specific health markers and lifestyle habits of the adult population, with demographic information (age, sex, etc.) used to predict disease diagnosis.

**Link to Tableau dashboard:** https://public.tableau.com/app/profile/hai.nguyen5235/viz/vis_data/Viz1

*This project was created for the TDP unit assignment and was completed as part of a team.*

---

## Project Overview

The repository contains multiple machine learning models and statistical analyses designed to:
- Predict asthma and cardiovascular disease (CVD) risk factors from NHANES health survey data
- Identify significant associations between lifestyle/demographic variables and disease outcomes
- Compare predictive performance across model families (Random Forest, AdaBoost, KNN, XGBoost, LightGBM, CatBoost, Logistic Regression, Decision Tree, MLP)
- Explain model decisions using SHAP values and feature importance

---

## Key Findings by Analysis

### Predictive Model Performance

#### RandomForestAsthma_v1.1.ipynb
- **ROC AUC:** 0.828 | **PR AUC:** 0.533
- **Accuracy:** 0.870
- **Precision:** 0.970 | **Recall:** 0.033 | **F1:** 0.064
- **Observation:** Excellent overall accuracy and ROC-AUC, but very low recall for the asthma diagnosis class. The high precision shows the model rarely predicts asthma incorrectly, but it misses most true asthma cases—a hallmark of severe class imbalance. Optimize for recall if early detection is the priority.

#### RandomForestCVD_v1.1.ipynb
- **ROC AUC:** 0.935 | **PR AUC:** 0.647
- **Accuracy:** 0.927
- **Precision:** 0.957 | **Recall:** 0.096 | **F1:** 0.175
- **Observation:** Best-performing model by AUC, but again minority-class recall is low. The model learns the healthy class well but fails to capture most CVD cases in the test set. The precision-recall trade-off reflects class imbalance; consider resampling or class weights for improved recall.

#### AdaBoost.ipynb
- **Accuracy:** ~0.62–0.70 (varies by cross-validation fold)
- **ROC AUC:** ~0.672 | **PR AUC:** ~0.256
- **Positive-class F1 (CV):** ~0.27–0.30
- **Observation:** Moderate predictive performance relative to Random Forest and ensemble methods. Demonstrates that boosting achieves reasonable separation between classes but does not overcome the class imbalance disadvantage without additional tuning (e.g., scale_pos_weight, focal loss).

#### Decision Tree (decision-tree.ipynb)
- **Test Accuracy:** ~0.86
- **Observation:** Decision tree achieves high accuracy on the test split but exhibits near-zero recall for the minority class in the snippets shown. Shallow depth (max_depth=5) provides interpretability but sacrifices minority-class detection. Useful for a baseline and for identifying top-level feature thresholds.

#### KNN (KNN.ipynb)
- **Status:** Optuna hyperparameter tuning implemented but final stable metrics not extracted.
- **Note:** Run the notebook end-to-end in a fresh kernel to reproduce tuned results. The Optuna study object and final best-params will be available after all cells execute.

#### XGB, LightGBM, CatBoost, Logistic Regression, MLP (XGB_LGBM_CatBoost_LR_DT_MLP.ipynb)
- **XGBoost (Asthma):** Tuned hyperparameters n_estimators=1034, max_depth=44, learning_rate≈0.0026. Optuna optimization over 30+ trials.
- **CatBoost (Asthma):** Tuned hyperparameters iterations=1385, depth=10, learning_rate≈0.124. Extended hyperparameter search.
- **Observation:** The notebook contains extensive hyperparameter tuning comparisons. Example trials show precision/recall tradeoffs (e.g., Trial #10 achieving precision 0.8711, recall 0.8310 on asthma). SHAP explanations are attempted; successful runs produce feature-importance rankings. Re-run to see full comparison and SHAP visualizations.

---

### Statistical Associations (Stats_analysis.ipynb)

The statistical tests reveal multiple significant associations (p < 0.05) between risk factors and outcomes:

#### Categorical Variables (Chi-Square Tests):

| Variable | Outcome | Chi² | p-value | Significant? |
|----------|---------|------|---------|--------------|
| Lifetime Smoking History | Asthma | 242.45 | <0.0001 | ✓ |
| Lifetime Smoking History | CVD | 1317.42 | <0.0001 | ✓ |
| Work w/ Moderate-Intense Activity | Asthma | 4.68 | 0.0305 | ✓ |
| Work w/ Moderate-Intense Activity | CVD | 85.59 | <0.0001 | ✓ |
| Education Level | Asthma | 441.87 | <0.0001 | ✓ |
| Education Level | CVD | 864.61 | <0.0001 | ✓ |

#### Continuous Variables (Mann-Whitney U Tests):

| Variable | Outcome | p-value | Significant? |
|----------|---------|---------|--------------|
| Binge Drinking Frequency | Asthma | 0.5457 | ✗ |
| Binge Drinking Frequency | CVD | <0.0001 | ✓ |
| Family Monthly Income | Asthma | <0.0001 | ✓ |
| Family Monthly Income | CVD | <0.0001 | ✓ |
| Days Mental Health Not Good (30-day) | Asthma | <0.0001 | ✓ |
| Days Mental Health Not Good (30-day) | CVD | <0.0001 | ✓ |

**Key Takeaway:** Smoking, physical activity level, education, income, and mental health are all statistically linked to both asthma and CVD risk. Binge drinking shows no association with asthma but is significant for CVD—a nuanced finding worth investigating further.

---

## Important Notes & Caveats

### Class Imbalance
Many outcome labels are heavily imbalanced (e.g., ~13% asthma cases, ~8% CVD cases in the full dataset). This results in:
- High accuracy and ROC-AUC but low positive-class recall/F1.
- High precision for the minority class when it is predicted (model is conservative).
- Standard metrics can be misleading; always examine **PR-AUC, F1, and class-weighted metrics**.

### SHAP Explainability Errors
Some notebooks attempt SHAP explanations but encounter an "AssertionError: shap_values shape does not match data" error. This typically occurs because:
- The explainer is passed a different data shape than expected.
- The test data used differs from what was used to generate shap_values.
- **Solution:** Run the notebook start-to-finish in one kernel and ensure the explainer receives the exact test or validation data slice used.

### Hyperparameter Tuning
Several notebooks run Optuna hyperparameter searches with 20–30+ trials:
- Runtime can be significant (some trials run 5–10+ minutes each).
- Results are reproducible only if random seeds are fixed and the same kernel session is used.
- **To reproduce faster:** Load saved best-params from a prior run if available, or reduce n_trials in the Optuna study.

### Data
- **Primary data file:** `filtered_data_v1.3.csv` (preprocessed NHANES data)
- **Columns:** 35 features including age, BMI, smoking status, blood glucose, HDL cholesterol, and more.
- See `data/README.md` (coming soon) for full column descriptions and provenance.

---

## Repository Structure

```
NHANES-Risk-Factor-Analysis/
├── README.md                          # This file
├── RandomForestAsthma_v1.1.ipynb      # Random Forest model for asthma prediction
├── RandomForestCVD_v1.1.ipynb         # Random Forest model for CVD prediction
├── AdaBoost.ipynb                     # AdaBoost ensemble model
├── Decision_Tree.ipynb                # Decision tree baseline
├── KNN.ipynb                          # K-Nearest Neighbors with Optuna tuning
├── XGB_LGBM_CatBoost_LR_DT_MLP.ipynb  # Comparison of multiple model families
├── Stats_analysis.ipynb               # Statistical tests and associations
├── filtered_data_v1.3.csv             # Preprocessed NHANES data
└── (other support files)
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gdawg95/NHANES-Risk-Factor-Analysis.git
   cd NHANES-Risk-Factor-Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap matplotlib seaborn
   ```

3. **Run a notebook:**
   - Open any `.ipynb` file in Jupyter Notebook or JupyterLab.
   - Execute cells top-to-bottom in sequence.
   - For large hyperparameter searches, expect 30–60 minutes or more.

4. **View results:**
   - Check the notebook outputs for confusion matrices, ROC curves, and SHAP plots.
   - Cross-reference metrics from the "Key Findings" section above.

---

## Contributing

See `CONTRIBUTING.md` (coming soon) for branching strategy, PR guidelines, and notebook reproduction tips.

---

## License & Attribution

*This project was created for academic/educational purposes. All NHANES data are publicly available via the CDC.*
