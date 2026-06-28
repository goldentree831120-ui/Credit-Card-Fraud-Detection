# Credit Card Fraud Detection

A machine learning fraud detection system built on real, anonymised European credit card transactions, comparing two models on a genuine precision-recall trade-off relevant to real-world fraud operations.

## Overview

This project detects fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud rate), comparing Logistic Regression and Random Forest not just on accuracy metrics but on the practical business trade-off between catching more fraud (recall) and minimising false alarms on legitimate customers (precision).

## Dataset

**Source:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — Machine Learning Group, Université Libre de Bruxelles (ULB)

284,807 real transactions over 2 days, with 492 confirmed frauds (0.173%). Features `V1`–`V28` are PCA-anonymised for confidentiality; `Time` and `Amount` are the only original, untransformed fields.

## Methodology

1. **EDA** — confirmed zero missing values and visualised class separation across the anonymised features; correlation analysis showed several features (V17, V14, V12) with correlations above 0.3 with the fraud label — a genuinely strong signal, unlike a prior credit risk project on a different dataset where the strongest feature correlation was only 0.02
2. **Preprocessing** — scaled `Time` and `Amount` (the only non-PCA features) with `StandardScaler`; left `V1`–`V28` untouched since they are already standardised by the PCA process
3. **Train/test split** — stratified 80/20 split to preserve the same ~0.17% fraud rate in both sets
4. **Modelling** — trained Logistic Regression (`class_weight='balanced'`) and Random Forest (`class_weight='balanced'`), evaluating with AUC-ROC, precision, recall, and confusion matrices rather than raw accuracy, which is meaningless on a 99.83%-majority-class dataset
5. **Business trade-off analysis** — went beyond AUC to interpret the confusion matrices in terms of real operational cost: false positives mean inconveniencing legitimate customers, false negatives mean undetected fraud losses

## Key results

| Model | AUC | Recall (fraud caught) | Precision (flag accuracy) | False positives | Frauds missed |
|---|---|---|---|---|---|
| Logistic Regression | **0.972** | 91.8% (90/98) | 6.1% | 1,386 | 8 |
| Random Forest | 0.953 | 74.5% (73/98) | **96.1%** | **3** | 25 |

**Top predictive features (Random Forest importance):** V14, V10, V12, V4, and V17 — consistent with the correlation analysis from the EDA step, reinforcing that this is genuine signal rather than overfitting noise.

**Business conclusion:** Although Logistic Regression has the higher AUC and catches more fraud cases, Random Forest is the more practical choice for deployment — it reduces false positives by 99.8% (3 vs 1,386) at the cost of missing 17 additional fraud cases out of nearly 57,000 test transactions. In a real bank, the operational cost of 1,386 false fraud alerts (blocked cards, customer service calls, eroded trust) would likely outweigh the value of catching those extra 17 frauds. This illustrates that the "best" model by AUC is not automatically the best model for the business — the right choice depends on the relative cost of false positives versus false negatives.

## Tech stack

- Python 3
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — Logistic Regression, Random Forest, preprocessing, evaluation metrics
- `matplotlib`, `seaborn` — EDA, ROC/precision-recall curves, confusion matrices

## Repository contents

| File | Description |
|---|---|
| `fraud_detection.ipynb` | Full notebook — EDA, preprocessing, modelling, evaluation, business trade-off analysis |
| `fraud_eda_dashboard.png` | 6-panel EDA — class distribution, amount/time patterns, feature separation, correlation ranking |
| `fraud_model_comparison.png` | ROC curves and precision-recall curves for both models |
| `fraud_confusion_matrices.png` | Side-by-side confusion matrices |
| `fraud_feature_importance.png` | Top 10 features by Random Forest importance |
| `fraud_model_comparison.csv` | Final summary table — AUC, recall, precision, false positives, frauds missed |

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook fraud_detection.ipynb
```

Download `creditcard.csv` from the Kaggle link above and place it in the same folder as the notebook before running. Run all cells in order.

## Notes & limitations

- The dataset's `V1`–`V28` features are PCA-anonymised, so individual feature meanings (e.g. "what is V14?") cannot be interpreted in business terms — only their relative importance can be reported
- The two-day data window is short; a production system would need to be validated across longer time periods and evolving fraud patterns (concept drift)
- Threshold tuning (adjusting the probability cutoff rather than using the default 0.5) could further shift the precision-recall balance and would be a natural next step
- This is an educational/portfolio exercise demonstrating methodology, not a production fraud system
