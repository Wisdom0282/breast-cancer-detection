# Breast Cancer Detection: Supervised Classification

A supervised machine learning project that classifies breast tumours as **malignant** or **benign** from measurements of fine needle aspirate (FNA) images.

> **Disclaimer:** educational project only. It is not a medical device and must not be used for diagnosis.

## Problem

| Item | Detail |
|---|---|
| Task | Binary classification (supervised) |
| Dataset | Wisconsin Diagnostic Breast Cancer (WDBC), bundled with scikit-learn |
| Size | 569 samples, 30 numeric features, no missing values |
| Classes | 357 benign (63%), 212 malignant (37%) |
| Positive class | **Malignant** (so recall = sensitivity to cancer) |
| Features | Mean, standard error and "worst" value of 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |

Note: scikit-learn encodes malignant as 0. This project flips it so that malignant = 1, which makes precision/recall/F1 refer to the clinically important class.

## Method

1. **EDA**: class balance, correlation heatmap, distributions of the most predictive features.
2. **Split**: stratified 80/20 train/test (455 / 114), seed 42. The test set is used once, at the end.
3. **Preprocessing**: `StandardScaler` inside a `Pipeline` for scale-sensitive models, so no information leaks from validation folds.
4. **Model comparison**: Logistic Regression, SVM (RBF), KNN, Random Forest, Gradient Boosting, using stratified 5-fold CV on the training set.
5. **Tuning**: `GridSearchCV` (scoring = ROC AUC) for Logistic Regression, SVM and Random Forest.
6. **Selection**: best tuned model by cross-validated AUC (not by test score).
7. **Evaluation**: confusion matrix, precision, recall, specificity, F1, ROC and PR curves, permutation importance.

## Results

### Cross-validation (training set, 5-fold, untuned baselines)

| Model | Accuracy | Recall (malignant) | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 0.974 | 0.953 | 0.964 | 0.996 |
| SVM (RBF) | 0.976 | 0.959 | 0.967 | 0.995 |
| Gradient Boosting | 0.967 | 0.947 | 0.955 | 0.991 |
| Random Forest | 0.960 | 0.935 | 0.946 | 0.988 |
| KNN | 0.965 | 0.918 | 0.951 | 0.987 |

After tuning, SVM (C=100, gamma=0.001, sigmoid-calibrated) reached CV AUC 0.9960 and Logistic Regression (C=1) 0.9958. They are effectively tied, and SVM was selected by a very small margin.

### Held-out test set (114 samples), selected model: SVM (RBF)

| Metric | Value |
|---|---|
| Accuracy | 0.974 |
| Precision (malignant) | 1.000 |
| Recall / sensitivity (malignant) | 0.929 |
| Specificity | 1.000 |
| F1 (malignant) | 0.963 |
| ROC AUC | 0.996 |
| Average precision | 0.994 |

Confusion matrix: 72 true negatives, 0 false positives, **3 false negatives**, 39 true positives.

### Honest caveats

- **The test set is small.** 3 missed cancers out of 42 is 7.1% of malignant cases; with so few samples, the confidence interval around recall is wide. Treat all test numbers as approximate.
- **Model differences are within noise.** CV standard deviations (see `reports/figures/04_model_comparison.png`) overlap across the top models. Logistic Regression is simpler and more interpretable and would be a perfectly defensible choice.
- **False negatives matter most clinically.** The default 0.5 threshold gave perfect precision but missed 3 cancers. In a real screening setting you would lower the threshold to raise recall and accept more false positives. The saved model stores the threshold in the bundle so it is easy to change.
- **Permutation importance values are small** because many features are strongly correlated (radius, perimeter and area), so shuffling one is compensated by the others. Read the ranking as indicative, not definitive.
- WDBC is a clean, well-studied benchmark. Performance on messier real-world data would be lower.

## Project structure

```
breast-cancer-detection/
├── train.py                 # full pipeline: EDA, CV, tuning, evaluation, saving
├── predict.py               # load model and classify new samples
├── requirements.txt
├── models/
│   └── breast_cancer_model.joblib
└── reports/
    ├── results.json         # all metrics and best hyperparameters
    ├── cv_model_comparison.csv
    ├── train_log.txt
    └── figures/             # 7 plots (EDA, comparison, confusion matrix, ROC/PR, importance)
```

## Usage

```bash
pip install -r requirements.txt
python train.py                      # trains and writes models/ and reports/
python predict.py                    # demo on 5 random dataset rows
python predict.py --csv new.csv      # CSV with the 30 WDBC feature columns
```

Example `predict.py` output:

```
     p_malignant prediction     actual
350        0.001     Benign     Benign
259        1.000  Malignant  Malignant
```

## Ideas to extend

- Lower the decision threshold and report the recall/precision trade-off explicitly.
- Add repeated CV or bootstrap confidence intervals on test metrics.
- Try feature selection or PCA to handle the correlated features.
- Add SHAP explanations for individual predictions.
- Wrap `predict.py` in a small Streamlit or FastAPI app.
