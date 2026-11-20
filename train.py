"""
Breast cancer detection: supervised binary classification.

Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), bundled with scikit-learn.
569 samples, 30 numeric features computed from digitised images of fine needle
aspirates (FNA) of breast masses. Target: malignant vs benign.

Pipeline:
  1. Load data and relabel so that MALIGNANT = 1 (the positive class).
  2. Exploratory plots.
  3. Stratified train/test split (the test set is touched only once, at the end).
  4. Compare several models with stratified 5-fold cross-validation (train set only).
  5. Tune the strongest candidates with GridSearchCV.
  6. Pick the final model on cross-validation score, then evaluate once on the test set.
  7. Save the model, metrics, and figures.

Run:  python train.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"
for d in (MODELS_DIR, FIG_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
CLASS_NAMES = ["Benign", "Malignant"]


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def load_data():
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    # In scikit-learn: 0 = malignant, 1 = benign. Flip so malignant = 1, which is
    # the positive class we care about (recall on malignant = sensitivity).
    y = (raw.target == 0).astype(int).rename("malignant")
    return X, y


# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
def run_eda(X, y):
    df = X.copy()
    df["diagnosis"] = y.map({0: "Benign", 1: "Malignant"})

    # Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["diagnosis"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                palette={"Benign": "#4C78A8", "Malignant": "#E45756"}, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 4, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class distribution")
    ax.set_ylabel("Samples")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_class_distribution.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap="vlag", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature correlation (many 'mean', 'worst' and 'perimeter/area/radius' features are highly correlated)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_correlation_heatmap.png", dpi=130)
    plt.close(fig)

    # Distributions of the features most associated with the target
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    top = corr_with_target.index[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, feat in zip(axes.ravel(), top):
        sns.kdeplot(data=df, x=feat, hue="diagnosis", fill=True, common_norm=False,
                    palette={"Benign": "#4C78A8", "Malignant": "#E45756"}, ax=ax, legend=False)
        ax.set_title(feat, fontsize=10)
        ax.set_ylabel("")
    axes[0, 0].legend(labels=CLASS_NAMES[::-1], loc="upper right", fontsize=8)
    fig.suptitle("Top 8 features most correlated with malignancy", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_top_feature_distributions.png", dpi=130)
    plt.close(fig)

    summary = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_benign": int((y == 0).sum()),
        "n_malignant": int((y == 1).sum()),
        "missing_values": int(X.isna().sum().sum()),
        "top_features_by_abs_correlation": {k: round(float(v), 3) for k, v in corr_with_target.head(8).items()},
    }
    return summary


# ----------------------------------------------------------------------------
# 3-5. Models
# ----------------------------------------------------------------------------
def candidate_models():
    scaler = lambda: StandardScaler()  # noqa: E731
    return {
        "Logistic Regression": Pipeline([("scale", scaler()),
                                         ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
        # Calibrated wrapper gives probabilities (SVC(probability=True) is deprecated).
        "SVM (RBF)": Pipeline([("scale", scaler()),
                               ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED),
                                                              method="sigmoid", cv=5, ensemble=False))]),
        "KNN": Pipeline([("scale", scaler()), ("clf", KNeighborsClassifier())]),
        "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("clf", GradientBoostingClassifier(random_state=SEED))]),
    }


PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "SVM (RBF)": {"clf__estimator__C": [0.1, 1, 10, 100],
                  "clf__estimator__gamma": ["scale", 0.001, 0.01, 0.1]},
    "Random Forest": {"clf__max_depth": [None, 4, 8], "clf__min_samples_leaf": [1, 2, 4],
                      "clf__max_features": ["sqrt", 0.5]},
}

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def compare_models(models, X_train, y_train, cv):
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
        print(f"  {name:<22} acc={row['accuracy_mean']:.3f}  recall={row['recall_mean']:.3f}  "
              f"f1={row['f1_mean']:.3f}  auc={row['roc_auc_mean']:.3f}")
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def tune_models(models, X_train, y_train, cv):
    tuned = {}
    for name, grid in PARAM_GRIDS.items():
        gs = GridSearchCV(models[name], grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        tuned[name] = {"estimator": gs.best_estimator_, "best_params": gs.best_params_,
                       "cv_roc_auc": float(gs.best_score_)}
        print(f"  {name:<22} best CV AUC={gs.best_score_:.4f}  params={gs.best_params_}")
    return tuned


# ----------------------------------------------------------------------------
# 6. Evaluation
# ----------------------------------------------------------------------------
def evaluate(model, X_test, y_test, name):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_malignant": precision_score(y_test, pred),
        "recall_malignant": recall_score(y_test, pred),  # sensitivity
        "specificity": tn / (tn + fp),
        "f1_malignant": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "average_precision": average_precision_score(y_test, proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    metrics = {k: (round(float(v), 4) if not isinstance(v, dict) else v) for k, v in metrics.items()}

    print(f"\nTest-set report for {name}")
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=CLASS_NAMES,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {name}")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC + PR
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, lw=2, label=f"AP = {metrics['average_precision']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve", ylim=(0, 1.02))
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_roc_pr_curves.png", dpi=150)
    plt.close(fig)

    return metrics, proba


def plot_model_comparison(cv_df):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics = ["accuracy", "recall", "f1", "roc_auc"]
    x = np.arange(len(cv_df))
    w = 0.2
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, cv_df[f"{m}_mean"], w, yerr=cv_df[f"{m}_std"], capsize=2, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(cv_df["model"], rotation=15)
    ax.set_ylim(0.88, 1.01)
    ax.set_ylabel("Score (5-fold CV, train set)")
    ax.set_title("Model comparison")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_importance(model, X_test, y_test, feature_names):
    r = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=SEED,
                               scoring="roc_auc", n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False).head(12)
    err = pd.Series(r.importances_std, index=feature_names)[imp.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index[::-1], imp.values[::-1], xerr=err.values[::-1], color="#4C78A8")
    ax.set_xlabel("Drop in ROC AUC when feature is shuffled")
    ax.set_title("Permutation importance (test set, top 12)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_permutation_importance.png", dpi=150)
    plt.close(fig)
    return {k: round(float(v), 4) for k, v in imp.items()}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    X, y = load_data()
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{int(y.sum())} malignant / {int((y == 0).sum())} benign")

    print("\n[1/5] Exploratory analysis")
    eda = run_eda(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("\n[2/5] Baseline model comparison (5-fold CV on train set)")
    models = candidate_models()
    cv_df = compare_models(models, X_train, y_train, cv)
    plot_model_comparison(cv_df)

    print("\n[3/5] Hyperparameter tuning (GridSearchCV, scoring=roc_auc)")
    tuned = tune_models(models, X_train, y_train, cv)

    best_name = max(tuned, key=lambda n: tuned[n]["cv_roc_auc"])
    best_model = tuned[best_name]["estimator"]
    print(f"\nSelected model (by CV AUC): {best_name}")

    print("\n[4/5] Final evaluation on held-out test set")
    test_metrics, _ = evaluate(best_model, X_test, y_test, best_name)

    print("[5/5] Interpretation and saving artefacts")
    importance = plot_importance(best_model, X_test, y_test, X.columns)

    joblib.dump({"model": best_model, "feature_names": list(X.columns),
                 "positive_class": "malignant", "threshold": 0.5},
                MODELS_DIR / "breast_cancer_model.joblib")

    cv_df.round(4).to_csv(REPORT_DIR / "cv_model_comparison.csv", index=False)
    results = {
        "dataset": eda,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "stratified": True, "seed": SEED},
        "cv_baseline": cv_df.round(4).to_dict(orient="records"),
        "tuned": {n: {"best_params": t["best_params"], "cv_roc_auc": round(t["cv_roc_auc"], 4)}
                  for n, t in tuned.items()},
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "top_permutation_importance": importance,
    }
    with open(REPORT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone. Artefacts:")
    print(f"  model   -> {MODELS_DIR / 'breast_cancer_model.joblib'}")
    print(f"  metrics -> {REPORT_DIR / 'results.json'}")
    print(f"  figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
