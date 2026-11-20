"""
Use the trained model to classify tumours.

Usage:
  python predict.py                      # demo on 5 random samples from the dataset
  python predict.py --csv new_data.csv   # CSV must contain the 30 WDBC feature columns

Output: predicted label and probability of malignancy for each row.
FOR EDUCATIONAL USE ONLY. Not a medical device.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

MODEL_PATH = Path(__file__).resolve().parent / "models" / "breast_cancer_model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Breast cancer prediction")
    parser.add_argument("--csv", type=str, help="CSV file with the 30 WDBC feature columns")
    args = parser.parse_args()

    bundle = joblib.load(MODEL_PATH)
    model, features, threshold = bundle["model"], bundle["feature_names"], bundle["threshold"]

    truth = None
    if args.csv:
        X = pd.read_csv(args.csv)
        missing = [c for c in features if c not in X.columns]
        if missing:
            raise SystemExit(f"Missing columns: {missing}")
        X = X[features]
    else:
        raw = load_breast_cancer(as_frame=True)
        X = raw.data.sample(5, random_state=7)
        truth = (raw.target.loc[X.index] == 0).map({True: "Malignant", False: "Benign"})

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({
        "p_malignant": proba.round(3),
        "prediction": ["Malignant" if p >= threshold else "Benign" for p in proba],
    }, index=X.index)
    if truth is not None:
        out["actual"] = truth
    print(out.to_string())


if __name__ == "__main__":
    main()
