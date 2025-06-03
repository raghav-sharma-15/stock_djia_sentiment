# src/train_model.py

import os

# Force single‐threading for OpenMP/MKL to avoid macOS segmentation faults:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_inter_op_parallelism=1"

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_xgb_on_djia(df: pd.DataFrame, save_path: str):
    """
    1) Input: df with columns [
          'date','label','pos_avg','neu_avg','neg_avg',
          'Open','High','Low','Close','Volume','Return',
          'RSI_14','EMA_10','SMA_20','ATR_14','Volume_lag1'
       ]
    2) Uses ['pos_avg','neu_avg','neg_avg','Return','RSI_14','EMA_10','SMA_20','ATR_14','Volume_lag1']
       as features to predict 'label' (0 or 1).
    3) Performs a TimeSeriesSplit (n_splits=5) with n_jobs=1 to avoid segmentation faults.
    4) Retrains on full data with n_jobs=1 and saves the final XGBClassifier via joblib to `save_path`.
    """
    df_sorted = df.sort_values("date").reset_index(drop=True).copy()

    feature_cols = [
        "pos_avg",
        "neu_avg",
        "neg_avg",
        "Return",
        "RSI_14",
        "EMA_10",
        "SMA_20",
        "ATR_14",
        "Volume_lag1",
    ]
    X = df_sorted[feature_cols].values
    y = df_sorted["label"].values  # 0 or 1

    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1
    print("\n=== TimeSeriesSplit Cross-Validation ===")
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            n_estimators=50,
            max_depth=4,
            n_jobs=1,            # force single‐threaded for XGBoost
            tree_method="hist",  # safer on macOS
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n--- Fold {fold} Accuracy: {acc:.4f} ---")
        print(classification_report(y_test, y_pred, target_names=["Down(0)", "Up(1)"]))
        fold += 1

    # Retrain on full dataset (also single‐threaded)
    final_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=100,
        max_depth=4,
        n_jobs=1,            # force single‐threaded for XGBoost
        tree_method="hist",
        random_state=42
    )
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(final_model, save_path)
    print(f"\nFinal model saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train XGBoost classifier on DJIA sentiment+technical features"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to merged CSV with columns "
             "['date','label','pos_avg','neu_avg','neg_avg','Open','High','Low','Close','Volume','Return','RSI_14','EMA_10','SMA_20','ATR_14','Volume_lag1']"
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="models/xgb_djia_sentiment_model.joblib",
        help="Where to save the trained model"
    )
    args = parser.parse_args()

    df_input = pd.read_csv(args.input_csv, parse_dates=["date"])
    train_xgb_on_djia(df_input, args.model_out)