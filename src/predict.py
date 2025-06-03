# src/predict.py

import pandas as pd
import joblib
import argparse


def load_model(model_path: str):
    """
    Load a joblib‐saved XGBClassifier.
    """
    return joblib.load(model_path)


def run_inference(model, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df_features with the same columns as training except 'label':
      ['date','pos_avg','neu_avg','neg_avg','Open','High','Low','Close','Volume','Return','RSI_14','EMA_10','SMA_20','ATR_14','Volume_lag1'].
    Output: df_features plus columns ['pred_prob','pred_label'] (0 or 1).
    """
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
    X = df_features[feature_cols].values
    prob = model.predict_proba(X)[:, 1]  # probability of class = 1 (Up)
    pred = (prob >= 0.5).astype(int)
    out = df_features.copy().reset_index(drop=True)
    out["pred_prob"] = prob
    out["pred_label"] = pred
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new DJIA feature data")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/xgb_djia_sentiment_model.joblib",
        help="Path to saved XGB model"
    )
    parser.add_argument(
        "--features_csv",
        type=str,
        required=True,
        help="Path to CSV with feature columns (same as training, except no 'label')."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Where to save output with predictions"
    )
    args = parser.parse_args()

    df_feat = pd.read_csv(args.features_csv, parse_dates=["date"])
    model = load_model(args.model_path)
    df_out = run_inference(model, df_feat)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Predictions written to {args.output_csv}")