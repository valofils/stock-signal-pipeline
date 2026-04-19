"""
Modeling module.
Trains an XGBoost binary classifier using walk-forward validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / "results"

FEATURE_COLS = [
    "ticker_encoded",
    "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "obv", "daily_return", "return_5d", "return_20d",
    "volatility_20d", "close_to_sma20", "close_to_sma50",
]

TARGET_COL = "target"


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    date_col: str = "Date",
):
    """
    Generate walk-forward train/test index splits based on time.
    Each fold: train on all data up to split point, test on next window.
    """
    dates = df[date_col].sort_values().unique()
    fold_size = len(dates) // (n_splits + 1)

    for i in range(1, n_splits + 1):
        train_end = dates[fold_size * i]
        test_end  = dates[min(fold_size * (i + 1) - 1, len(dates) - 1)]

        train_idx = df[df[date_col] < train_end].index
        test_idx  = df[(df[date_col] >= train_end) & (df[date_col] <= test_end)].index

        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


def train_walk_forward(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> tuple[list[dict], XGBClassifier]:
    """
    Train XGBoost using walk-forward validation and return per-fold metrics.
    """
    df = df.sort_values("Date").reset_index(drop=True)

    # Class weight to handle imbalance
    neg = (df[TARGET_COL] == 0).sum()
    pos = (df[TARGET_COL] == 1).sum()
    scale_pos_weight = round(neg / pos, 2)
    print(f"scale_pos_weight: {scale_pos_weight} (neg={neg}, pos={pos})")

    fold_metrics = []
    print(f"\nWalk-forward validation | {n_splits} folds\n{'='*50}")

    for fold, (train_idx, test_idx) in enumerate(walk_forward_split(df, n_splits), 1):
        X_train = df.loc[train_idx, FEATURE_COLS]
        y_train = df.loc[train_idx, TARGET_COL]
        X_test  = df.loc[test_idx, FEATURE_COLS]
        y_test  = df.loc[test_idx, TARGET_COL]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "fold":       fold,
            "train_size": len(train_idx),
            "test_size":  len(test_idx),
            "roc_auc":    round(roc_auc_score(y_test, y_proba), 4),
            "precision":  round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":     round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":         round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        fold_metrics.append(metrics)

        print(
            f"Fold {fold} | Train: {len(train_idx):>5} | Test: {len(test_idx):>5} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}"
        )

    # Final model trained on ALL data
    print(f"\nTraining final model on full dataset ({len(df)} rows)...")
    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(df[FEATURE_COLS], df[TARGET_COL])

    return fold_metrics, final_model


def save_model(model: XGBClassifier, filename: str = "xgb_model.joblib") -> Path:
    """Save trained model to results directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(filename: str = "xgb_model.joblib") -> XGBClassifier:
    """Load trained model from results directory."""
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Train first.")
    return joblib.load(path)


def summarise_metrics(fold_metrics: list[dict]) -> pd.DataFrame:
    """Print and return a summary DataFrame of walk-forward metrics."""
    metrics_df = pd.DataFrame(fold_metrics)
    print(f"\n{'='*50}")
    print("Walk-forward summary:")
    print(metrics_df.to_string(index=False))
    print(f"\nMean ROC-AUC : {metrics_df['roc_auc'].mean():.4f}")
    print(f"Mean F1      : {metrics_df['f1'].mean():.4f}")
    return metrics_df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
    from stock_signal.features import add_ticker_encoding

    df = pd.read_parquet("data/processed/features.parquet")
    df = add_ticker_encoding(df)
    print(f"Loaded features | shape: {df.shape}")

    fold_metrics, final_model = train_walk_forward(df, n_splits=5)
    summarise_metrics(fold_metrics)
    save_model(final_model)
