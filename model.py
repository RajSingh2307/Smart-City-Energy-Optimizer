# backend/model.py
"""
Production-ready ML training & prediction pipeline for energy forecasting.

Usage (examples):
  # Train (reads config/config.yaml for settings)
  python -m backend.model train

  # Train with small random-search
  python -m backend.model train --random-search

  # Predict using saved model
  python -m backend.model predict --model models/random_forest__20250801T123000.pkl --input path/to/input.csv

Design choices:
- Uses sklearn Pipeline (StandardScaler + model) to ensure consistent preprocessing at predict time.
- Saves model + metadata (features, metrics, config snapshot) together.
- Supports time-series aware splitting (TimeSeriesSplit) or random shuffle split.
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project utils (ensure backend/utils.py exists)
from backend.utils import load_config, get_logger, ensure_dir, save_json

logger = get_logger(__name__)

# Load config
CONFIG = load_config()
PATHS = CONFIG.get("paths", {})
PROCESSED_DIR = Path(PATHS.get("processed_data_dir", "data/processed"))
MODEL_DIR = Path(PATHS.get("models_dir", "models"))
ensure_dir(MODEL_DIR)

DEFAULT_PROCESSED_FILENAME = "processed_data.csv"
TARGET_COL = "energy_consumption"


# -------------------------
# Helpers
# -------------------------
def _get_timestamp() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def _make_model_pipeline(model_name: str = "random_forest", model_params: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Build sklearn Pipeline with StandardScaler and chosen regressor.
    Supported model_name: 'ridge', 'random_forest'
    """
    model_params = model_params or {}
    if model_name == "ridge":
        reg = Ridge(**model_params)
    elif model_name == "random_forest":
        reg = RandomForestRegressor(n_jobs=-1, random_state=42, **model_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", reg)])
    return pipeline


# -------------------------
# Data loading
# -------------------------
def load_processed_data(filename: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed dataset from data/processed/ (CSV expected).
    Raises FileNotFoundError if not found.
    """
    filename = filename or DEFAULT_PROCESSED_FILENAME
    p = PROCESSED_DIR / filename
    if not p.exists():
        logger.error("Processed data not found at %s", p)
        raise FileNotFoundError(f"Processed data not found: {p}")
    logger.info("Loading processed data from %s", p)
    # probe header to check if timestamp column exists
    header = pd.read_csv(p, nrows=0)
    parse_dates = [c for c in header.columns if c == "timestamp"]
    df = pd.read_csv(p, parse_dates=parse_dates)
    return df

# -------------------------
# Train / Evaluate
# -------------------------
def train_model(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = TARGET_COL,
    model_name: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    use_timeseries_split: bool = True,
    n_splits: int = 3,
    random_search: bool = False,
    random_search_params: Optional[Dict[str, List[Any]]] = None,
    random_search_iters: int = 10,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Trains a model and returns results + saved artifact path.
    - df: processed DataFrame
    - features: list of feature column names (if None: all except target + timestamp)
    - model_name: 'random_forest' | 'ridge'
    - random_search: whether to run RandomizedSearchCV (lightweight)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe")

    # Feature selection
    if features:
        X = df[features].copy()
    else:
        drop_cols = [target, "timestamp"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    y = df[target].astype(float).values
    feature_names = list(X.columns)

    logger.info("Training model '%s' using %d features", model_name, len(feature_names))

    # Split
    if use_timeseries_split and "timestamp" in df.columns:
        # Use the last fold's train/test as final evaluation
        df_sorted = df.sort_values("timestamp")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(df_sorted))
        train_idx, test_idx = splits[-1]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        logger.info("Using TimeSeriesSplit (last split) with train=%d test=%d", len(train_idx), len(test_idx))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=not use_timeseries_split)
        logger.info("Using random train/test split: train=%d test=%d", X_train.shape[0], X_test.shape[0])

    # Build pipeline
    pipeline = _make_model_pipeline(model_name=model_name, model_params=model_params)

    # Optional random search for small tuning
    best_estimator = pipeline
    search_info = None
    if random_search:
        logger.info("Running randomized search for %d iterations", random_search_iters)
        # Map param names for pipeline
        param_dist = {}
        if model_name == "random_forest":
            param_dist = {
                "regressor__n_estimators": [100, 200, 400],
                "regressor__max_depth": [6, 8, 12, None],
                "regressor__max_features": ["sqrt", "log2", 0.8],
            }
        elif model_name == "ridge":
            param_dist = {"regressor__alpha": [0.1, 1.0, 10.0, 50.0]}

        # allow overriding via random_search_params
        if random_search_params:
            # Expecting keys in "regressor__param" format
            param_dist.update(random_search_params)

        rnd = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=random_search_iters, n_jobs=-1, cv=3, random_state=42, verbose=0)
        rnd.fit(X_train, y_train)
        best_estimator = rnd.best_estimator_
        search_info = {"best_params": rnd.best_params_, "best_score": float(rnd.best_score_)}
        logger.info("Random search complete: best_score=%s", search_info["best_score"])
    else:
        best_estimator.fit(X_train, y_train)

    # Evaluate
    preds = best_estimator.predict(X_test)
    metrics = _evaluate(y_test, preds)
    logger.info("Evaluation metrics: %s", metrics)

    # Save model artifact + metadata
    artifact_ts = _get_timestamp()
    model_filename = f"{model_name}__{artifact_ts}.pkl"
    model_path = MODEL_DIR / model_filename
    joblib.dump(best_estimator, model_path)
    logger.info("Saved model to %s", model_path)

    metadata = {
        "model_filename": str(model_path.name),
        "model_path": str(model_path),
        "model_name": model_name,
        "timestamp": artifact_ts,
        "features": feature_names,
        "metrics": metrics,
        "config_snapshot": CONFIG,
    }
    # Save metadata json
    meta_path = MODEL_DIR / f"{model_name}__{artifact_ts}__meta.json"
    save_json(metadata, meta_path)
    logger.info("Saved model metadata to %s", meta_path)

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "metrics": metrics,
        "search_info": search_info,
    }


# -------------------------
# Predict
# -------------------------
def load_model(model_path: str) -> Any:
    """Load model artifact via joblib."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found: {p}")
    logger.info("Loading model artifact %s", p)
    return joblib.load(p)


def predict(model_path: str, input_df: pd.DataFrame) -> np.ndarray:
    """
    Produce predictions using a saved pipeline artifact.
    Expects input_df to contain same feature columns used during training.
    """
    model = load_model(model_path)
    # If pipeline expects scaling, it will be applied inside pipeline
    preds = model.predict(input_df)
    return np.asarray(preds)


# -------------------------
# CLI entrypoint
# -------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train / predict energy forecasting models")
    sub = p.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train", help="Train a model")
    train_p.add_argument("--processed-file", type=str, default=DEFAULT_PROCESSED_FILENAME, help="Processed CSV filename")
    train_p.add_argument("--model-name", type=str, default="random_forest", choices=["random_forest", "ridge"])
    train_p.add_argument("--use-timeseries-split", action="store_true")
    train_p.add_argument("--random-search", action="store_true")
    train_p.add_argument("--random-search-iters", type=int, default=10)

    pred_p = sub.add_parser("predict", help="Predict using an existing model")
    pred_p.add_argument("--model-path", type=str, required=True, help="Path to saved .pkl model")
    pred_p.add_argument("--input-csv", type=str, required=True, help="CSV with input features (columns must match training features)")
    pred_p.add_argument("--out-csv", type=str, help="Path to write predictions CSV (optional)")

    return p.parse_args()


def main():
    args = _parse_args()
    if args.cmd == "train":
        df = load_processed_data(args.processed_file)
        res = train_model(
            df,
            model_name=args.model_name,
            use_timeseries_split=args.use_timeseries_split,
            random_search=args.random_search,
            random_search_iters=args.random_search_iters,
        )
        logger.info("Training finished. Metrics: %s", res["metrics"])
        print(json.dumps(res, indent=2))
    elif args.cmd == "predict":
        input_df = pd.read_csv(args.input_csv)
        preds = predict(args.model_path, input_df)
        out_df = input_df.copy()
        out_df["prediction"] = preds
        if args.out_csv:
            out_df.to_csv(args.out_csv, index=False)
            logger.info("Wrote predictions to %s", args.out_csv)
        else:
            print(out_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
