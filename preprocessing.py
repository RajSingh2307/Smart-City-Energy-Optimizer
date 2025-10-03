# backend/preprocessing.py

"""
Preprocessing module for energy + weather datasets.
Includes data loading, cleaning, feature engineering, and saving processed data.
Industry-standard with config-driven paths, schema validation, and logging.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from loguru import logger
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional

# =========================
# Config Loader
# =========================
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()


# =========================
# Schema Validation
# =========================
class EnergyWeatherSchema(BaseModel):
    timestamp: List[pd.Timestamp] = Field(..., description="List of timestamps")
    energy_consumption: List[float] = Field(..., description="Energy consumption values")
    temperature: Optional[List[float]] = Field(None, description="Temperature values")
    humidity: Optional[List[float]] = Field(None, description="Humidity values")
    wind_speed: Optional[List[float]] = Field(None, description="Wind speed values")

    model_config = {
        "arbitrary_types_allowed": True  # <-- allows pd.Timestamp
    }

    @validator("energy_consumption", "temperature", "humidity", "wind_speed", pre=True, each_item=False)
    def series_to_list(cls, v):
        if isinstance(v, pd.Series):
            return v.tolist()
        return v

    @validator("timestamp", pre=True, each_item=False)
    def timestamp_series_to_list(cls, v):
        if isinstance(v, pd.Series):
            return v.tolist()
        return v

    @validator("energy_consumption")
    def check_lengths(cls, v, values):
        ts_len = len(values.get("timestamp", []))
        if ts_len != len(v):
            raise ValueError("Length of energy_consumption must match timestamp length")
        return v

# =========================
# Data Loading
# =========================
def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw CSV data into DataFrame and standardize column names.

    Supports datasets like AEP_hourly.csv (Datetime, AEP_MW) and also
    expects a normalized output with columns:
        timestamp, energy_consumption, temperature, humidity, wind_speed
    """
    try:
        # Try common timestamp column names (prefer 'timestamp' but fall back to 'Datetime' etc.)
        # We'll attempt to read with the most common name first, then fallback.
        # Use nrows=0 to probe columns quickly
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(path)

        # Probe header to find timestamp-like column
        preview = pd.read_csv(path, nrows=0)
        cols = [c.lower() for c in preview.columns]

        # pick parse date column candidates
        ts_candidates = [c for c in preview.columns if c.lower() in ("timestamp", "datetime", "time", "date")]
        if ts_candidates:
            parse_col = ts_candidates[0]
            df = pd.read_csv(path, parse_dates=[parse_col])
            # rename canonical timestamp to 'timestamp'
            if parse_col != "timestamp":
                df.rename(columns={parse_col: "timestamp"}, inplace=True)
        else:
            # fallback: read without parse, then try converting common column
            df = pd.read_csv(path)
            if "Datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["Datetime"])
            elif "datetime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["datetime"])
            else:
                # last resort: try index as timestamp (rare)
                logger.warning("No timestamp-like column found in %s. 'timestamp' will be created from index.", path)
                df["timestamp"] = pd.to_datetime(df.index)

        # Normalize known energy column names to 'energy_consumption'
        if "AEP_MW" in df.columns:
            df.rename(columns={"AEP_MW": "energy_consumption"}, inplace=True)
        elif "energy" in df.columns and "energy_consumption" not in df.columns:
            df.rename(columns={"energy": "energy_consumption"}, inplace=True)
        # if neither exists, but there's a single numeric column other than timestamp, try to infer
        if "energy_consumption" not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "timestamp"]
            if len(numeric_cols) == 1:
                df.rename(columns={numeric_cols[0]: "energy_consumption"}, inplace=True)
                logger.info("Inferred energy_consumption from numeric column %s", numeric_cols[0])
            else:
                logger.warning("No energy_consumption column detected. Ensure your raw file has energy values.")

        # Ensure weather columns exist (fill with NaN if missing)
        for c in ("temperature", "humidity", "wind_speed"):
            if c not in df.columns:
                df[c] = np.nan

        # Final logging and return
        logger.info("Loaded raw data from %s with shape %s", path, df.shape)
        # ensure timestamp is datetime dtype
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    except Exception as e:
        logger.error("Error loading raw data: %s", e)
        raise



# =========================
# Data Cleaning
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data by handling duplicates, missing values, and invalid entries."""
    logger.info("Cleaning data...")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {before - len(df)} duplicate rows")

    # Handle missing values
    df = df.ffill().bfill()

    # Ensure numeric types
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Remove unrealistic values
    if "energy_consumption" in df.columns:
        df = df[df["energy_consumption"] >= 0]  # no negative consumption

    logger.info("Data cleaning complete")
    return df


# =========================
# Feature Engineering
# =========================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based, lagged, and interaction features."""
    logger.info("Performing feature engineering...")

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Rolling features
    df["energy_rolling_mean_3h"] = (
        df["energy_consumption"].rolling(window=3, min_periods=1).mean()
    )
    df["temp_rolling_mean_3h"] = (
        df["temperature"].rolling(window=3, min_periods=1).mean()
    )

    # Lag features
    df["energy_lag_1h"] = df["energy_consumption"].shift(1)
    df["energy_lag_6h"] = df["energy_consumption"].shift(6)

    # Interaction feature
    df["temp_energy_interaction"] = df["temperature"] * df["energy_consumption"]

    logger.info("Feature engineering complete")
    return df


# =========================
# Save Processed Data
# =========================
def save_processed_data(df: pd.DataFrame, file_path: Path) -> None:
    """Save processed DataFrame to CSV."""
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


# =========================
# Pipeline
# =========================
def preprocess_pipeline():
    """Full preprocessing pipeline: load, clean, feature engineer, and save."""
    raw_path = Path(config["data"]["raw"]) / "energy_data.csv"
    processed_path = Path(config["data"]["processed"]) / "processed_data.csv"

    df = load_raw_data(raw_path)
    
    try:
        EnergyWeatherSchema(
            timestamp=df["timestamp"],
            energy_consumption=df["energy_consumption"],
            temperature=df["temperature"],
            humidity=df.get("humidity"),
            wind_speed=df.get("wind_speed"),
        )
    except ValidationError as e:
        logger.error(f"Schema validation failed: {e}")
        raise

    df = clean_data(df)
    df = feature_engineering(df)
    save_processed_data(df, processed_path)


if __name__ == "__main__":
    preprocess_pipeline()
