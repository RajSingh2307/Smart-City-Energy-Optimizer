import os
import joblib
import json
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from backend.preprocessing import preprocess_pipeline
from loguru import logger

# -----------------------------
# Load configuration
# -----------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

RAW_DATA_DIR = Path(config["paths"]["raw_data_dir"])
PROCESSED_DATA_DIR = Path(config["paths"]["processed_data_dir"])
MODELS_DIR = Path(config["paths"]["models_dir"])
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Setup logging
# -----------------------------
LOG_FILE = config["logging"].get("log_file", "logs/train_model.log")
logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="14 days",
    level=config["logging"].get("level", "INFO"),
)
logger.info("🚀 Starting model training...")

# -----------------------------
# Load raw data
# -----------------------------
DATA_FILE = RAW_DATA_DIR / "energy_data.csv"
if not DATA_FILE.exists():
    logger.error(f"❌ Raw data file not found: {DATA_FILE}")
    raise FileNotFoundError(f"Raw data file not found: {DATA_FILE}")

# Run preprocessing pipeline
preprocess_pipeline()

# Read processed data
processed_file = PROCESSED_DATA_DIR / "processed_data.csv"
if not processed_file.exists():
    logger.error(f"❌ Processed data not found after preprocessing: {processed_file}")
    raise FileNotFoundError(processed_file)

df_processed = pd.read_csv(processed_file, parse_dates=["timestamp"])
logger.info(f"✅ Loaded processed data with shape: {df_processed.shape}")

# -----------------------------
# Split features and target
# -----------------------------
TARGET_COLUMN = "energy_consumption"  # update if needed
X = df_processed.drop(columns=[TARGET_COLUMN])
y = df_processed[TARGET_COLUMN]

# Drop datetime columns (not supported by XGBoost)
datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
if datetime_cols:
    logger.warning(f"⚠️ Dropping datetime columns: {datetime_cols}")
    X = X.drop(columns=datetime_cols)

# -----------------------------
# Save feature metadata (order) so API can build inputs exactly the same way
# -----------------------------
FEATURES_PATH = MODELS_DIR / "energy_model_features.json"
feature_names = list(X.columns)
with open(FEATURES_PATH, "w", encoding="utf-8") as fh:
    json.dump({"feature_names": feature_names}, fh, indent=2, ensure_ascii=False)
logger.info(f"Saved feature metadata to {FEATURES_PATH}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
logger.info(
    f"📊 Data split complete -> Train: {X_train.shape}, Test: {X_test.shape}"
)

# -----------------------------
# Train model
# -----------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

logger.info("🧠 Training XGBoost model...")
model.fit(X_train, y_train)
logger.info("✅ Model training complete.")

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

logger.info(f"📈 Evaluation -> RMSE: {rmse:.4f}, R²: {r2:.4f}")

# -----------------------------
# Save trained model
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILENAME = MODELS_DIR / f"energy_model_{timestamp}.pkl"
joblib.dump(model, MODEL_FILENAME)
logger.info(f"💾 Model saved to {MODEL_FILENAME}")

# Save latest model consistently for API
LATEST_MODEL_PATH = MODELS_DIR / "energy_model.pkl"
joblib.dump(model, LATEST_MODEL_PATH)
logger.info(f"🔄 Latest model updated at {LATEST_MODEL_PATH}")
