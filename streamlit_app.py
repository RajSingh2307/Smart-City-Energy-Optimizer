# frontend/streamlit_app.py
"""
Streamlit UI for SmartCity Energy Optimizer
Polished, integrated, and robust — ready to paste into frontend/streamlit_app.py
"""

from pathlib import Path
import json
import joblib
import requests
import pandas as pd
import numpy as np
import streamlit as st
from loguru import logger
from datetime import datetime

# -------------------------
# Configuration / paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LOCAL_MODEL_PATH = MODEL_DIR / "energy_model.pkl"
FEATURES_JSON = MODEL_DIR / "energy_model_features.json"

API_BASE = "http://127.0.0.1:8000"  # adjust if your API is different

st.set_page_config(
    page_title="SmartCity Energy Insights",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helpers
# -------------------------
@st.cache_data(show_spinner=False)
def list_model_artifacts():
    if not MODEL_DIR.exists():
        return []
    files = sorted(MODEL_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.name) for p in files if p.suffix in (".pkl", ".json")]

@st.cache_data(ttl=60, show_spinner=False)
def load_feature_metadata():
    # Try explicit JSON
    if FEATURES_JSON.exists():
        try:
            data = json.loads(FEATURES_JSON.read_text(encoding="utf-8"))
            # support multiple key names
            fnames = data.get("feature_names") or data.get("features") or data.get("feature_list")
            if isinstance(fnames, list) and fnames:
                return fnames, data
        except Exception:
            pass

    # fallback: look for any meta json in models folder
    for p in MODEL_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            # look for 'features' or 'feature_names'
            fn = d.get("features") or d.get("feature_names") or d.get("feature_list")
            if isinstance(fn, list) and fn:
                return fn, d
        except Exception:
            continue

    # last resort: try to infer from processed data
    proc = PROCESSED_DIR / "processed_data.csv"
    if proc.exists():
        df = pd.read_csv(proc, nrows=5)
        candidates = [c for c in df.columns if c not in ("timestamp", "energy_consumption")]
        return candidates, {"inferred_from": str(proc)}
    return [], {}

@st.cache_resource(show_spinner=False)
def load_local_model(path=LOCAL_MODEL_PATH):
    if path.exists():
        try:
            mdl = joblib.load(path)
            return mdl
        except Exception:
            return None
    return None

def call_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.ok:
            return True, r.json()
    except Exception:
        pass
    return False, None

def call_api_predict(payload: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
        if r.ok:
            return True, r.json()
        else:
            return False, {"status": r.status_code, "text": r.text}
    except Exception as e:
        return False, {"error": str(e)}

def predict_local_model(model, features_ordered: list):
    """model can be a scikit pipeline or raw model. features_ordered is a list of numeric values"""
    if model is None:
        raise RuntimeError("Local model not loaded")
    # If pipeline: call predict directly
    try:
        pred = model.predict([features_ordered])
        return float(pred[0])
    except Exception as e:
        raise

def get_feature_defaults(feature_names, sample_stats=None):
    """Return a dict name->(min, max, default) for UI sliders."""
    defaults = {}
    for f in feature_names:
        fn = f.lower()
        if "hour" in fn or "hour_of_day" in fn:
            defaults[f] = (0, 23, 12)
        elif "day" in fn and "week" in fn:
            defaults[f] = (0, 6, 0)
        elif "humidity" in fn:
            defaults[f] = (0, 100, 50)
        elif "temp" in fn or "temperature" in fn:
            defaults[f] = (-20, 50, 20)
        elif "is_weekend" in fn:
            defaults[f] = (0, 1, 0)
        else:
            # numeric fallback - use sample_stats if available
            if sample_stats is not None and f in sample_stats:
                mn, mx, med = sample_stats[f]
                defaults[f] = (float(mn), float(mx), float(med))
            else:
                defaults[f] = (0.0, 1.0, 0.0)
    return defaults

def compute_sample_stats(df, feature_names):
    stats = {}
    for f in feature_names:
        if f in df.columns:
            col = pd.to_numeric(df[f], errors="coerce")
            stats[f] = (np.nanmin(col.fillna(0)), np.nanmax(col.fillna(0)), float(np.nanmedian(col.fillna(0))))
    return stats

# -------------------------
# Load artifacts / metadata
# -------------------------
available_artifacts = list_model_artifacts()
FEATURE_NAMES, FEATURE_META = load_feature_metadata()
local_model = load_local_model()

# -------------------------
# App UI
# -------------------------
st.title("⚡ SmartCity Energy Insights")
st.caption("Interactive demo — Forecast energy consumption. Backend-integrated, production-friendly UI.")

left_col, right_col = st.columns([3, 1])

with right_col:
    st.markdown("### Model / Environment")
    st.write("**Local model:**", LOCAL_MODEL_PATH.name if LOCAL_MODEL_PATH.exists() else "None")
    st.write("**Model artifacts:**")
    st.write(available_artifacts[:10] or "No artifacts found")
    api_ok, api_info = call_api_health()
    if api_ok:
        st.success("API health: OK")
    else:
        st.warning("API health: unavailable (UI will try local model)")

with left_col:
    st.header("Quick predict (single row)")
    if not FEATURE_NAMES:
        st.error("No feature metadata found. Run training (model metadata) or place a features JSON at data/models/energy_model_features.json")
    else:
        # load processed_data stats (if available) to give nice slider ranges
        stats = {}
        proc_file = PROCESSED_DIR / "processed_data.csv"
        if proc_file.exists():
            try:
                df_proc = pd.read_csv(proc_file, nrows=10000, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(proc_file, nrows=0).columns else None)
                stats = compute_sample_stats(df_proc, FEATURE_NAMES)
            except Exception:
                stats = {}

        defaults = get_feature_defaults(FEATURE_NAMES, sample_stats=stats)

        with st.form("predict_form"):
            st.subheader("Input features")
            input_values = {}
            # layout features in 2 columns
            cols = st.columns(2)
            for i, fname in enumerate(FEATURE_NAMES):
                col = cols[i % 2]
                mn, mx, default = defaults.get(fname, (0.0, 1.0, 0.0))
                # choose numeric input type based on ranges
                if isinstance(mn, (int, np.integer)) and isinstance(mx, (int, np.integer)) and (mx - mn) <= 200:
                    val = col.slider(fname, min_value=float(mn), max_value=float(mx), value=float(default))
                else:
                    # numeric input
                    val = col.number_input(fname, value=float(default))
                input_values[fname] = float(val)
            submitted = st.form_submit_button("Predict")

        if submitted:
            st.info("Running prediction...")
            # Build payload mapping feature_name -> value (API-friendly)
            payload = {k: float(v) for k, v in input_values.items()}
            # Try API first
            if api_ok:
                ok, resp = call_api_predict(payload)
                if ok:
                    pred = resp.get("prediction_kwh") or resp.get("prediction")
                    st.success(f"Prediction (API): {pred:.4f} kWh")
                    if "used_features" in resp:
                        st.caption(f"Used features: {resp['used_features']}")
                else:
                    st.error(f"API prediction failed: {resp}")
                    # fallback to local
                    if local_model is not None:
                        try:
                            ordered = [payload.get(fn, 0.0) for fn in FEATURE_NAMES]
                            pred_local = predict_local_model(local_model, ordered)
                            st.success(f"Prediction (local): {pred_local:.4f} kWh")
                        except Exception as e:
                            st.exception(e)
                    else:
                        st.warning("Local model not available for fallback.")
            else:
                # use local model if available
                if local_model is not None:
                    try:
                        ordered = [payload.get(fn, 0.0) for fn in FEATURE_NAMES]
                        pred_local = predict_local_model(local_model, ordered)
                        st.success(f"Prediction (local): {pred_local:.4f} kWh")
                    except Exception as e:
                        st.exception(e)
                else:
                    st.error("No API and no local model available. Start your API (uvicorn) or place model at data/models/energy_model.pkl")

# -------------------------
# Model metadata / explainability
# -------------------------
st.write("---")
meta_col, viz_col = st.columns([2, 3])

with meta_col:
    st.subheader("Model metadata")
    # try to read explicit metadata json if present
    meta_files = list(MODEL_DIR.glob("*meta*.json"))
    if meta_files:
        try:
            meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
            st.write("**Model file:**", meta.get("model_filename") or "unknown")
            st.write("**Timestamp:**", meta.get("timestamp"))
            st.write("**Model name:**", meta.get("model_name"))
            st.write("**Metrics:**")
            st.json(meta.get("metrics", {}))
        except Exception:
            st.write("Could not parse model metadata file.")
    else:
        # if no meta file, display what we have
        st.write("No metadata file found. Showing inferred metadata:")
        st.write("FEATURES:", FEATURE_NAMES or "None")
        if local_model is not None:
            try:
                # attempt to get feature importances
                fi = None
                if hasattr(local_model, "feature_importances_"):
                    fi = local_model.feature_importances_
                elif hasattr(local_model, "named_steps"):
                    # pipeline
                    reg = local_model.named_steps.get("regressor")
                    if hasattr(reg, "feature_importances_"):
                        fi = reg.feature_importances_
                if fi is not None:
                    st.write("Feature importances (top 10):")
                    imp = sorted(zip(FEATURE_NAMES, fi), key=lambda x: x[1], reverse=True)[:10]
                    st.table(pd.DataFrame(imp, columns=["feature", "importance"]))
                else:
                    st.info("No feature_importances_ available in model artifact (could be a pipeline or linear model).")
            except Exception:
                st.info("Could not extract feature importances from model.")

with viz_col:
    st.subheader("Quick visualizations")
    if PROCESSED_DIR.exists() and (PROCESSED_DIR / "processed_data.csv").exists():
        try:
            df_viz = pd.read_csv(PROCESSED_DIR / "processed_data.csv", parse_dates=["timestamp"] if "timestamp" in pd.read_csv(PROCESSED_DIR / "processed_data.csv", nrows=0).columns else None)
            if "timestamp" in df_viz.columns and "energy_consumption" in df_viz.columns:
                recent = df_viz.tail(500).set_index("timestamp")
                st.line_chart(recent["energy_consumption"])
            else:
                st.write("Provide processed data with 'timestamp' & 'energy_consumption' to see time series.")
        except Exception as e:
            st.warning(f"Could not render visualization: {e}")
    else:
        st.info("No processed dataset found (data/processed/processed_data.csv). Run preprocessing first.")

# -------------------------
# Batch predictions
# -------------------------
st.write("---")
st.header("Batch predictions")
st.write("Upload a CSV with columns matching training features (same names). The app will predict and let you download results.")

uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded is not None:
    try:
        df_batch = pd.read_csv(uploaded)
        st.write("Preview of uploaded file:")
        st.dataframe(df_batch.head())
        missing = [c for c in FEATURE_NAMES if c not in df_batch.columns]
        if missing:
            st.warning(f"Uploaded file is missing features: {missing}. Missing columns will be filled with 0.0")
            for m in missing:
                df_batch[m] = 0.0
        # ensure order
        X_batch = df_batch[FEATURE_NAMES].astype(float)
        # Try API batch endpoint (not implemented in base FastAPI), so use local model
        if local_model is not None:
            preds = local_model.predict(X_batch.values) if hasattr(local_model, "predict") else None
            if preds is not None:
                out = df_batch.copy()
                out["prediction_kwh"] = preds
                st.write("Predictions preview:")
                st.dataframe(out.head())
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
            else:
                st.error("Local model does not support predict on this input.")
        else:
            st.error("Local model not available for batch prediction. Start API with a batch endpoint or put model into data/models/")
    except Exception as e:
        st.exception(e)

# -------------------------
# Debug / developer tools
# -------------------------
with st.expander("Developer / debug"):
    st.write("Artifacts in models folder:")
    st.write(available_artifacts)
    st.write("Loaded feature metadata (first 50):")
    st.write(FEATURE_NAMES[:50])
    st.write("Local model loaded:", bool(local_model))
    st.write("API base:", API_BASE)
    st.write("Config path:", CONFIG_PATH)

st.info("Tip: For best match to training behavior, serve the same preprocessing and model artifact used at training time. You can add more model explainability (SHAP/Lime) if you want deeper insights.")
