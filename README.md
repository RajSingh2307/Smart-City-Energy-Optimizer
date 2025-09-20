# Smart-City-Energy-Optimizer# ⚡ SmartCity Energy Optimizer — Energy & Weather Insights

An **end-to-end ML project** that forecasts energy demand for smart cities by combining **energy usage data + real-time weather API inputs**.  
The system is built with a modular architecture: **data ingestion → preprocessing → ML modeling → FastAPI REST API → Streamlit dashboard**.

---

## 🔹 Features
- 🔄 **Automated Data Pipeline** — Ingests raw energy + weather data (`data_ingestion.py`, `weather_api.py`).
- 🧹 **Data Preprocessing & Feature Engineering** — Cleans, transforms, and extracts key signals (temperature, humidity, holidays).
- 🤖 **ML Modeling & Prediction** — XGBoost/RandomForest model to forecast short-term energy demand.
- ⚡ **REST API (FastAPI)** — Exposes trained model for real-time predictions.
- 📊 **Frontend Dashboard** — Interactive Streamlit app to visualize forecasts, actuals, and weather impact.
- 🧪 **Unit Tests** — Ensures reliability (`tests/`).

---

## 🔹 Tech Stack
- **Python** (pandas, NumPy, scikit-learn, XGBoost)
- **APIs**: FastAPI (REST)
- **Frontend**: Streamlit (local UI)
- **Config & Deployment**: YAML, Docker-ready structure
- **Version Control**: Git & GitHub
- **Data Viz**: Matplotlib, Seaborn

---

## 🔹 Repository Structure
