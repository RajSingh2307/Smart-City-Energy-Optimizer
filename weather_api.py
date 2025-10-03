# backend/weather_api.py
"""
Production-ready OpenWeather (or synthetic) wrapper.

Features:
- Resilient HTTP calls with session reuse and exponential backoff.
- Local cache (parquet) with configurable TTL to avoid hitting API limits.
- Structured, testable interface that returns a pandas.DataFrame (hourly forecast rows).
- Accepts either city name or lat/lon. Falls back to synthetic generator when API key missing.
- CLI entrypoint for quick local runs.

Drop this file into: energy-weather-insights/backend/weather_api.py
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path
import requests
import pandas as pd
import yaml
import json

# ---- Module-level config ----
# ---- Module-level config ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = PROJECT_ROOT / "config" / "config.yaml"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Logger (replaceable by centralized logger later)
logger = logging.getLogger("weather_api")
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(name)s — %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


# ---- Helpers: load config ----
def _load_config() -> Dict[str, Any]:
    if CONFIG_YAML.exists():
        try:
            raw = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8")) or {}
            logger.debug("Loaded config from %s", CONFIG_YAML)
            return raw
        except Exception as e:
            logger.warning("Failed to read %s: %s", CONFIG_YAML, e)
    return {}


# ---- Simple exponential-backoff request helper ----
def _get_with_backoff(session: requests.Session, url: str, params: Dict[str, Any], max_retries: int = 4, backoff_base: float = 1.0, timeout: int = 10) -> Dict[str, Any]:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("HTTP GET attempt %d url=%s params=%s", attempt, url, {k: v for k, v in params.items() if k != "appid"})
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_exc = e
            sleep = backoff_base * (2 ** (attempt - 1))
            logger.warning("Request failed (attempt %d/%d): %s — sleeping %.1fs", attempt, max_retries, e, sleep)
            time.sleep(sleep)
    logger.error("All HTTP attempts failed for url=%s", url)
    raise last_exc


# ---- Cache helpers ----
def _cache_path_for_key(key: str) -> Path:
    safe = key.replace(" ", "_").replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"weather_cache__{safe}.parquet"


def _is_cache_fresh(p: Path, ttl_seconds: int) -> bool:
    if not p.exists():
        return False
    age = time.time() - p.stat().st_mtime
    return age <= ttl_seconds


# ---- Core public functions ----
def fetch_hourly_forecast(
    city: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    use_cache: bool = True,
    cache_ttl_seconds: int = 60 * 60,  # 1 hour default
    force_refresh: bool = False,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Fetch hourly (or 3-hour) forecast as a pandas.DataFrame with columns:
      ts (pd.Timestamp), temp_c, humidity, wind_mps, clouds_pct, source, raw (json string)

    Priority: city -> lat/lon. If no API key found in config, returns synthetic timeseries (48h).

    Args:
        city: city name like "New York"
        lat, lon: coordinates (takes precedence if provided)
        use_cache: whether to use local cache
        cache_ttl_seconds: TTL for cache freshness
        force_refresh: ignore cache and fetch fresh
        max_retries: HTTP retries

    Returns:
        pd.DataFrame
    """
    cfg = _load_config()
    api_key = None
    # config.yml expected structure (optional):
    # weather:
    #   openweather_api_key: your_key
    #   default_lat: 40.7128
    #   default_lon: -74.0060
    if isinstance(cfg, dict):
        weather_cfg = cfg.get("weather", {})
        api_key = weather_cfg.get("openweather_api_key") or weather_cfg.get("api_key") or None

    # build cache key
    key = city or f"{lat}_{lon}" or "default_city"
    cache_file = _cache_path_for_key(key)

    if use_cache and not force_refresh and _is_cache_fresh(cache_file, cache_ttl_seconds):
        try:
            df = pd.read_parquet(cache_file)
            logger.info("Loaded weather from cache: %s (rows=%d)", cache_file, len(df))
            return df
        except Exception as e:
            logger.warning("Failed to load cache %s: %s — will re-fetch", cache_file, e)

    # If no API key -> synthetic generator
    if not api_key:
        logger.warning("No OpenWeather API key found in config; returning synthetic weather (for dev).")
        return _generate_synthetic_weather(hours=48)

    # determine coordinates
    if lat is None or lon is None:
        # if city provided, use geocoding via OpenWeather (not separate call) or fallback coordinates
        # For simplicity, attempt city-based call using q parameter
        use_coords = False
    else:
        use_coords = True

    session = requests.Session()
    base_url = "https://api.openweathermap.org/data/2.5/forecast"  # 5 day / 3 hour forecast

    params: Dict[str, Any] = {"appid": api_key, "units": "metric"}
    if use_coords:
        params.update({"lat": lat, "lon": lon})
    else:
        if not city:
            # fallback to default from config or NYC coords
            city = cfg.get("weather", {}).get("default_city", "New York")
            logger.info("No city provided; falling back to %s", city)
        params.update({"q": city})

    raw = _get_with_backoff(session, base_url, params, max_retries=max_retries)
    # raw["list"] is a list of 3-hour forecasts; convert to DataFrame with hourly interpolation
    df = _openweather_raw_to_dataframe(raw)
    try:
        # attempt to write to cache
        df.to_parquet(cache_file, index=False)
        logger.info("Wrote weather forecast to cache: %s", cache_file)
    except Exception as e:
        logger.warning("Could not write weather cache: %s", e)

    return df


# ---- Convert OpenWeather JSON to tidy DataFrame ----
def _openweather_raw_to_dataframe(raw_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert response JSON (5-day forecast) to DataFrame.
    Expects structure: { "list": [ { "dt": 12345, "main": {...}, "wind": {...}, ... }, ... ] }
    """
    entries: List[Dict[str, Any]] = raw_json.get("list", [])
    rows = []
    for item in entries:
        dt = item.get("dt")
        ts = pd.to_datetime(dt, unit="s", utc=True) if dt else None
        main = item.get("main", {})
        wind = item.get("wind", {})
        clouds = item.get("clouds", {})
        rows.append(
            {
                "ts": ts,
                "temp_c": main.get("temp"),
                "temp_min_c": main.get("temp_min"),
                "temp_max_c": main.get("temp_max"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_mps": wind.get("speed"),
                "wind_deg": wind.get("deg"),
                "clouds_pct": clouds.get("all"),
                "raw": json.dumps(item),
                "source": "openweather_forecast",
            }
        )
    df = pd.DataFrame(rows)
    # Expand to hourly by interpolation if desired (simple approach: set index ts and resample hourly)
    if not df.empty and df["ts"].notna().all():
        df = df.set_index("ts").sort_index()
        # resample hourly and interpolate numeric columns
        numeric_cols = ["temp_c", "temp_min_c", "temp_max_c", "humidity", "pressure", "wind_mps", "wind_deg", "clouds_pct"]
        df = df[["temp_c", "temp_min_c", "temp_max_c", "humidity", "pressure", "wind_mps", "wind_deg", "clouds_pct", "raw", "source"]]
        df = df.resample("1H").asfreq()
        # forward/backward fill raw/source with nearest (string) values
        df["raw"] = df["raw"].ffill().bfill()
        df["source"] = df["source"].ffill().bfill()
        # interpolate numeric
        df[numeric_cols] = df[numeric_cols].interpolate(method="time").ffill().bfill()
        df = df.reset_index().rename(columns={"index": "ts"})
    else:
        # Keep as-is but ensure ts column exists
        if "ts" not in df.columns:
            df["ts"] = pd.NaT
    logger.info("Converted raw forecast to DataFrame rows=%d", len(df))
    return df


# ---- Synthetic generator for dev without API key ----
def _generate_synthetic_weather(hours: int = 48, base_temp: float = 18.0) -> pd.DataFrame:
    """
    Create deterministic synthetic hourly weather for development/testing.
    """
    now = pd.Timestamp.utcnow().floor("H")
    times = pd.date_range(now, periods=hours, freq="H", tz="UTC")
    import numpy as np

    x = np.linspace(0, 2 * np.pi, len(times))
    temps = base_temp + 6.0 * np.sin(x)  # simple daily sinusoid
    humid = 55 + 10 * np.cos(x / 2)
    wind = 2.0 + 1.0 * np.sin(x * 0.5)
    clouds = 20 + 40 * (np.sin(x * 0.7) + 1) / 2
    rows = []
    for ts, t, h, w, c in zip(times, temps, humid, wind, clouds):
        rows.append(
            {
                "ts": ts,
                "temp_c": float(round(t, 2)),
                "temp_min_c": float(round(t - 1.0, 2)),
                "temp_max_c": float(round(t + 1.0, 2)),
                "humidity": float(round(h, 2)),
                "pressure": 1012.0,
                "wind_mps": float(round(w, 2)),
                "wind_deg": 180,
                "clouds_pct": float(round(c, 2)),
                "raw": "{}",
                "source": "synthetic",
            }
        )
    df = pd.DataFrame(rows)
    logger.info("Generated synthetic weather rows=%d", len(df))
    return df


# ---- CLI convenience ----
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch hourly weather forecast (OpenWeather) and cache locally")
    p.add_argument("--city", type=str, help="City name (e.g., 'New York')")
    p.add_argument("--lat", type=float, help="Latitude")
    p.add_argument("--lon", type=float, help="Longitude")
    p.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable cache (force fetch)")
    p.add_argument("--ttl", type=int, default=3600, help="Cache TTL seconds (default 3600)")
    p.add_argument("--out", type=str, help="Optional path to save output parquet")
    args = p.parse_args()

    try:
        df = fetch_hourly_forecast(
            city=args.city,
            lat=args.lat,
            lon=args.lon,
            use_cache=args.use_cache,
            cache_ttl_seconds=args.ttl,
            force_refresh=not args.use_cache,
        )
        if args.out:
            outp = Path(args.out)
            df.to_parquet(outp, index=False)
            logger.info("Wrote forecast to %s", outp)
        else:
            print(df.head().to_string(index=False))
    except Exception as exc:
        logger.exception("Failed to fetch weather: %s", exc)
        raise
