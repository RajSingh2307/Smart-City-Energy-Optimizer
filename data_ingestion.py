# backend/data_ingestion.py
"""
Production-ready data ingestion utilities.

Responsibilities:
- Download CSV/JSON from URL with retry/backoff and streaming (memory-efficient).
- Read local CSV into a validated pandas.DataFrame (parse dates, coerce types).
- Save raw file to data/raw/ with timestamped filenames for traceability.
- Basic validation for required columns (optional).
- Clear exceptions & logging to help debugging during runs and CI.

Usage (example):
> python -m backend.data_ingestion --url "https://example.com/data.csv" --required "ts,kwh"
> python -m backend.data_ingestion --path data/raw/my_local.csv --required "ts,kwh"

Drop this file into: energy-weather-insights/backend/data_ingestion.py
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import csv
import io

import requests
import pandas as pd
import yaml

# --------- Configuration ----------
# Resolve project root relative to this file (safe when executing from any CWD)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_RAW = PROJECT_ROOT / "data" / "raw"
CONFIG_YAML = PROJECT_ROOT / "config" / "config.yaml"

# Setup module logger (if the project provides a centralized logger later,
# you can replace this with importing that logger).
logger = logging.getLogger("energy_ingest")
if not logger.handlers:
    # Basic logging config (keeps it self-contained)
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(name)s — %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# --------- Helpers ----------
def _load_config() -> dict:
    """Load optional config/config.yaml if present. Non-fatal."""
    if CONFIG_YAML.exists():
        try:
            with open(CONFIG_YAML, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                logger.debug("Loaded config from %s", CONFIG_YAML)
                return cfg
        except Exception as e:
            logger.warning("Failed to load %s: %s", CONFIG_YAML, e)
    return {}


def _ensure_raw_dir(raw_dir: Optional[Union[str, Path]] = None) -> Path:
    raw = Path(raw_dir) if raw_dir else DEFAULT_DATA_RAW
    raw.mkdir(parents=True, exist_ok=True)
    return raw


# --------- Network download with simple retry/backoff ----------
def download_file_stream(
    url: str,
    dest_path: Path,
    max_retries: int = 4,
    backoff_seconds: float = 1.0,
    timeout: int = 30,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Download a file with streaming and basic retry/backoff.
    Writes to dest_path (overwrites if exists).
    Raises requests.HTTPError or last Exception on repeated failure.
    """
    attempt = 0
    last_exc = None
    logger.info("Downloading %s -> %s", url, dest_path)
    while attempt < max_retries:
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # Create parent dir
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Stream-write
                with open(dest_path, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                logger.info("Download successful: %s", dest_path)
                return dest_path
        except Exception as e:
            last_exc = e
            attempt += 1
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            logger.warning("Download attempt %d failed: %s — retrying in %.1f s", attempt, e, sleep_for)
            time.sleep(sleep_for)
    logger.error("All download attempts failed for %s", url)
    raise last_exc


# --------- CSV/JSON reading helpers ----------
def _read_csv_with_dateparse(path: Union[str, Path], parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read CSV robustly. Auto-detect encoding fallbacks, handle large files.
    """
    path = Path(path)
    enc_candidates = ["utf-8", "utf-8-sig", "latin1"]
    last_exc = None
    for enc in enc_candidates:
        try:
            df = pd.read_csv(path, parse_dates=parse_dates, encoding=enc)
            logger.info("Loaded CSV %s with encoding=%s shape=%s", path, enc, df.shape)
            return df
        except Exception as e:
            last_exc = e
            logger.debug("Failed reading %s with encoding %s: %s", path, enc, e)
    logger.exception("Could not read CSV %s (tried encodings). Last err: %s", path, last_exc)
    raise last_exc


def _validate_columns(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame contains required columns.
    Returns (is_valid, missing_columns)
    """
    if not required_columns:
        return True, []
    missing = [c for c in required_columns if c not in df.columns]
    return (len(missing) == 0), missing


# --------- Public API ----------
def ingest_from_url(
    url: str,
    raw_dir: Optional[Union[str, Path]] = None,
    filename_hint: Optional[str] = None,
    parse_dates: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Download a dataset (CSV/JSON) from `url` to data/raw/ and return a pandas DataFrame.

    - filename_hint: used to form the saved filename (kept safe).
    - parse_dates: list of columns to parse as datetimes in pd.read_csv.
    - required_columns: list of columns that *must* exist in the resulting DataFrame (validation).
    - max_retries: download retry attempts.

    Raises:
        FileNotFoundError, ValueError, requests.HTTPError, Exception
    """
    cfg = _load_config()
    raw_dir = _ensure_raw_dir(raw_dir)
    # Build a safe filename with timestamp to ensure traceability
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_hint = Path(filename_hint or Path(url).name).name
    dest_name = f"{ts}__{safe_hint}"
    dest_path = raw_dir / dest_name

    # Download
    download_file_stream(url, dest_path, max_retries=max_retries)

    # If it's a CSV-like file - attempt to read
    # Basic detection by extension
    suffix = dest_path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        df = _read_csv_with_dateparse(dest_path, parse_dates=parse_dates)
    elif suffix in [".json"]:
        df = pd.read_json(dest_path)
        logger.info("Loaded JSON %s shape=%s", dest_path, df.shape)
    else:
        # Try reading as CSV as a fallback
        try:
            df = _read_csv_with_dateparse(dest_path, parse_dates=parse_dates)
        except Exception as e:
            logger.error("Unsupported file type and could not fallback-read: %s", dest_path)
            raise

    valid, missing = _validate_columns(df, required_columns)
    if not valid:
        # keep raw file but raise an informative error for pipeline logic
        logger.error("Validation failed. Missing columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing} in downloaded file {dest_path}")

    return df


def ingest_from_local(
    local_path: Union[str, Path],
    raw_dir: Optional[Union[str, Path]] = None,
    parse_dates: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read a local CSV/JSON file, validate, and copy it to the raw data folder (timestamped).
    Returns the DataFrame.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        logger.error("Local file not found: %s", local_path)
        raise FileNotFoundError(local_path)

    raw_dir = _ensure_raw_dir(raw_dir)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest_path = raw_dir / f"{ts}__{local_path.name}"

    # Copy file to raw_dir (preserve original)
    with open(local_path, "rb") as src, open(dest_path, "wb") as dst:
        dst.write(src.read())
    logger.info("Copied local file %s -> %s", local_path, dest_path)

    # Read
    suffix = local_path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        df = _read_csv_with_dateparse(dest_path, parse_dates=parse_dates)
    elif suffix in [".json"]:
        df = pd.read_json(dest_path)
        logger.info("Loaded JSON %s shape=%s", dest_path, df.shape)
    else:
        # attempt csv fallback
        df = _read_csv_with_dateparse(dest_path, parse_dates=parse_dates)

    valid, missing = _validate_columns(df, required_columns)
    if not valid:
        logger.error("Validation failed on copied local file. Missing columns: %s", missing)
        raise ValueError(f"Missing required columns: {missing} in file {dest_path}")

    return df


# --------- Command-line convenience ---------
def _parse_required_cols(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [c.strip() for c in raw.split(",") if c.strip()]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest dataset (download or copy into data/raw/) and return a validated dataframe")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="Remote URL to download (CSV/JSON)")
    group.add_argument("--path", type=str, help="Local file path to ingest")
    p.add_argument("--required", type=str, help="Comma-separated required columns (e.g. ts,kwh)")
    p.add_argument("--parse-dates", type=str, help="Comma-separated columns to parse as dates (e.g. ts)")
    p.add_argument("--raw-dir", type=str, help="Override data raw dir")
    args = p.parse_args()

    try:
        req_cols = _parse_required_cols(args.required)
        parse_dates = _parse_required_cols(args.parse_dates)
        if args.url:
            df = ingest_from_url(args.url, raw_dir=args.raw_dir, parse_dates=parse_dates, required_columns=req_cols)
        else:
            df = ingest_from_local(args.path, raw_dir=args.raw_dir, parse_dates=parse_dates, required_columns=req_cols)
        logger.info("Ingestion finished. Data shape: %s", df.shape)
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        raise
