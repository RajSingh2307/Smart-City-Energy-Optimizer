# backend/utils.py
"""
Utility functions for SmartCity Energy Optimizer.
- Centralized config loader
- Logger setup with rotation
- Common helpers (paths, retries, JSON I/O)
"""

import os
import sys
import json
import yaml
import logging
import logging.handlers
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Cache loaded config to avoid repeated file reads
_config_cache: Optional[Dict[str, Any]] = None

# --- CONFIG LOADER ---
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML config with env var overrides.
    """
    global _config_cache
    if _config_cache:
        return _config_cache

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    # Override with env vars if set
    db_url = os.getenv("DB_URL")
    if db_url:
        config.setdefault("database", {})["url"] = db_url

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        config.setdefault("weather", {})["openweather_api_key"] = api_key

    _config_cache = config
    return config


# --- LOGGER SETUP ---
def get_logger(name: str = "smartcity") -> logging.Logger:
    """
    Get a project-wide logger with rotation.
    """
    config = load_config()
    log_cfg = config.get("logging", {})

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already initialized

    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(console)

    # Rotating file handler
    if log_cfg.get("rotate", {}).get("enabled", True):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        handler = logging.handlers.TimedRotatingFileHandler(
            log_dir / "app.log",
            when=log_cfg["rotate"].get("when", "midnight"),
            backupCount=log_cfg["rotate"].get("backup_count", 7),
            encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)

    return logger


# --- HELPER FUNCTIONS ---
def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return its Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def retry(times: int = 3, exceptions: tuple = (Exception,), delay: float = 1.0) -> Callable:
    """
    Decorator to retry a function call on exceptions.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(times):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_err = e
                    logger = get_logger()
                    logger.warning(f"Retry {attempt+1}/{times} for {func.__name__}: {e}")
                    if delay > 0:
                        import time; time.sleep(delay)
            raise last_err
        return wrapper
    return decorator


def save_json(data: Any, path: str | Path):
    """Save dict/list as JSON with safe UTF-8 encoding."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON file safely."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
