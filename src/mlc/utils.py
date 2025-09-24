import os
import random
import logging
from typing import Any, Dict
import joblib
import numpy as np

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_deterministic(seed: int = 42) -> None:
    set_seed(seed)

set_global_seed = set_seed

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    else:
        logger.setLevel(level)
    return logger

def save_object(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_object(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return joblib.load(path)

def log_metrics(metrics: Dict[str, float], logger: logging.Logger) -> None:
    for k, v in metrics.items():
        try:
            logger.info(f"{k}: {v:.4f}")
        except Exception:
            logger.info(f"{k}: {v}")