"""
inventory.py
------------
Handles all inventory logging using pandas.
Every scanned package becomes one row in a CSV file that persists
between app restarts (so you don't lose your log when you close Streamlit).
"""

import pandas as pd
import os
from datetime import datetime

LOG_FILE = "logs/inventory_log.csv"

COLUMNS = [
    "timestamp",
    "barcode",
    "width_cm",
    "height_cm",
    "size_category",
    "condition",
    "damage_score",
    "assigned_zone",
]


def load_log() -> pd.DataFrame:
    """Load existing log from disk, or create an empty one if it doesn't exist."""
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame(columns=COLUMNS)


def append_entry(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    """Add a new scan entry to the dataframe and persist it to disk."""
    new_row = pd.DataFrame([entry], columns=COLUMNS)
    df = pd.concat([df, new_row], ignore_index=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    df.to_csv(LOG_FILE, index=False)
    return df


def make_entry(barcode, width_cm, height_cm, size_category, condition, damage_score, zone) -> dict:
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "barcode": barcode if barcode else "UNKNOWN",
        "width_cm": round(width_cm, 1) if width_cm else None,
        "height_cm": round(height_cm, 1) if height_cm else None,
        "size_category": size_category,
        "condition": condition,
        "damage_score": round(damage_score, 3) if damage_score is not None else None,
        "assigned_zone": zone,
    }


def clear_log() -> pd.DataFrame:
    """Delete the persisted log file and return a fresh, empty dataframe."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    return pd.DataFrame(columns=COLUMNS)


def get_stats(df: pd.DataFrame) -> dict:
    """Quick summary stats for the dashboard."""
    if df.empty:
        return {"total": 0, "damaged": 0, "damage_rate": 0.0, "by_size": {}}

    total = len(df)
    damaged = (df["condition"] == "Damaged").sum()
    damage_rate = round((damaged / total) * 100, 1) if total else 0.0
    by_size = df["size_category"].value_counts().to_dict()

    return {
        "total": total,
        "damaged": int(damaged),
        "damage_rate": damage_rate,
        "by_size": by_size,
    }