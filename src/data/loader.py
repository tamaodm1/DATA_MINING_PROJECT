"""
loader.py – Đọc dữ liệu thô và kiểm tra schema.
"""
import pandas as pd
import yaml
import os


def load_config(config_path: str = "configs/params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    """Đọc file CSV thô, parse datetime."""
    df = pd.read_csv(path)
    # Parse datetime
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    print(f"[loader] Loaded {len(df):,} rows × {df.shape[1]} cols from {path}")
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Kiểm tra schema cơ bản."""
    required_cols = [
        "Formatted Date", "Summary", "Precip Type",
        "Temperature (C)", "Apparent Temperature (C)",
        "Humidity", "Wind Speed (km/h)", "Wind Bearing (degrees)",
        "Visibility (km)", "Loud Cover", "Pressure (millibars)", "Daily Summary"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[loader] Missing columns: {missing}")
        return False
    print("[loader] Schema OK.")
    return True


def load_processed_data(path: str) -> pd.DataFrame:
    """Đọc dữ liệu đã xử lý từ parquet."""
    df = pd.read_parquet(path)
    print(f"[loader] Loaded processed data: {df.shape}")
    return df
