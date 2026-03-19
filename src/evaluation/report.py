"""
report.py – Tổng hợp bảng và lưu kết quả ra outputs/tables/.
"""
import pandas as pd
import os


def save_table(df: pd.DataFrame, name: str, output_dir: str = "outputs/tables"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(path, index=True)
    print(f"[report] Saved table: {path}")
    return path


def print_section(title: str):
    sep = "=" * 60
    print(f"\n{sep}\n  {title}\n{sep}")
