"""
cleaner.py – Làm sạch, xử lý missing/outlier, encoding, tạo nhãn nhóm thời tiết.
"""
import pandas as pd
import numpy as np


# Ánh xạ Summary -> nhóm WeatherType (5 nhóm chính)
WEATHER_GROUP_MAP = {
    "Clear": "Clear",
    "Partly Cloudy": "Cloudy",
    "Mostly Cloudy": "Cloudy",
    "Overcast": "Cloudy",
    "Foggy": "Foggy",
    "Breezy and Overcast": "Windy",
    "Breezy and Mostly Cloudy": "Windy",
    "Breezy and Partly Cloudy": "Windy",
    "Breezy": "Windy",
    "Windy and Overcast": "Windy",
    "Windy and Partly Cloudy": "Windy",
    "Windy and Mostly Cloudy": "Windy",
    "Light Rain": "Rainy",
    "Drizzle": "Rainy",
    "Rain": "Rainy",
    "Heavy Rain": "Rainy",
    "Dry": "Clear",
    "Dry and Partly Cloudy": "Cloudy",
    "Dry and Mostly Cloudy": "Cloudy",
    "Humid and Mostly Cloudy": "Cloudy",
    "Humid and Partly Cloudy": "Cloudy",
    "Breezy and Foggy": "Foggy",
}


class DataCleaner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.drop_cols = cfg["preprocessing"].get("drop_columns", [])
        self.datetime_col = cfg["preprocessing"]["datetime_col"]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        print(f"[cleaner] Input shape: {df.shape}")

        # 1. Drop cột không cần thiết
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        # 2. Parse datetime nếu chưa parse
        if df[self.datetime_col].dtype == object:
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], utc=True)

        # 3. Xử lý missing values
        before = len(df)
        df["Precip Type"] = df["Precip Type"].fillna("unknown")
        df = df.dropna(subset=["Temperature (C)", "Humidity", "Pressure (millibars)"])
        print(f"[cleaner] Dropped {before - len(df)} rows with missing numeric values")

        # 4. Loại bỏ Pressure = 0 (bất thường rõ ràng)
        before = len(df)
        df = df[df["Pressure (millibars)"] > 800]
        print(f"[cleaner] Dropped {before - len(df)} rows with Pressure <= 800 (outlier)")

        # 5. Tạo nhãn nhóm thời tiết
        df["WeatherType"] = df["Summary"].map(WEATHER_GROUP_MAP)
        df["WeatherType"] = df["WeatherType"].fillna("Other")
        df = df[df["WeatherType"] != "Other"]  # Loại nhãn hiếm

        # 6. Tạo các đặc trưng thời gian
        dt = df[self.datetime_col].dt
        df["Hour"] = dt.hour
        df["Month"] = dt.month
        df["DayOfWeek"] = dt.dayofweek
        df["Season"] = df["Month"].map(
            {12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Fall", 10: "Fall", 11: "Fall"}
        )

        # 7. Encode PrecipType
        df["PrecipType_enc"] = df["Precip Type"].map({"rain": 0, "snow": 1, "unknown": -1})

        print(f"[cleaner] Output shape: {df.shape}")
        print(f"[cleaner] WeatherType distribution:\n{df['WeatherType'].value_counts()}")
        return df

    def get_stats(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """So sánh thống kê trước/sau tiền xử lý."""
        numeric = ["Temperature (C)", "Humidity", "Wind Speed (km/h)",
                   "Visibility (km)", "Pressure (millibars)"]
        rows = []
        for col in numeric:
            if col in df_before.columns and col in df_after.columns:
                rows.append({
                    "Column": col,
                    "Before_mean": round(df_before[col].mean(), 3),
                    "After_mean": round(df_after[col].mean(), 3),
                    "Before_missing": df_before[col].isnull().sum(),
                    "After_missing": df_after[col].isnull().sum(),
                    "Before_rows": len(df_before),
                    "After_rows": len(df_after),
                })
        return pd.DataFrame(rows)
