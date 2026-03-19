"""
builder.py – Feature engineering: lag features, rolling stats, rời rạc hoá,
             chuẩn hoá cho clustering/classification, và basket building cho association.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


NUMERIC_FEATURES = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Wind Bearing (degrees)",
    "Visibility (km)", "Pressure (millibars)"
]


class FeatureBuilder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.le = LabelEncoder()

    def build_lag_features(self, df: pd.DataFrame, target_col: str = "Temperature (C)",
                           lags: list = [1, 2, 3, 24]) -> pd.DataFrame:
        """Tạo lag features và rolling mean/std cho chuỗi thời gian."""
        df = df.copy().sort_values("Formatted Date").reset_index(drop=True)
        for lag in lags:
            df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
        for win in [24, 168]:  # 1 ngày, 1 tuần
            df[f"{target_col}_roll{win}_mean"] = df[target_col].rolling(win).mean()
            df[f"{target_col}_roll{win}_std"] = df[target_col].rolling(win).std()
        df = df.dropna()
        print(f"[builder] Lag features built. Shape: {df.shape}")
        return df

    def build_scaled_features(self, df: pd.DataFrame) -> np.ndarray:
        """Chuẩn hoá numeric features để clustering/classification."""
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        X = df[cols].fillna(df[cols].median())
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, cols

    def discretize_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rời rạc hoá điều kiện thời tiết thành bins để làm Association Mining.
        Tạo các cột dạng: Temp_bin, Humid_bin, Wind_bin, Vis_bin, Press_bin
        """
        df = df.copy()
        df["Temp_bin"] = pd.cut(
            df["Temperature (C)"],
            bins=[-np.inf, 0, 10, 20, np.inf],
            labels=["Cold", "Cool", "Warm", "Hot"]
        )
        df["Humid_bin"] = pd.cut(
            df["Humidity"],
            bins=[0, 0.4, 0.6, 0.8, 1.01],
            labels=["Dry", "Moderate", "Humid", "VeryHumid"]
        )
        df["Wind_bin"] = pd.cut(
            df["Wind Speed (km/h)"],
            bins=[-np.inf, 10, 20, 30, np.inf],
            labels=["Calm", "Breezy", "Windy", "Stormy"]
        )
        df["Vis_bin"] = pd.cut(
            df["Visibility (km)"],
            bins=[-np.inf, 5, 10, 16.1],
            labels=["LowVis", "ModVis", "HighVis"]
        )
        df["Press_bin"] = pd.cut(
            df["Pressure (millibars)"],
            bins=[-np.inf, 1005, 1015, 1025, np.inf],
            labels=["LowPress", "NormPress", "HighPress", "VHighPress"]
        )
        return df

    def build_basket_for_association(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo basket cho luật kết hợp: mỗi hàng là 1 "giao dịch" theo ngày,
        các cột là điều kiện rời rạc + loại thời tiết.
        """
        df = self.discretize_conditions(df)
        # Dùng từng bản ghi (hàng theo giờ) làm 1 transaction
        item_cols = ["Temp_bin", "Humid_bin", "Wind_bin", "Vis_bin", "Press_bin", "WeatherType"]
        basket_df = df[item_cols].astype(str)
        # One-hot encode để dùng với mlxtend
        basket_encoded = pd.get_dummies(basket_df, prefix_sep="=")
        return basket_encoded.astype(bool)

    def encode_labels(self, series: pd.Series) -> np.ndarray:
        return self.le.fit_transform(series)

    def get_feature_names(self) -> list:
        return NUMERIC_FEATURES
