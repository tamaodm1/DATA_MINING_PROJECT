"""
anomaly.py – Phát hiện ngày thời tiết bất thường (anomaly mining).
Nhánh thay thế cho bán giám sát (Rubric F).

Kỹ thuật:
  1. Isolation Forest: phát hiện outlier trong không gian đa chiều
  2. Z-score thống kê: phát hiện ngày có chỉ số lệch chuẩn ≥ 3σ
  3. LOF (Local Outlier Factor): phát hiện điểm bất thường dựa trên mật độ cục bộ
Đánh giá:
  - Tỷ lệ anomaly, overlap giữa các phương pháp
  - Phân tích profile ngày bất thường vs bình thường
  - So sánh ≥2 phương pháp
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


NUMERIC_FEATURES = [
    "Temperature (C)", "Apparent Temperature (C)", "Humidity",
    "Wind Speed (km/h)", "Wind Bearing (degrees)",
    "Visibility (km)", "Pressure (millibars)"
]


class WeatherAnomalyDetector:
    def __init__(self, cfg: dict, contamination: float = 0.05):
        """
        contamination: tỷ lệ anomaly kỳ vọng (5% mặc định).
        """
        self.cfg = cfg
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.results = {}

    def prepare_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample dữ liệu hourly → daily aggregation.
        Mỗi ngày: mean(temp, humidity, wind, vis, pressure), std(temp), range(temp).
        """
        df = df.copy()
        df = df.set_index("Formatted Date")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        daily = df[NUMERIC_FEATURES].resample("D").agg(["mean", "std", "min", "max"])
        daily.columns = ["_".join(col) for col in daily.columns]

        # Thêm range cho Temperature
        daily["Temp_range"] = daily["Temperature (C)_max"] - daily["Temperature (C)_min"]

        # Thêm thông tin Season, Month
        daily["Month"] = daily.index.month
        daily["Season"] = daily["Month"].map(
            {12: "Winter", 1: "Winter", 2: "Winter",
             3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer",
             9: "Fall", 10: "Fall", 11: "Fall"}
        )

        daily = daily.dropna()
        print(f"[anomaly] Daily aggregated: {daily.shape}")
        return daily

    def get_feature_matrix(self, daily: pd.DataFrame) -> tuple:
        """Lấy ma trận đặc trưng số + chuẩn hoá."""
        feat_cols = [c for c in daily.columns
                     if daily[c].dtype in [np.float64, np.int64, float, int]
                     and c != "Month"]
        X = daily[feat_cols].fillna(daily[feat_cols].median())
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, feat_cols

    def fit_isolation_forest(self, X: np.ndarray, daily: pd.DataFrame) -> pd.Series:
        """Isolation Forest: phát hiện anomaly."""
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42, n_estimators=200, n_jobs=-1
        )
        labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
        scores = model.decision_function(X)

        result = pd.Series(labels, index=daily.index, name="IsoForest")
        self.results["IsolationForest"] = {
            "labels": result,
            "scores": scores,
            "n_anomaly": (labels == -1).sum(),
            "pct_anomaly": round((labels == -1).mean() * 100, 2),
        }
        n = self.results["IsolationForest"]["n_anomaly"]
        pct = self.results["IsolationForest"]["pct_anomaly"]
        print(f"[anomaly] IsolationForest: {n} anomalies ({pct}%)")
        return result

    def fit_lof(self, X: np.ndarray, daily: pd.DataFrame) -> pd.Series:
        """Local Outlier Factor: phát hiện anomaly dựa trên mật độ."""
        model = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=20, n_jobs=-1
        )
        labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
        scores = model.negative_outlier_factor_

        result = pd.Series(labels, index=daily.index, name="LOF")
        self.results["LOF"] = {
            "labels": result,
            "scores": scores,
            "n_anomaly": (labels == -1).sum(),
            "pct_anomaly": round((labels == -1).mean() * 100, 2),
        }
        n = self.results["LOF"]["n_anomaly"]
        pct = self.results["LOF"]["pct_anomaly"]
        print(f"[anomaly] LOF: {n} anomalies ({pct}%)")
        return result

    def fit_zscore(self, X: np.ndarray, daily: pd.DataFrame,
                   threshold: float = 3.0) -> pd.Series:
        """Z-score: đánh dấu ngày có bất kỳ feature nào |z| > threshold."""
        z = np.abs(X)  # X đã scaled → z-score
        is_anomaly = (z > threshold).any(axis=1)
        labels = np.where(is_anomaly, -1, 1)

        result = pd.Series(labels, index=daily.index, name="ZScore")
        self.results["ZScore"] = {
            "labels": result,
            "scores": z.max(axis=1),
            "n_anomaly": (labels == -1).sum(),
            "pct_anomaly": round((labels == -1).mean() * 100, 2),
        }
        n = self.results["ZScore"]["n_anomaly"]
        pct = self.results["ZScore"]["pct_anomaly"]
        print(f"[anomaly] Z-Score (σ>{threshold}): {n} anomalies ({pct}%)")
        return result

    def compare_methods(self) -> pd.DataFrame:
        """So sánh kết quả giữa các phương pháp."""
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Method": name,
                "N_anomaly": res["n_anomaly"],
                "Pct_anomaly(%)": res["pct_anomaly"],
            })
        return pd.DataFrame(rows)

    def overlap_analysis(self) -> pd.DataFrame:
        """Phân tích overlap: ngày nào được ≥2 phương pháp đánh dấu anomaly."""
        if len(self.results) < 2:
            return pd.DataFrame()

        all_labels = pd.DataFrame({
            name: (res["labels"] == -1).astype(int)
            for name, res in self.results.items()
        })
        all_labels["n_methods_flagged"] = all_labels.sum(axis=1)
        all_labels["consensus_anomaly"] = (all_labels["n_methods_flagged"] >= 2).astype(int)

        n_consensus = all_labels["consensus_anomaly"].sum()
        total = len(all_labels)
        print(f"[anomaly] Consensus (≥2 methods): {n_consensus} anomalies "
              f"({n_consensus/total*100:.2f}%)")
        return all_labels

    def profile_anomalies(self, daily: pd.DataFrame, feat_cols: list,
                          method: str = "IsolationForest") -> pd.DataFrame:
        """
        So sánh profile ngày bất thường vs bình thường.
        Trả về bảng mean từng feature cho 2 nhóm.
        """
        if method not in self.results:
            return pd.DataFrame()

        labels = self.results[method]["labels"]
        daily_copy = daily.copy()
        daily_copy["is_anomaly"] = (labels == -1).astype(int)

        # Chỉ dùng cột numeric
        num_cols = [c for c in feat_cols if c in daily_copy.columns
                    and daily_copy[c].dtype in [np.float64, float]]
        # Lấy top features (mean columns)
        mean_cols = [c for c in num_cols if "_mean" in c][:7]
        if not mean_cols:
            mean_cols = num_cols[:7]

        profile = daily_copy.groupby("is_anomaly")[mean_cols].mean().round(3)
        profile.index = profile.index.map({0: "Normal", 1: "Anomaly"})

        # Thêm count
        counts = daily_copy.groupby("is_anomaly").size()
        profile.insert(0, "Count", counts.values)

        return profile

    def anomaly_by_season(self, daily: pd.DataFrame,
                          method: str = "IsolationForest") -> pd.DataFrame:
        """Phân tích tỷ lệ anomaly theo mùa."""
        if method not in self.results:
            return pd.DataFrame()

        labels = self.results[method]["labels"]
        daily_copy = daily.copy()
        daily_copy["is_anomaly"] = (labels == -1).astype(int)

        season_stats = daily_copy.groupby("Season").agg(
            total=("is_anomaly", "count"),
            n_anomaly=("is_anomaly", "sum"),
        )
        season_stats["pct_anomaly(%)"] = (
            season_stats["n_anomaly"] / season_stats["total"] * 100
        ).round(2)
        return season_stats.sort_values("pct_anomaly(%)", ascending=False)

    def get_top_anomaly_days(self, daily: pd.DataFrame,
                             method: str = "IsolationForest",
                             n: int = 10) -> pd.DataFrame:
        """Lấy top-N ngày bất thường nhất (theo anomaly score)."""
        if method not in self.results:
            return pd.DataFrame()

        scores = self.results[method]["scores"]
        labels = self.results[method]["labels"]
        daily_copy = daily.copy()
        daily_copy["anomaly_score"] = scores
        daily_copy["is_anomaly"] = (labels == -1).astype(int)

        anomalies = daily_copy[daily_copy["is_anomaly"] == 1]
        # IsolationForest: score thấp hơn = bất thường hơn
        # LOF: score âm hơn = bất thường hơn
        anomalies = anomalies.sort_values("anomaly_score", ascending=True)

        # Select readable columns
        show_cols = [c for c in daily_copy.columns if "_mean" in c][:5]
        show_cols += ["anomaly_score"]
        if "Season" in daily_copy.columns:
            show_cols.append("Season")

        return anomalies[show_cols].head(n)
