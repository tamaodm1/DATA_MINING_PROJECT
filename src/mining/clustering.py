"""
clustering.py – Phân cụm ngày kiểu thời tiết với KMeans/HAC/DBSCAN.
                Chọn K tối ưu, profiling cụm.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class WeatherClusterer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        mc = cfg["mining"]["clustering"]
        self.k_range = mc["k_range"]
        self.best_k = mc["best_k"]
        self.model = None

    def find_best_k(self, X: np.ndarray) -> dict:
        """Tính inertia, silhouette, DBI cho range K."""
        results = []
        for k in range(self.k_range[0], self.k_range[-1] + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
            results.append({
                "k": k,
                "inertia": km.inertia_,
                "silhouette": round(sil, 4),
                "dbi": round(dbi, 4)
            })
            print(f"  k={k}: silhouette={sil:.4f}, DBI={dbi:.4f}")
        return pd.DataFrame(results)

    def fit_kmeans(self, X: np.ndarray, k: int = None) -> np.ndarray:
        """Fit KMeans với k tốt nhất."""
        k = k or self.best_k
        self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = self.model.fit_predict(X)
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        print(f"[cluster] KMeans k={k}: silhouette={sil:.4f}, DBI={dbi:.4f}")
        return labels

    def fit_hac(self, X: np.ndarray, k: int = None) -> np.ndarray:
        """Fit Hierarchical Agglomerative Clustering."""
        k = k or self.best_k
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"[cluster] HAC k={k}: silhouette={sil:.4f}")
        return labels

    def profile_clusters(self, df: pd.DataFrame, labels: np.ndarray,
                          feature_cols: list) -> pd.DataFrame:
        """Tạo bảng profile cho từng cụm."""
        df = df.copy()
        df["Cluster"] = labels
        agg = df.groupby("Cluster")[feature_cols].mean().round(3)
        agg["Count"] = df.groupby("Cluster").size()
        agg["Pct"] = (agg["Count"] / len(df) * 100).round(1)

        # Thêm distribution nhãn thời tiết
        if "WeatherType" in df.columns:
            top_weather = df.groupby("Cluster")["WeatherType"].agg(
                lambda x: x.value_counts().index[0]
            )
            agg["DominantWeather"] = top_weather

        if "Season" in df.columns:
            top_season = df.groupby("Cluster")["Season"].agg(
                lambda x: x.value_counts().index[0]
            )
            agg["DominantSeason"] = top_season

        # Đặt tên cụm dựa trên profile
        agg["ClusterName"] = agg.index.map(self._name_cluster(agg))
        return agg

    def _name_cluster(self, profile_df: pd.DataFrame) -> dict:
        """Gợi ý tên cụm theo nhiệt độ & độ ẩm."""
        names = {}
        for idx, row in profile_df.iterrows():
            temp = row.get("Temperature (C)", 10)
            humid = row.get("Humidity", 0.5)
            wind = row.get("Wind Speed (km/h)", 10)
            if temp <= 2:
                name = "Arctic Cold"
            elif temp <= 10 and humid >= 0.75:
                name = "Cold & Damp"
            elif temp >= 22 and humid < 0.6:
                name = "Hot & Dry"
            elif temp >= 18 and humid >= 0.7:
                name = "Warm & Humid"
            elif wind >= 20:
                name = "Windy"
            else:
                name = "Mild"
            names[idx] = name
        return names
