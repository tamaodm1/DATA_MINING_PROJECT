"""
plots.py – Hàm vẽ dùng chung cho toàn bộ pipeline.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = "Set2"
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def savefig(name: str):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"[plots] Saved: {path}")
    return path


# ── EDA ──────────────────────────────────────────────────────────
def plot_temperature_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["Temperature (C)"], bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("Phân phối nhiệt độ (°C)")
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Số lượng")
    axes[1].boxplot(df["Temperature (C)"], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    axes[1].set_title("Boxplot nhiệt độ")
    savefig("eda_temperature_dist")


def plot_weather_type_counts(df: pd.DataFrame, col: str = "WeatherType"):
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = df[col].value_counts()
    vc.plot(kind="barh", ax=ax, color=sns.color_palette(PALETTE, len(vc)))
    ax.set_title(f"Phân phối {col}")
    ax.set_xlabel("Số lượng bản ghi")
    for bar, val in zip(ax.patches, vc.values):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    savefig(f"eda_{col}_counts")


def plot_correlation_heatmap(df: pd.DataFrame, cols: list):
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title("Ma trận tương quan đặc trưng thời tiết")
    savefig("eda_correlation_heatmap")


def plot_monthly_temperature(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly = df.groupby("Month")["Temperature (C)"].mean()
    ax.plot(monthly.index, monthly.values, "o-", color="#4C72B0", linewidth=2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_title("Nhiệt độ trung bình theo tháng")
    ax.set_ylabel("Temperature (°C)")
    savefig("eda_monthly_temperature")


def plot_season_weather(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(df["Season"], df["WeatherType"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, colormap=PALETTE)
    ax.set_title("Phân bố loại thời tiết theo mùa (%)")
    ax.set_xlabel("Mùa")
    ax.set_ylabel("Tỷ lệ (%)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    savefig("eda_season_weather")


def plot_hourly_humidity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    hourly = df.groupby("Hour")["Humidity"].mean()
    ax.bar(hourly.index, hourly.values, color="#55A868")
    ax.set_title("Độ ẩm trung bình theo giờ trong ngày")
    ax.set_xlabel("Giờ")
    ax.set_ylabel("Humidity")
    ax.set_xticks(range(0, 24))
    savefig("eda_hourly_humidity")


# ── Clustering ───────────────────────────────────────────────────
def plot_elbow_silhouette(k_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(k_df["k"], k_df["inertia"], "o-", color="#4C72B0")
    axes[0].set_title("Elbow – Inertia")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_df["k"], k_df["silhouette"], "o-", color="#55A868")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")

    axes[2].plot(k_df["k"], k_df["dbi"], "o-", color="#C44E52")
    axes[2].set_title("Davies-Bouldin Index")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("DBI (↓)")
    savefig("cluster_elbow_silhouette_dbi")


def plot_cluster_profile(profile_df: pd.DataFrame, feature_cols: list):
    """Radar/bar chart profile từng cụm."""
    data = profile_df[feature_cols].copy()
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(feature_cols))
    width = 0.18
    colors = sns.color_palette(PALETTE, len(data_norm))
    for i, (idx, row) in enumerate(data_norm.iterrows()):
        label = profile_df.loc[idx, "ClusterName"] if "ClusterName" in profile_df.columns else f"C{idx}"
        ax.bar(x + i * width, row.values, width, label=label, color=colors[i], alpha=0.8)
    ax.set_xticks(x + width * len(data_norm) / 2)
    ax.set_xticklabels([c.replace(" (C)", "").replace(" (km/h)", "")
                         .replace(" (km)", "").replace(" (millibars)", "")
                         for c in feature_cols], rotation=30, ha="right")
    ax.set_title("Profile từng cụm (chuẩn hoá 0-1)")
    ax.legend(loc="upper right", fontsize=8)
    savefig("cluster_profile_bar")


def plot_cluster_scatter(X_2d: np.ndarray, labels: np.ndarray, title: str = "Clusters"):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                          cmap="Set2", alpha=0.4, s=5)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    savefig("cluster_scatter_pca")


# ── Association ──────────────────────────────────────────────────
def plot_top_rules(rules_df: pd.DataFrame, n: int = 15):
    top = rules_df.head(n).copy()
    top["rule_str"] = (top["antecedents"].apply(lambda x: ", ".join(list(x)))
                       + " → " + top["consequents"].apply(lambda x: ", ".join(list(x))))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top["rule_str"][::-1], top["lift"][::-1],
                    color="#4C72B0", alpha=0.8)
    ax.set_xlabel("Lift")
    ax.set_title(f"Top-{n} Luật kết hợp (theo Lift)")
    for bar, val in zip(bars, top["lift"][::-1].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)
    savefig("association_top_rules_lift")


def plot_support_confidence_scatter(rules_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(rules_df["support"], rules_df["confidence"],
                     c=rules_df["lift"], cmap="YlOrRd", alpha=0.7, s=30)
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Support – Confidence – Lift")
    savefig("association_support_conf_lift_scatter")


# ── Classification ───────────────────────────────────────────────
def plot_confusion_matrix_heatmap(y_true, y_pred, labels: list, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {model_name}")
    savefig(f"clf_confusion_{model_name.lower().replace(' ', '_')}")


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "F1_macro"):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#C44E52" if "Baseline" in m else "#4C72B0"
              for m in results_df["Model"]]
    bars = ax.barh(results_df["Model"], results_df[metric], color=colors, alpha=0.85)
    ax.set_xlabel(metric)
    ax.set_title(f"So sánh mô hình – {metric}")
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    savefig(f"clf_model_comparison_{metric.lower()}")


# ── Time Series ──────────────────────────────────────────────────
def plot_timeseries_forecast(train, test, forecasts: dict, title: str = "Dự báo nhiệt độ"):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index[-90:], train.values[-90:], color="black",
             linewidth=1.5, label="Train (90 ngày cuối)")
    ax.plot(test.index, test.values, color="gray", linewidth=1.5,
             linestyle="--", label="Actual")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for (name, pred), color in zip(forecasts.items(), colors):
        ax.plot(test.index, pred.values, color=color,
                 linewidth=1.5, label=name, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(fontsize=9)
    savefig("ts_forecast_comparison")


def plot_residuals(test, pred, model_name: str):
    residuals = test.values - pred.values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(test.index, residuals, color="#4C72B0", alpha=0.7)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"Residuals – {model_name}")
    axes[0].set_xlabel("Ngày")
    axes[1].hist(residuals, bins=30, color="#4C72B0", edgecolor="white")
    axes[1].set_title("Phân phối residuals")
    axes[1].set_xlabel("Residual (°C)")
    savefig(f"ts_residuals_{model_name.lower().replace(' ', '_')}")


def plot_acf_pacf(series: pd.Series, lags: int = 40):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title="ACF – Nhiệt độ")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title="PACF – Nhiệt độ",
              method="ywm")
    savefig("ts_acf_pacf")


# ══════════════════════════════════════════════════════════════════
#  PHÂN TÍCH LỖI SÂU – BIỂU ĐỒ
# ══════════════════════════════════════════════════════════════════

def plot_normalized_confusion_matrix(y_true, y_pred, labels: list, model_name: str):
    """Confusion matrix chuẩn hoá theo hàng (% mỗi lớp thật)."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="OrRd",
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=0, vmax=100, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (%) – {model_name}\n(mỗi hàng tổng = 100%)")
    savefig(f"error_confusion_pct_{model_name.lower()}")


def plot_per_class_error_rate(per_class_df):
    """Bar chart tỷ lệ lỗi từng lớp."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#C44E52" if e > 20 else "#E8A838" if e > 10 else "#55A868"
              for e in per_class_df["Error_rate(%)"]]
    bars = ax.barh(per_class_df["Class"], per_class_df["Error_rate(%)"],
                    color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Tỷ lệ lỗi (%)")
    ax.set_title("Tỷ lệ phân lớp sai theo từng loại thời tiết")
    for bar, val in zip(bars, per_class_df["Error_rate(%)"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.axvline(x=per_class_df["Error_rate(%)"].mean(), color="gray",
               linestyle="--", linewidth=1, label=f"TB = {per_class_df['Error_rate(%)'].mean():.1f}%")
    ax.legend(fontsize=9)
    savefig("error_per_class_rate")


def plot_error_by_season(season_df):
    """Bar chart tỷ lệ lỗi theo mùa."""
    fig, ax = plt.subplots(figsize=(8, 5))
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    plot_data = season_df.reindex([s for s in season_order if s in season_df.index])
    colors = sns.color_palette("coolwarm", len(plot_data))
    bars = ax.bar(plot_data.index, plot_data["error_rate(%)"], color=colors,
                   edgecolor="white", alpha=0.85)
    ax.set_ylabel("Tỷ lệ lỗi (%)")
    ax.set_title("Tỷ lệ phân lớp sai theo mùa")
    for bar, val, pair in zip(bars, plot_data["error_rate(%)"], plot_data["top_error_pair"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%\n({pair})", ha="center", fontsize=9)
    savefig("error_by_season")


def plot_error_by_hour(hour_df):
    """Line chart tỷ lệ lỗi theo giờ trong ngày."""
    fig, ax = plt.subplots(figsize=(12, 4))
    hour_df_sorted = hour_df.sort_index()
    ax.plot(hour_df_sorted.index, hour_df_sorted["error_rate(%)"],
            "o-", color="#4C72B0", linewidth=2, markersize=5)
    ax.fill_between(hour_df_sorted.index, hour_df_sorted["error_rate(%)"],
                     alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("Tỷ lệ lỗi (%)")
    ax.set_title("Tỷ lệ phân lớp sai theo giờ")
    ax.set_xticks(range(0, 24))
    avg = hour_df_sorted["error_rate(%)"].mean()
    ax.axhline(y=avg, color="red", linestyle="--", linewidth=1,
               label=f"TB = {avg:.1f}%")
    ax.legend(fontsize=9)
    savefig("error_by_hour")


def plot_extreme_condition_errors(extreme_df):
    """Bar chart lỗi theo điều kiện cực trị."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#C44E52" if "Normal" not in c else "#55A868"
              for c in extreme_df["Condition"]]
    bars = ax.barh(extreme_df["Condition"], extreme_df["Error_rate(%)"],
                    color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Tỷ lệ lỗi (%)")
    ax.set_title("Tỷ lệ lỗi theo điều kiện thời tiết (cực trị vs bình thường)")
    for bar, val, n in zip(bars, extreme_df["Error_rate(%)"], extreme_df["N_samples"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}% (n={n:,})", va="center", fontsize=9)
    savefig("error_extreme_conditions")


def plot_actionable_insights_summary(insights: list):
    """Tạo hình tổng hợp 5+ insights dưới dạng bảng visual."""
    fig, ax = plt.subplots(figsize=(14, max(3, len(insights) * 1.2)))
    ax.axis("off")
    ax.set_title("ACTIONABLE INSIGHTS – TỔNG HỢP", fontsize=14,
                  fontweight="bold", pad=20, loc="left")

    y = 0.95
    for ins in insights:
        ax.text(0.02, y, f"#{ins['id']}", fontsize=11, fontweight="bold",
                color="#C44E52", transform=ax.transAxes, va="top")
        ax.text(0.06, y, ins["title"], fontsize=10, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.06
        ax.text(0.06, y, ins["finding"], fontsize=8.5, color="#333333",
                transform=ax.transAxes, va="top", wrap=True,
                fontfamily="monospace")
        y -= 0.06
        ax.text(0.06, y, ins["action"], fontsize=8.5, color="#1a5276",
                transform=ax.transAxes, va="top", wrap=True,
                fontstyle="italic")
        y -= 0.09

    savefig("actionable_insights_summary")


# ══════════════════════════════════════════════════════════════════
#  ANOMALY MINING – BIỂU ĐỒ
# ══════════════════════════════════════════════════════════════════

def plot_anomaly_timeline(daily, labels, method_name: str = "IsolationForest"):
    """Scatter plot timeline: ngày bình thường vs bất thường."""
    fig, ax = plt.subplots(figsize=(14, 5))
    temp_col = [c for c in daily.columns if "Temperature" in c and "_mean" in c]
    if not temp_col:
        temp_col = [daily.select_dtypes(include=[np.number]).columns[0]]
    y_col = temp_col[0]

    normal = daily[labels == 1]
    anomaly = daily[labels == -1]

    ax.scatter(normal.index, normal[y_col], c="#4C72B0", s=5, alpha=0.4, label="Normal")
    ax.scatter(anomaly.index, anomaly[y_col], c="#C44E52", s=25, alpha=0.9,
               label=f"Anomaly ({len(anomaly)})", marker="x", linewidths=1.5)
    ax.set_title(f"Anomaly Detection – {method_name}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel(y_col)
    ax.legend(fontsize=9)
    savefig(f"anomaly_timeline_{method_name.lower()}")


def plot_anomaly_comparison(compare_df):
    """Bar chart so sánh số anomaly giữa các phương pháp."""
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("Set2", len(compare_df))
    bars = ax.barh(compare_df["Method"], compare_df["N_anomaly"],
                    color=colors, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Số ngày bất thường")
    ax.set_title("So sánh phương pháp phát hiện anomaly")
    for bar, pct in zip(bars, compare_df["Pct_anomaly(%)"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct}%", va="center", fontsize=10)
    savefig("anomaly_method_comparison")


def plot_anomaly_profile(profile_df):
    """Bar chart so sánh profile Normal vs Anomaly."""
    # Bỏ cột Count
    data = profile_df.drop(columns=["Count"], errors="ignore")
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(data_norm.columns))
    width = 0.35
    colors = ["#4C72B0", "#C44E52"]

    for i, (idx, row) in enumerate(data_norm.iterrows()):
        ax.bar(x + i * width, row.values, width, label=idx, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width / 2)
    short_names = [c.replace("_mean", "").replace("Temperature (C)", "Temp")
                    .replace("Wind Speed (km/h)", "Wind")
                    .replace("Visibility (km)", "Vis")
                    .replace("Pressure (millibars)", "Press")
                    .replace("Apparent Temperature (C)", "AppTemp")
                   for c in data_norm.columns]
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_title("Profile: Normal vs Anomaly (chuẩn hoá 0-1)")
    ax.legend(fontsize=10)
    savefig("anomaly_profile_comparison")


def plot_anomaly_by_season(season_df):
    """Bar chart tỷ lệ anomaly theo mùa."""
    fig, ax = plt.subplots(figsize=(8, 4))
    season_order = ["Spring", "Summer", "Fall", "Winter"]
    plot_data = season_df.reindex([s for s in season_order if s in season_df.index])
    colors = sns.color_palette("coolwarm", len(plot_data))
    bars = ax.bar(plot_data.index, plot_data["pct_anomaly(%)"], color=colors, alpha=0.85)
    ax.set_ylabel("Tỷ lệ anomaly (%)")
    ax.set_title("Tỷ lệ ngày bất thường theo mùa")
    for bar, val, n in zip(bars, plot_data["pct_anomaly(%)"], plot_data["n_anomaly"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val}%\n(n={n})", ha="center", fontsize=9)
    savefig("anomaly_by_season")
