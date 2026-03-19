"""
run_pipeline.py – Chạy toàn bộ pipeline khai phá dữ liệu thời tiết.
Cách dùng:
    python scripts/run_pipeline.py
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Thêm root vào path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.data.loader import load_config, load_raw_data, validate_schema
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder, NUMERIC_FEATURES
from src.mining.association import AssociationMiner
from src.mining.clustering import WeatherClusterer
from src.models.supervised import WeatherClassifier
from src.models.forecasting import TimeSeriesForecaster, check_stationarity
from src.evaluation.metrics import clustering_metrics
from src.evaluation.report import save_table, print_section
from src.visualization import plots

# Setup output dirs
for d in ["outputs/figures", "outputs/tables", "outputs/models", "data/processed"]:
    os.makedirs(d, exist_ok=True)
plots.FIG_DIR = "outputs/figures"


def run():
    cfg = load_config("configs/params.yaml")
    print_section("BƯỚC 1: LOAD & TIỀN XỬ LÝ")

    df_raw = load_raw_data(cfg["paths"]["raw_data"])
    validate_schema(df_raw)
    cleaner = DataCleaner(cfg)
    df = cleaner.clean(df_raw)
    stats = cleaner.get_stats(df_raw, df)
    save_table(stats, "preprocessing_stats")
    df.to_parquet(cfg["paths"]["processed_data"], index=False)
    print(f"Saved processed: {df.shape}")

    # EDA plots
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    plots.plot_temperature_distribution(df)
    plots.plot_weather_type_counts(df, "WeatherType")
    plots.plot_correlation_heatmap(df, num_cols)
    plots.plot_monthly_temperature(df)
    plots.plot_season_weather(df)
    plots.plot_hourly_humidity(df)

    print_section("BƯỚC 2: LUẬT KẾT HỢP")
    builder = FeatureBuilder(cfg)
    basket = builder.build_basket_for_association(df)
    miner = AssociationMiner(cfg)
    freq_items = miner.mine(basket, algorithm="fpgrowth")
    rules = miner.get_rules(freq_items)
    top_rules = miner.top_rules(rules, n=20)
    save_table(top_rules, "association_top_rules")
    if len(rules) > 0:
        plots.plot_top_rules(rules, n=15)
        plots.plot_support_confidence_scatter(rules)

    print_section("BƯỚC 3: PHÂN CỤM")
    from sklearn.utils import resample as sk_resample
    X_scaled, feat_cols = builder.build_scaled_features(df)
    idx = sk_resample(range(len(df)), n_samples=min(20000, len(df)), random_state=42)
    X_sample = X_scaled[idx]
    df_sample = df.iloc[idx].reset_index(drop=True)
    clusterer = WeatherClusterer(cfg)
    k_results = clusterer.find_best_k(X_sample)
    save_table(k_results, "cluster_k_selection")
    plots.plot_elbow_silhouette(k_results)
    labels = clusterer.fit_kmeans(X_sample, k=4)
    profile = clusterer.profile_clusters(df_sample, labels, feat_cols)
    save_table(profile, "cluster_profile")
    plots.plot_cluster_profile(profile, feat_cols)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_sample)
    plots.plot_cluster_scatter(X_2d, labels, "KMeans Clusters (PCA 2D)")

    print_section("BƯỚC 4: PHÂN LỚP")
    from sklearn.model_selection import train_test_split
    feat_cls = [c for c in NUMERIC_FEATURES if c in df.columns] + ["Hour", "Month", "PrecipType_enc"]
    clf = WeatherClassifier(cfg)
    X, y = clf.prepare_Xy(df, feat_cls, "WeatherType")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results_df = clf.train_all(X_train, y_train, X_test, y_test)
    save_table(results_df, "clf_results_comparison")
    plots.plot_model_comparison(results_df, "F1_macro")
    for mname in ["XGBoost", "RandomForest"]:
        if mname in clf.results:
            plots.plot_confusion_matrix_heatmap(
                y_test, clf.results[mname]["y_pred"],
                list(clf.le.classes_), mname
            )
    clf.save_model("XGBoost", "outputs/models")
    print(results_df.to_string(index=False))

    # ── PHÂN TÍCH LỖI SÂU ──────────────────────────────────────
    print_section("BƯỚC 4b: PHÂN TÍCH LỖI SÂU + ACTIONABLE INSIGHTS")
    from src.evaluation.metrics import (
        per_class_error_analysis,
        misclassification_matrix_pct,
        error_analysis_by_season,
        error_analysis_by_hour,
        extreme_condition_analysis,
        generate_actionable_insights,
    )
    class_names = list(clf.le.classes_)
    y_pred_best = clf.results["XGBoost"]["y_pred"]

    # Lấy df_test gốc (với Season, Hour, các cột numeric)
    _, df_test_rows, _, _ = train_test_split(
        df, df["WeatherType"], test_size=0.2, random_state=42, stratify=df["WeatherType"]
    )
    df_test_reset = df_test_rows.reset_index(drop=True)

    # 1. Per-class error analysis
    per_class_df = per_class_error_analysis(y_test, y_pred_best, class_names)
    save_table(per_class_df, "error_per_class_analysis")
    print("\n[Per-class error analysis]")
    print(per_class_df.to_string(index=False))

    # 2. Normalized confusion matrix
    cm_pct_df = misclassification_matrix_pct(y_test, y_pred_best, class_names)
    save_table(cm_pct_df, "error_confusion_matrix_pct")
    plots.plot_normalized_confusion_matrix(y_test, y_pred_best, class_names, "XGBoost")
    plots.plot_per_class_error_rate(per_class_df)

    # 3. Error by season
    season_err = error_analysis_by_season(df_test_reset, y_test, y_pred_best, class_names)
    save_table(season_err, "error_by_season")
    plots.plot_error_by_season(season_err)
    print("\n[Error by season]")
    print(season_err.to_string())

    # 4. Error by hour
    hour_err = error_analysis_by_hour(df_test_reset, y_test, y_pred_best, class_names)
    save_table(hour_err, "error_by_hour")
    plots.plot_error_by_hour(hour_err)

    # 5. Extreme conditions
    extreme_err = extreme_condition_analysis(df_test_reset, y_test, y_pred_best, class_names)
    save_table(extreme_err, "error_extreme_conditions")
    plots.plot_extreme_condition_errors(extreme_err)
    print("\n[Extreme condition errors]")
    print(extreme_err.to_string(index=False))

    print_section("BƯỚC 5: CHUỖI THỜI GIAN")
    forecaster = TimeSeriesForecaster(cfg)
    series = forecaster.prepare_series(df)
    stat_r = check_stationarity(series)
    print("ADF Test:", stat_r)
    plots.plot_acf_pacf(series)
    train_ts, test_ts = forecaster.train_test_split_ts(series)
    pred_naive = forecaster.naive_baseline(train_ts, test_ts)
    pred_ma = forecaster.moving_average(train_ts, test_ts, window=7)
    pred_arima = forecaster.fit_arima(train_ts, test_ts)
    pred_hw = forecaster.fit_holtwinters(train_ts, test_ts)
    ts_results = forecaster.get_results_table()
    save_table(ts_results, "ts_forecast_results")
    print(ts_results.to_string(index=False))
    forecasts = {"Naive": pred_naive, "MA(7)": pred_ma,
                 "ARIMA": pred_arima, "HoltWinters": pred_hw}
    plots.plot_timeseries_forecast(train_ts, test_ts, forecasts)
    best_name = ts_results.iloc[0]["Model"]
    plots.plot_residuals(test_ts, forecasts[best_name], best_name)

    # ── ANOMALY MINING (nhánh thay thế bán giám sát) ────────────
    print_section("BƯỚC 5b: ANOMALY MINING (NHÁNH THAY THẾ BÁN GIÁM SÁT)")
    from src.mining.anomaly import WeatherAnomalyDetector
    detector = WeatherAnomalyDetector(cfg, contamination=0.05)
    daily = detector.prepare_daily(df)
    X_anom, feat_anom = detector.get_feature_matrix(daily)
    labels_iso = detector.fit_isolation_forest(X_anom, daily)
    labels_lof = detector.fit_lof(X_anom, daily)
    labels_z = detector.fit_zscore(X_anom, daily, threshold=3.0)
    compare_df = detector.compare_methods()
    save_table(compare_df, "anomaly_method_comparison")
    print(compare_df.to_string(index=False))
    plots.plot_anomaly_comparison(compare_df)
    plots.plot_anomaly_timeline(daily, labels_iso.values, "IsolationForest")
    overlap = detector.overlap_analysis()
    save_table(overlap.reset_index(), "anomaly_overlap")
    profile_anom = detector.profile_anomalies(daily, feat_anom)
    save_table(profile_anom, "anomaly_profile")
    plots.plot_anomaly_profile(profile_anom)
    season_anom = detector.anomaly_by_season(daily)
    save_table(season_anom, "anomaly_by_season")
    plots.plot_anomaly_by_season(season_anom)

    # ── ACTIONABLE INSIGHTS ─────────────────────────────────────
    print_section("BƯỚC 6: ACTIONABLE INSIGHTS (≥ 5)")
    insights = generate_actionable_insights(
        per_class_df=per_class_df,
        season_df=season_err,
        extreme_df=extreme_err,
        ts_results_df=ts_results,
    )
    for ins in insights:
        print(f"\n{'─' * 50}")
        print(f"  INSIGHT #{ins['id']}: {ins['title']}")
        print(f"{'─' * 50}")
        print(f"  Phát hiện: {ins['finding']}")
        print(f"  Hành động: {ins['action']}")
    save_table(pd.DataFrame(insights), "actionable_insights")
    plots.plot_actionable_insights_summary(insights)

    print_section("PIPELINE HOÀN TẤT")
    print("✓ Figures:", os.listdir("outputs/figures"))
    print("✓ Tables:", os.listdir("outputs/tables"))
    print("✓ Models:", os.listdir("outputs/models"))


if __name__ == "__main__":
    run()
