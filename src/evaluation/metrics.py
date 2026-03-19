"""
metrics.py – Tổng hợp metric đánh giá cho classification, clustering, forecasting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score,
    mean_absolute_error, mean_squared_error
)


def classification_metrics(y_true, y_pred, y_prob=None, labels=None) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc_ovr"] = round(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"), 4
            )
        except Exception:
            metrics["roc_auc_ovr"] = None
    return metrics


def clustering_metrics(X, labels) -> dict:
    return {
        "silhouette": round(silhouette_score(X, labels), 4),
        "davies_bouldin": round(davies_bouldin_score(X, labels), 4),
        "n_clusters": len(set(labels)),
    }


def forecasting_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    sp = float(np.mean(np.abs(y_true_arr - y_pred_arr) / denom) * 100)
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "sMAPE(%)": round(sp, 2),
    }


def summarize_results(results_dict: dict) -> pd.DataFrame:
    """Tổng hợp kết quả nhiều model thành DataFrame."""
    rows = []
    for name, m in results_dict.items():
        row = {"Model": name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
#  PHÂN TÍCH LỖI SÂU (Deep Error Analysis)
# ══════════════════════════════════════════════════════════════════

def per_class_error_analysis(y_true, y_pred, class_names: list) -> pd.DataFrame:
    """
    Phân tích lỗi chi tiết từng lớp: precision, recall, f1,
    số mẫu đúng/sai, tỷ lệ lỗi, top lớp bị nhầm sang.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    rows = []
    for i in range(n_classes):
        total = cm[i].sum()
        correct = cm[i, i]
        wrong = total - correct
        error_rate = wrong / total if total > 0 else 0

        # Tìm top lớp bị nhầm sang (top confused-with)
        confused_counts = cm[i].copy()
        confused_counts[i] = 0  # bỏ chính nó
        top_confused_idx = np.argsort(confused_counts)[::-1][:2]
        top_confused = []
        for idx in top_confused_idx:
            if confused_counts[idx] > 0:
                top_confused.append(
                    f"{class_names[idx]} ({confused_counts[idx]:,}, "
                    f"{confused_counts[idx]/total*100:.1f}%)"
                )

        prec = precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
        rec = recall_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)

        rows.append({
            "Class": class_names[i],
            "Total_samples": total,
            "Correct": correct,
            "Wrong": wrong,
            "Error_rate(%)": round(error_rate * 100, 2),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "Top_confused_with": " | ".join(top_confused) if top_confused else "—",
        })
    return pd.DataFrame(rows).sort_values("Error_rate(%)", ascending=False)


def misclassification_matrix_pct(y_true, y_pred, class_names: list) -> pd.DataFrame:
    """
    Ma trận nhầm lẫn dạng phần trăm theo hàng (row-normalized).
    Mỗi ô (i,j) = % mẫu lớp i bị dự đoán thành lớp j.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    df = pd.DataFrame(cm_pct.round(2), index=class_names, columns=class_names)
    df.index.name = "Actual"
    df.columns.name = "Predicted"
    return df


def error_analysis_by_feature(df_test: pd.DataFrame, y_true, y_pred,
                               class_names: list, feature_col: str) -> pd.DataFrame:
    """
    Phân tích tỷ lệ lỗi theo một feature cụ thể (ví dụ: Season, Hour).
    Giúp tìm ra điều kiện nào mô hình yếu nhất.
    """
    df = df_test.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["is_error"] = (df["y_true"] != df["y_pred"]).astype(int)
    df["true_label"] = df["y_true"].map(lambda x: class_names[x])
    df["pred_label"] = df["y_pred"].map(lambda x: class_names[x])

    grouped = df.groupby(feature_col).agg(
        total=("is_error", "count"),
        errors=("is_error", "sum"),
    )
    grouped["error_rate(%)"] = (grouped["errors"] / grouped["total"] * 100).round(2)

    # Top loại nhầm phổ biến nhất trong mỗi nhóm
    top_err = []
    for val in grouped.index:
        sub = df[(df[feature_col] == val) & (df["is_error"] == 1)]
        if len(sub) > 0:
            pair = sub.groupby(["true_label", "pred_label"]).size().idxmax()
            top_err.append(f"{pair[0]}→{pair[1]}")
        else:
            top_err.append("—")
    grouped["top_error_pair"] = top_err
    return grouped.sort_values("error_rate(%)", ascending=False)


def error_analysis_by_season(df_test: pd.DataFrame, y_true, y_pred,
                              class_names: list) -> pd.DataFrame:
    """Phân tích lỗi theo mùa – đặc biệt quan trọng cho đề tài thời tiết."""
    return error_analysis_by_feature(df_test, y_true, y_pred, class_names, "Season")


def error_analysis_by_hour(df_test: pd.DataFrame, y_true, y_pred,
                            class_names: list) -> pd.DataFrame:
    """Phân tích lỗi theo giờ trong ngày."""
    return error_analysis_by_feature(df_test, y_true, y_pred, class_names, "Hour")


def extreme_condition_analysis(df_test: pd.DataFrame, y_true, y_pred,
                                class_names: list) -> pd.DataFrame:
    """
    Phân tích lỗi ở điều kiện cực trị (nhiệt độ rất cao/thấp,
    gió rất mạnh, độ ẩm rất cao).
    """
    df = df_test.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["is_error"] = (y_true != y_pred).astype(int)
    df["true_label"] = df["y_true"].map(lambda x: class_names[x])

    conditions = {
        "Temp < 0°C (very cold)": df["Temperature (C)"] < 0,
        "Temp > 30°C (very hot)": df["Temperature (C)"] > 30,
        "Humidity > 0.9 (very humid)": df["Humidity"] > 0.9,
        "Wind > 25 km/h (strong wind)": df["Wind Speed (km/h)"] > 25,
        "Visibility < 3 km (very low)": df["Visibility (km)"] < 3,
        "Normal conditions": (
            (df["Temperature (C)"].between(5, 25)) &
            (df["Humidity"].between(0.3, 0.7)) &
            (df["Wind Speed (km/h)"] < 15) &
            (df["Visibility (km)"] > 10)
        ),
    }

    rows = []
    for cond_name, mask in conditions.items():
        sub = df[mask]
        if len(sub) == 0:
            continue
        n_err = sub["is_error"].sum()
        err_rate = n_err / len(sub) * 100
        # Dominant true class
        dom_class = sub["true_label"].mode().iloc[0] if len(sub) > 0 else "—"
        rows.append({
            "Condition": cond_name,
            "N_samples": len(sub),
            "N_errors": n_err,
            "Error_rate(%)": round(err_rate, 2),
            "Dominant_class": dom_class,
        })
    return pd.DataFrame(rows).sort_values("Error_rate(%)", ascending=False)


def generate_actionable_insights(
    per_class_df: pd.DataFrame,
    season_df: pd.DataFrame,
    extreme_df: pd.DataFrame,
    ts_results_df: pd.DataFrame = None,
    rules_df: pd.DataFrame = None,
) -> list:
    """
    Tự động sinh 5+ actionable insights dựa trên kết quả phân tích lỗi.
    Mỗi insight gồm: phát hiện + hành động đề xuất.
    """
    insights = []

    # Insight 1: Lớp dễ nhầm nhất
    worst = per_class_df.iloc[0]
    insights.append({
        "id": 1,
        "title": f"Lớp '{worst['Class']}' có tỷ lệ lỗi cao nhất ({worst['Error_rate(%)']}%)",
        "finding": (
            f"Lớp '{worst['Class']}' thường bị nhầm sang: {worst['Top_confused_with']}. "
            f"Trong {worst['Total_samples']:,} mẫu, có {worst['Wrong']:,} mẫu bị phân lớp sai."
        ),
        "action": (
            f"→ Bổ sung đặc trưng phân biệt giữa '{worst['Class']}' và lớp hay nhầm. "
            f"Ví dụ: kết hợp Wind Speed + Visibility + Humidity thành interaction features. "
            f"Có thể dùng SMOTE chỉ cho lớp này nếu thiếu mẫu, hoặc tăng class_weight."
        ),
    })

    # Insight 2: Mùa khó nhất
    if season_df is not None and len(season_df) > 0:
        worst_season = season_df.sort_values("error_rate(%)", ascending=False).iloc[0]
        insights.append({
            "id": 2,
            "title": f"Mùa '{worst_season.name}' có tỷ lệ lỗi cao nhất ({worst_season['error_rate(%)']}%)",
            "finding": (
                f"Giao mùa '{worst_season.name}' là giai đoạn khó phân loại nhất, "
                f"nhầm phổ biến: {worst_season['top_error_pair']}. "
                f"Nguyên nhân: điều kiện thời tiết chuyển tiếp, ranh giới giữa các loại mờ."
            ),
            "action": (
                f"→ Thêm feature 'Season' dạng one-hot hoặc cyclical encoding (sin/cos month) "
                f"vào mô hình. Có thể train mô hình riêng cho từng mùa (ensemble theo mùa)."
            ),
        })

    # Insight 3: Điều kiện cực trị
    if extreme_df is not None and len(extreme_df) > 0:
        worst_ext = extreme_df[extreme_df["Condition"] != "Normal conditions"]
        if len(worst_ext) > 0:
            w = worst_ext.iloc[0]
            normal_row = extreme_df[extreme_df["Condition"] == "Normal conditions"]
            normal_rate = normal_row["Error_rate(%)"].values[0] if len(normal_row) > 0 else 0
            insights.append({
                "id": 3,
                "title": f"Điều kiện cực trị '{w['Condition']}' lỗi {w['Error_rate(%)']}% vs bình thường {normal_rate}%",
                "finding": (
                    f"Khi {w['Condition']}, mô hình sai {w['N_errors']}/{w['N_samples']} mẫu "
                    f"({w['Error_rate(%)']}%), cao hơn đáng kể so với điều kiện bình thường ({normal_rate}%)."
                ),
                "action": (
                    f"→ Tạo feature nhị phân 'is_extreme' cho các điều kiện cực trị. "
                    f"Augment dữ liệu ở vùng cực trị hoặc dùng cost-sensitive learning. "
                    f"Xem xét ensemble: model chính + model chuyên biệt cho cực trị."
                ),
            })

    # Insight 4: Lớp tốt nhất (benchmark)
    best = per_class_df.iloc[-1]
    insights.append({
        "id": 4,
        "title": f"Lớp '{best['Class']}' dễ phân loại nhất (lỗi chỉ {best['Error_rate(%)']}%)",
        "finding": (
            f"'{best['Class']}' có đặc trưng rõ ràng, mô hình nhận diện tốt. "
            f"Precision={best['Precision']}, Recall={best['Recall']}."
        ),
        "action": (
            f"→ Nghiên cứu tại sao '{best['Class']}' dễ phân biệt (feature importance) "
            f"và áp dụng logic tương tự cho các lớp khó hơn. "
            f"Ví dụ: nếu Visibility là feature quyết định cho '{best['Class']}', "
            f"tạo thêm biến phái sinh từ Visibility cho các lớp khác."
        ),
    })

    # Insight 5: Cặp nhầm phổ biến nhất (actionable pair)
    # Tìm cặp (true→pred) phổ biến nhất từ per_class
    confused_pairs = []
    for _, row in per_class_df.iterrows():
        if row["Top_confused_with"] != "—":
            confused_pairs.append((row["Class"], row["Top_confused_with"], row["Wrong"]))
    if confused_pairs:
        worst_pair = max(confused_pairs, key=lambda x: x[2])
        insights.append({
            "id": 5,
            "title": f"Cặp nhầm phổ biến nhất: '{worst_pair[0]}' → {worst_pair[1]}",
            "finding": (
                f"'{worst_pair[0]}' bị nhầm sang {worst_pair[1]} nhiều nhất ({worst_pair[2]:,} mẫu sai tổng). "
                f"Đây là nguồn lỗi chính kéo giảm F1-macro của mô hình."
            ),
            "action": (
                f"→ Xây dựng binary sub-classifier chuyên phân biệt cặp này. "
                f"Hoặc dùng hierarchical classification: phân loại thô trước (nhóm gần), "
                f"rồi phân loại tinh trong từng nhóm. "
                f"Feature engineering: tạo tỷ số Wind/Humidity, delta(Temp, ApparentTemp)."
            ),
        })

    # Insight 6 (bonus): Chuỗi thời gian (nếu có)
    if ts_results_df is not None and len(ts_results_df) > 0:
        best_ts = ts_results_df.iloc[0]
        worst_ts = ts_results_df.iloc[-1]
        insights.append({
            "id": 6,
            "title": f"Holt-Winters/ARIMA vượt baseline {worst_ts['MAE'] - best_ts['MAE']:.2f}°C MAE",
            "finding": (
                f"Model tốt nhất ({best_ts['Model']}) đạt MAE={best_ts['MAE']}°C, "
                f"trong khi baseline ({worst_ts['Model']}) MAE={worst_ts['MAE']}°C."
            ),
            "action": (
                f"→ Seasonality rõ ràng (chu kỳ năm) là yếu tố chính. "
                f"Cải tiến: thêm exogenous variables (Humidity, Pressure) vào SARIMAX. "
                f"Hoặc thử Prophet/LSTM để bắt non-linear patterns."
            ),
        })

    return insights
