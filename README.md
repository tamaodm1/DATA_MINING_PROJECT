# Đề tài 5: Dự báo Thời Tiết – Data Mining Project

**Học phần:** Dữ liệu lớn, Khai phá dữ liệu  
**Học kỳ II – Năm học 2025–2026**  
**Giảng viên:** ThS. Lê Thị Thùy Trang

---

## Dataset

**Nguồn:** [Kaggle – Weather History Dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset)  
**File:** `data/raw/weatherHistory.csv`  
**Kích thước:** 96,453 hàng × 12 cột (dữ liệu theo giờ, 2006–2016)

### Data Dictionary

| Cột | Kiểu | Ý nghĩa |
|-----|------|---------|
| Formatted Date | datetime | Thời điểm đo (có timezone) |
| Summary | string | Mô tả thời tiết gốc |
| Precip Type | string | Loại mưa: rain / snow |
| Temperature (C) | float | Nhiệt độ (°C) |
| Apparent Temperature (C) | float | Nhiệt độ cảm nhận (°C) |
| Humidity | float | Độ ẩm (0–1) |
| Wind Speed (km/h) | float | Tốc độ gió |
| Wind Bearing (degrees) | float | Hướng gió (độ) |
| Visibility (km) | float | Tầm nhìn |
| Loud Cover | float | Mức độ mây phủ (toàn bộ = 0 trong dataset này) |
| Pressure (millibars) | float | Áp suất khí quyển |
| Daily Summary | string | Mô tả thời tiết ngày |

**Target:** `WeatherType` (nhãn được tổng hợp từ `Summary`): Clear, Cloudy, Foggy, Windy, Rainy

**Rủi ro dữ liệu:**
- `Loud Cover` toàn bộ = 0 → bỏ cột
- `Precip Type` thiếu 517 bản ghi → fill "unknown"
- `Pressure (millibars)` có vài bản ghi = 0 → outlier, loại bỏ
- Mất cân bằng lớp: Cloudy chiếm ~80%

---

## Cấu trúc thư mục

```
DATA_MINING_PROJECT/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── params.yaml              # Tham số toàn cục
├── data/
│   ├── raw/                     # Dữ liệu gốc (không commit nếu lớn)
│   └── processed/               # Dữ liệu sau tiền xử lý (.parquet)
├── notebooks/
│   ├── 01_eda.ipynb             # EDA & Preprocessing
│   ├── 02_association_mining.ipynb  # Luật kết hợp
│   ├── 03_clustering.ipynb      # Phân cụm
│   ├── 04_modeling.ipynb        # Phân lớp
│   ├── 04b_anomaly_mining.ipynb # Nhánh thay thế
│   └── 05_evaluation_report.ipynb   # Chuỗi thời gian & Tổng kết
├── src/
│   ├── data/
│   │   ├── loader.py            # Đọc & validate dữ liệu
│   │   └── cleaner.py           # Làm sạch, encoding, tạo nhãn
│   ├── features/
│   │   └── builder.py           # Feature engineering (lag, discretize, basket)
│   ├── mining/
│   │   ├── association.py       # FP-Growth / Apriori
│   │   └── clustering.py        # KMeans / HAC + profiling
│   ├── models/
│   │   ├── supervised.py        # Classification (LR, RF, XGBoost)
│   │   └── forecasting.py       # ARIMA, Holt-Winters, Naive
│   ├── evaluation/
│   │   ├── metrics.py           # Metric tổng hợp
│   │   └── report.py            # Lưu bảng kết quả
│   └── visualization/
│       └── plots.py             # Hàm vẽ dùng chung
├── scripts/
│   ├── run_pipeline.py          # Chạy toàn bộ pipeline
│   └── run_papermill.py         # Chạy notebook bằng papermill
└── outputs/
    ├── figures/                 # Hình ảnh EDA, clustering, forecast
    ├── tables/                  # CSV kết quả
    ├── models/                  # Model đã lưu (.joblib)
    └── reports/
```

---

## Quy trình khai phá (Pipeline)

```
weatherHistory.csv
       │
       ▼
[1] Preprocessing   → làm sạch, encoding, tạo WeatherType, lag features
       │
       ├──▶ [2] Association Mining  → FP-Growth, luật theo mùa
       │
       ├──▶ [3] Clustering          → KMeans k=4, profile cụm
       │
       ├──▶ [4] Classification      → LogReg/RF/XGBoost, F1-macro
       │
       └──▶ [5] Time Series         → Naive/MA/ARIMA/Holt-Winters, MAE/RMSE
```

---

## Cài đặt & Chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Tải dữ liệu

Tải file từ Kaggle (link trên) và đặt vào:
```
data/raw/weatherHistory.csv
```

### 3. Cập nhật đường dẫn (nếu cần)

Mở `configs/params.yaml` và chỉnh `paths.raw_data`.

### 4. Chạy toàn bộ pipeline

```bash
python scripts/run_pipeline.py
```

Hoặc chạy từng notebook theo thứ tự trong `notebooks/`.

### 5. Xem kết quả

```
outputs/figures/   ← Tất cả hình ảnh
outputs/tables/    ← Bảng kết quả CSV
outputs/models/    ← Model đã lưu
```

---

## Kết quả chính

| Thành phần | Kết quả |
|-----------|---------|
| Association Mining | Luật nổi bật: VeryHumid+LowVis → Foggy (lift cao) |
| Clustering (k=4) | Silhouette > 0.35; 4 cụm: Arctic Cold, Cold&Damp, Warm&Humid, Hot&Dry |
| Classification | XGBoost F1-macro tốt nhất, vượt baseline ~25% |
| Forecasting | Holt-Winters MAE thấp nhất (~2°C) |

---

## Metrics sử dụng

- **Classification:** F1-macro, F1-weighted, ROC-AUC (OvR), Confusion Matrix
- **Clustering:** Silhouette Score, Davies-Bouldin Index
- **Association:** Support, Confidence, Lift
- **Forecasting:** MAE, RMSE, sMAPE
