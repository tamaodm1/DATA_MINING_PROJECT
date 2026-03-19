"""
forecasting.py – Dự báo chuỗi thời gian nhiệt độ/độ ẩm.
               Baseline: Naive/Moving Average
               Models: ARIMA, Holt-Winters (ExponentialSmoothing)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def check_stationarity(series: pd.Series) -> dict:
    """ADF test để kiểm tra tính dừng."""
    result = adfuller(series.dropna())
    return {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05
    }


class TimeSeriesForecaster:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        tc = cfg["timeseries"]
        self.target = tc["target"]
        self.resample_freq = tc["resample_freq"]
        self.train_ratio = tc["train_ratio"]
        self.arima_order = tuple(tc["arima_order"])
        self.horizon = tc["forecast_horizon"]
        self.results = {}

    def prepare_series(self, df: pd.DataFrame) -> pd.Series:
        """Resample thành chuỗi ngày, tính trung bình."""
        df = df.copy()
        df = df.set_index("Formatted Date")
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        series = df[self.target].resample(self.resample_freq).mean().dropna()
        print(f"[forecast] Series length: {len(series)} days, "
              f"{series.index.min()} → {series.index.max()}")
        return series

    def train_test_split_ts(self, series: pd.Series):
        """Split theo thời gian (không shuffle)."""
        n = len(series)
        split_idx = int(n * self.train_ratio)
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        print(f"[forecast] Train: {len(train)}, Test: {len(test)}")
        return train, test

    def naive_baseline(self, train: pd.Series, test: pd.Series) -> pd.Series:
        """Naive: predict = last value in train."""
        pred = pd.Series([train.iloc[-1]] * len(test), index=test.index)
        self._record("Naive", test, pred)
        return pred

    def moving_average(self, train: pd.Series, test: pd.Series, window: int = 7) -> pd.Series:
        """Moving Average baseline."""
        last_ma = train.rolling(window).mean().iloc[-1]
        pred = pd.Series([last_ma] * len(test), index=test.index)
        self._record(f"MA({window})", test, pred)
        return pred

    def fit_arima(self, train: pd.Series, test: pd.Series) -> pd.Series:
        """Fit ARIMA model."""
        print(f"[forecast] Fitting ARIMA{self.arima_order}...")
        model = ARIMA(train, order=self.arima_order)
        fitted = model.fit()
        pred = fitted.forecast(steps=len(test))
        pred.index = test.index
        self._record("ARIMA", test, pred)
        return pred

    def fit_holtwinters(self, train: pd.Series, test: pd.Series) -> pd.Series:
        """Fit Holt-Winters ExponentialSmoothing."""
        print("[forecast] Fitting Holt-Winters...")
        model = ExponentialSmoothing(
            train, trend="add", seasonal="add",
            seasonal_periods=365, damped_trend=True
        )
        fitted = model.fit(optimized=True)
        pred = fitted.forecast(len(test))
        pred.index = test.index
        self._record("HoltWinters", test, pred)
        return pred

    def _record(self, name: str, y_true: pd.Series, y_pred: pd.Series):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        sp = smape(y_true, y_pred)
        self.results[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "sMAPE": round(sp, 4),
            "y_pred": y_pred,
        }
        print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, sMAPE={sp:.2f}%")

    def get_results_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            rows.append({"Model": name, "MAE": res["MAE"],
                          "RMSE": res["RMSE"], "sMAPE(%)": res["sMAPE"]})
        return pd.DataFrame(rows).sort_values("MAE")
