"""
supervised.py – Train/predict cho bài toán phân lớp loại thời tiết.
               Baseline: DummyClassifier, LogisticRegression
               Mô hình mạnh: RandomForest, XGBoost
"""
import pandas as pd
import numpy as np
import time
import joblib
import os

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODELS = {
    "Baseline_Dummy": DummyClassifier(strategy="most_frequent", random_state=42),
    "Baseline_LogReg": LogisticRegression(max_iter=500, random_state=42, C=1.0),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42,
                                            n_jobs=-1, class_weight="balanced"),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                              use_label_encoder=False, eval_metric="mlogloss",
                              verbosity=0),
}


class WeatherClassifier:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.le = LabelEncoder()
        self.results = {}

    def prepare_Xy(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = self.le.fit_transform(df[target_col])
        return X.values, y

    def train_all(self, X_train, y_train, X_test, y_test) -> pd.DataFrame:
        """Train tất cả models, trả về bảng kết quả."""
        rows = []
        for name, base_model in MODELS.items():
            print(f"\n[supervised] Training {name}...")
            t0 = time.time()

            # Wrap trong pipeline để scale (trừ XGBoost/RF không cần thiết nhưng nhất quán)
            if name in ["Baseline_LogReg"]:
                model = Pipeline([("scaler", StandardScaler()), ("clf", base_model)])
            else:
                model = base_model

            model.fit(X_train, y_train)
            t_train = time.time() - t0

            y_pred = model.predict(X_test)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)

            # ROC-AUC (one-vs-rest)
            try:
                y_prob = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except Exception:
                auc = None

            rows.append({
                "Model": name,
                "F1_macro": round(f1_macro, 4),
                "F1_weighted": round(f1_weighted, 4),
                "Accuracy": round(acc, 4),
                "ROC_AUC": round(auc, 4) if auc else "N/A",
                "Train_time_s": round(t_train, 2),
            })
            self.results[name] = {"model": model, "y_pred": y_pred}
            print(f"  F1-macro={f1_macro:.4f}, Acc={acc:.4f}, Time={t_train:.1f}s")

        return pd.DataFrame(rows).sort_values("F1_macro", ascending=False)

    def get_confusion_matrix(self, model_name: str, y_test) -> np.ndarray:
        y_pred = self.results[model_name]["y_pred"]
        return confusion_matrix(y_test, y_pred)

    def get_classification_report(self, model_name: str, y_test) -> str:
        y_pred = self.results[model_name]["y_pred"]
        return classification_report(y_test, y_pred,
                                     target_names=self.le.classes_)

    def save_model(self, model_name: str, output_dir: str = "outputs/models"):
        os.makedirs(output_dir, exist_ok=True)
        model = self.results[model_name]["model"]
        path = os.path.join(output_dir, f"{model_name}.joblib")
        joblib.dump(model, path)
        print(f"[supervised] Model saved to {path}")
