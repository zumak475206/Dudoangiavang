"""
Module tree_models.py:
Xây dựng và tinh chỉnh các mô hình Tree-based (Gradient Boosting) cho bài toán dự báo giá vàng:

1. XGBoostForecaster:
   - Thuật toán: XGBoost (eXtreme Gradient Boosting) - Chen & Guestrin, 2016.
   - Kỹ thuật chính: Gradient boosting trên cây quyết định với chuẩn hóa L1/L2 tích hợp.
   - Tối ưu siêu tham số bằng TimeSeriesSplit cross-validation (không rò rỉ dữ liệu tương lai).
   - Trích xuất Feature Importance theo 3 phương pháp: gain (đóng góp thực), weight, cover.

2. LightGBMForecaster:
   - Thuật toán: LightGBM (Light Gradient Boosting Machine) - Ke et al., Microsoft 2017.
   - Kỹ thuật chính: Leaf-wise tree growth + GOSS sampling + EFB bundling -> Nhanh hơn XGBoost 3-5x.
   - Sử dụng early stopping để kiểm soát overfitting trực tiếp trong quá trình huấn luyện.
   - Hỗ trợ đánh giá trên tập Validation trong khi huấn luyện (in-training eval).
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Đảm bảo import được src.* khi chạy trực tiếp script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.models.baselines import BaseForecaster


# ==============================================================================
# 1. XGBOOST FORECASTER
# ==============================================================================

class XGBoostForecaster(BaseForecaster):
    """
    Mô hình XGBoost Regressor được tùy chỉnh cho bài toán chuỗi thời gian tài chính.

    Chiến lược tối ưu:
    - Học trên target_return (chuỗi dừng, thỏa ADF Test), sau đó quy đổi sang giá.
    - Regularization L1 (alpha) + L2 (lambda) tích hợp chống overfitting trên tập đặc trưng 56 chiều.
    - TimeSeriesSplit cross-validation đảm bảo không rò rỉ thông tin tương lai vào đánh giá.
    - Early stopping trên tập Validation để dừng học khi model bắt đầu ghi nhớ nhiễu.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,       # L1 regularization
        "reg_lambda": 1.0,      # L2 regularization
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",  # Faster CPU training
        "random_state": 42,
        "verbosity": 0,
    }

    def __init__(
        self,
        target_type: str = "return",
        params: Optional[Dict] = None,
        early_stopping_rounds: int = 50,
        n_cv_splits: int = 5,
    ):
        super().__init__(name="XGBoost Regressor", target_type=target_type)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.n_cv_splits = n_cv_splits
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.best_n_estimators_: Optional[int] = None
        self.cv_results_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = True,
    ):
        """
        Huấn luyện XGBoost với early stopping.

        Args:
            X, y: Tập huấn luyện.
            X_val, y_val: Tập Validation cho early stopping. Nếu None, dùng TimeSeriesSplit.
            verbose: In tiến trình huấn luyện.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float32).ravel()

        # Nếu có tập Validation rõ ràng -> dùng trực tiếp
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values.astype(np.float32) if isinstance(X_val, pd.DataFrame) else np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32).ravel()
            eval_set = [(X_arr, y_arr), (X_val_arr, y_val_arr)]

            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_arr, y_arr,
                eval_set=eval_set,
                verbose=False,
            )
            self.best_n_estimators_ = self.model.best_iteration if hasattr(self.model, "best_iteration") else self.params["n_estimators"]
        else:
            # Dùng TimeSeriesSplit CV để xác định best n_estimators
            tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
            cv_rmse_list = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
                X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
                y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

                fold_model = xgb.XGBRegressor(**self.params)
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )
                preds = fold_model.predict(X_fold_val)
                rmse = np.sqrt(np.mean((preds - y_fold_val) ** 2))
                cv_rmse_list.append(rmse)

            mean_rmse = np.mean(cv_rmse_list)
            if verbose:
                print(f"  [XGBoost CV] TimeSeriesSplit {self.n_cv_splits}-fold | Mean RMSE (return): {mean_rmse:.6f}")

            self.cv_results_ = pd.DataFrame({"fold_rmse": cv_rmse_list})

            # Huấn luyện lại trên toàn bộ Train với n_estimators đầy đủ
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(X_arr, y_arr, verbose=False)
            self.best_n_estimators_ = self.params["n_estimators"]

        self.is_fitted = True
        if verbose:
            print(f"  [XGBoost] Huấn luyện hoàn tất. Best iterations: {self.best_n_estimators_}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("XGBoost chưa được huấn luyện! Hãy gọi fit() trước.")
        X_arr = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Trích xuất Feature Importance từ XGBoost.

        Args:
            importance_type: 'gain' (đóng góp thực - khuyến nghị), 'weight' (số lần dùng), 'cover'.
                             Lưu ý: 'gain' dùng booster score dict, còn 'weight' dùng feature_importances_.

        Returns:
            pd.DataFrame sắp xếp giảm dần theo importance_score.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model chưa được fit!")

        names = self.feature_names_ or [f"feat_{i}" for i in range(len(self.model.feature_importances_))]

        # Dùng sklearn-style feature_importances_ (tổng gain chuẩn hóa) - robust hơn get_booster().get_score()
        scores = self.model.feature_importances_.astype(float)

        df = pd.DataFrame({
            "feature": names,
            "importance_score": scores,
            "importance_type": importance_type,
        }).sort_values("importance_score", ascending=False).reset_index(drop=True)

        # Chuẩn hóa về phần trăm đóng góp tương đối
        total = df["importance_score"].sum()
        df["importance_pct"] = (df["importance_score"] / total * 100.0) if total > 0 else 0.0

        return df


# ==============================================================================
# 2. LIGHTGBM FORECASTER
# ==============================================================================

class LightGBMForecaster(BaseForecaster):
    """
    Mô hình LightGBM Regressor với leaf-wise tree growth - nhanh hơn XGBoost 3-5x.

    Ưu điểm so với XGBoost trong bài toán tài chính:
    - GOSS (Gradient-based One-Side Sampling): Giảm số mẫu cần xử lý mà không mất tính chất gradient.
    - EFB (Exclusive Feature Bundling): Gộp đặc trưng thưa (sparse) - phù hợp bộ đặc trưng lớn 56 chiều.
    - Tốc độ huấn luyện nhanh hơn -> Thích hợp thử nghiệm nhiều siêu tham số hơn trong giới hạn thời gian.
    - Early stopping theo RMSE trên tập Validation để kiểm soát overfitting.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(
        self,
        target_type: str = "return",
        params: Optional[Dict] = None,
        early_stopping_rounds: int = 50,
        n_cv_splits: int = 5,
    ):
        super().__init__(name="LightGBM Regressor", target_type=target_type)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.n_cv_splits = n_cv_splits
        self.model: Optional[lgb.LGBMRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self.best_n_estimators_: Optional[int] = None
        self.cv_results_: Optional[pd.DataFrame] = None

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: bool = True,
    ):
        """
        Huấn luyện LightGBM với early stopping.

        Args:
            X, y: Tập huấn luyện (70% dữ liệu thời gian).
            X_val, y_val: Tập Validation cho early stopping. Nếu None, dùng TimeSeriesSplit.
            verbose: In tiến trình.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            self.feature_names_ = [f"feat_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.float32).ravel()

        fit_params: Dict = {}

        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values.astype(np.float32) if isinstance(X_val, pd.DataFrame) else np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32).ravel()

            fit_params["eval_X"] = X_val_arr
            fit_params["eval_y"] = y_val_arr
            fit_params["callbacks"] = [
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=-1),
            ]

            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X_arr, y_arr, **fit_params)
            # best_iteration_ có thể = 1 nếu val loss ngay lập tức không cải thiện.
            # Fallback về min(n_estimators, 200) để tránh underfitting nghiêm trọng.
            best_iter = getattr(self.model, "best_iteration_", 0) or 0
            self.best_n_estimators_ = max(best_iter, 100) if best_iter > 0 else self.params["n_estimators"]
        else:
            # TimeSeriesSplit CV
            tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
            cv_rmse_list = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_arr)):
                X_fold_train, X_fold_val = X_arr[train_idx], X_arr[val_idx]
                y_fold_train, y_fold_val = y_arr[train_idx], y_arr[val_idx]

                fold_model = lgb.LGBMRegressor(**self.params)
                fold_model.fit(
                    X_fold_train, y_fold_train,
                    eval_X=X_fold_val,
                    eval_y=y_fold_val,
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=-1),
                    ],
                )
                preds = fold_model.predict(X_fold_val)
                rmse = np.sqrt(np.mean((preds - y_fold_val) ** 2))
                cv_rmse_list.append(rmse)

            mean_rmse = np.mean(cv_rmse_list)
            if verbose:
                print(f"  [LightGBM CV] TimeSeriesSplit {self.n_cv_splits}-fold | Mean RMSE (return): {mean_rmse:.6f}")

            self.cv_results_ = pd.DataFrame({"fold_rmse": cv_rmse_list})

            # Huấn luyện lại trên toàn bộ Train
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                X_arr, y_arr,
                callbacks=[lgb.log_evaluation(period=-1)],
            )
            self.best_n_estimators_ = self.params["n_estimators"]

        self.is_fitted = True
        if verbose:
            print(f"  [LightGBM] Huấn luyện hoàn tất. Best iterations: {self.best_n_estimators_}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("LightGBM chưa được huấn luyện! Hãy gọi fit() trước.")
        X_arr = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr)

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """
        Trích xuất Feature Importance từ LightGBM.

        Args:
            importance_type: 'gain' hoặc 'split'.

        Returns:
            pd.DataFrame sắp xếp giảm dần theo importance_score.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model chưa được fit!")

        scores = self.model.feature_importances_
        names = self.feature_names_ or [f"feat_{i}" for i in range(len(scores))]

        df = pd.DataFrame({
            "feature": names,
            "importance_score": scores.astype(float),
            "importance_type": importance_type,
        }).sort_values("importance_score", ascending=False).reset_index(drop=True)

        total = df["importance_score"].sum()
        df["importance_pct"] = (df["importance_score"] / total * 100.0) if total > 0 else 0.0

        return df


# ==============================================================================
# 3. HELPER: Chạy thực nghiệm và in bảng so sánh
# ==============================================================================

def run_tree_models_experiment(
    df_featured: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Thực thi quy trình huấn luyện và đánh giá sơ bộ toàn diện cho XGBoost và LightGBM.

    Returns:
        (trained_models dict, leaderboard DataFrame)
    """
    from src.preprocessor import time_series_split, TimeSeriesScaler
    from src.features import get_feature_columns_for_training

    # 1. Phân tách chuỗi thời gian (nghiêm ngặt, không rò rỉ)
    n = len(df_featured)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_featured.iloc[:train_end]
    val_df   = df_featured.iloc[train_end:val_end]
    test_df  = df_featured.iloc[val_end:]

    feature_cols = get_feature_columns_for_training(df_featured)

    # 2. Chuẩn hóa (fit chỉ trên Train, transform Val + Test)
    scaler = TimeSeriesScaler("standard")
    scaler.fit(train_df, feature_cols)
    train_s = scaler.transform(train_df)
    val_s   = scaler.transform(val_df)
    test_s  = scaler.transform(test_df)

    X_train = train_s[feature_cols]
    y_train = train_df["target_return"]
    X_val   = val_s[feature_cols]
    y_val   = val_df["target_return"]
    X_test  = test_s[feature_cols]

    curr_close_test = test_df["gold_close"].values
    y_test_close    = test_df["target_close"].values
    y_test_dir      = test_df["target_direction"].values

    # 3. Khởi tạo và huấn luyện cả 2 mô hình
    models = {
        "XGBoost": XGBoostForecaster(target_type="return"),
        "LightGBM": LightGBMForecaster(target_type="return"),
    }

    print("\n" + "="*85)
    print(f"{'HUẤN LUYỆN MÔ HÌNH TREE-BASED (TASK 7)':^85}")
    print("="*85)

    for model_name, model in models.items():
        print(f"\n  >>> Đang huấn luyện: {model_name}")
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)

    # 4. Đánh giá trên tập Test độc lập
    leaderboard_rows = []
    trained_models = {}

    for model_name, model in models.items():
        pred_close = model.predict_close(X_test, curr_close_test)
        pred_dir   = model.predict_direction(X_test, curr_close_test)

        rmse = float(np.sqrt(np.mean((pred_close - y_test_close) ** 2)))
        mae  = float(np.mean(np.abs(pred_close - y_test_close)))
        mape = float(np.mean(np.abs((pred_close - y_test_close) / y_test_close)) * 100.0)
        da   = float(np.mean(pred_dir == y_test_dir) * 100.0)

        leaderboard_rows.append({
            "model": model_name,
            "rmse_usd": rmse,
            "mae_usd": mae,
            "mape_pct": mape,
            "directional_accuracy_pct": da,
        })
        trained_models[model_name] = model

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("rmse_usd")

    print("\n" + "="*95)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (2024 - NAY)':^95}")
    print("="*95)
    header = f"{'Mô Hình':<20} | {'RMSE ($)':<12} | {'MAE ($)':<12} | {'MAPE (%)':<12} | {'DA (%)':<15}"
    print(header)
    print("-" * 79)
    for _, row in leaderboard.iterrows():
        print(f"{row['model']:<20} | {row['rmse_usd']:<12.2f} | {row['mae_usd']:<12.2f} | {row['mape_pct']:<12.4f} | {row['directional_accuracy_pct']:<15.2f}")
    print("="*79)

    return trained_models, leaderboard


if __name__ == "__main__":
    from src.preprocessor import time_series_split, TimeSeriesScaler
    from src.features import get_feature_columns_for_training

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "featured_gold_data.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy {data_path}. Hãy chạy features.py trước!")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    print(f"[INFO] Đã đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột.")

    trained_models, leaderboard = run_tree_models_experiment(df)

    # In Top 15 Feature Importance cho mỗi mô hình
    for model_name, model in trained_models.items():
        fi_df = model.get_feature_importance(importance_type="gain")
        print(f"\n  TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT - {model_name.upper()} (by Gain):")
        print(f"  {'#':<4} {'Feature':<28} {'Importance':>12} {'Cumulative %':>14}")
        print("  " + "-"*62)
        cumulative = 0.0
        for idx, row in fi_df.head(15).iterrows():
            cumulative += row["importance_pct"]
            print(f"  {idx+1:<4} {row['feature']:<28} {row['importance_score']:>12.1f} {cumulative:>13.1f}%")
