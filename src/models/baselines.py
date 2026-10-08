"""
Module baselines.py:
Chịu trách nhiệm cung cấp các mô hình nền tảng (Baseline Benchmarks) cho bài toán dự báo giá vàng:
1. BaseForecaster: Abstract class chuẩn hóa giao diện cho toàn bộ mô hình trong dự án.
2. NaivePersistenceModel (Random Walk Benchmark - Martingale Property):
   - Chuẩn mực bắt buộc trong kinh tế lượng lượng tử & tài chính (Fama, 1970).
   - Dự báo giá đóng cửa ngày mai bằng giá ngày hôm nay: P_hat_{t+1} = P_t.
3. LinearRegressionBaseline (Mô hình Hồi quy Tuyến tính OLS & Ridge):
   - Huấn luyện trên không gian 56 đặc trưng kinh tế vĩ mô và chỉ báo kỹ thuật.
   - Hỗ trợ giải thuật hồi quy Ridge (L2 Regularization) để kiểm soát hiện tượng đa cộng tuyến (Multicollinearity).
   - Trích xuất hệ số hồi quy (Coefficients) giúp giải thích mức độ ảnh hưởng của từng nhân tố vĩ mô.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
# Đảm bảo import được các module từ src khi chạy trực tiếp script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

class BaseForecaster(ABC):
    """
    Lớp cơ sở trừu tượng chuẩn hóa giao diện cho toàn bộ mô hình dự báo
    (Baseline, XGBoost, LightGBM, PyTorch LSTM).
    """
    def __init__(self, name: str, target_type: str = "return"):
        """
        Args:
            name: Tên mô hình (phục vụ hiển thị trên bảng so sánh).
            target_type: Loại biến mục tiêu huấn luyện:
                - 'return': Học trên tỷ suất sinh lời r_{t+1} (Dừng - Chuẩn học thuật).
                - 'close': Học trực tiếp trên mức giá đóng cửa Close_{t+1}.
        """
        if target_type not in ["return", "close"]:
            raise ValueError("target_type phải là 'return' hoặc 'close'!")
        self.name = name
        self.target_type = target_type
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Huấn luyện mô hình trên dữ liệu Train."""
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Dự báo giá trị của biến mục tiêu (return hoặc close)."""
        pass

    def predict_close(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        current_close: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        """
        Quy đổi kết quả dự báo ra mức giá đóng cửa tương lai P_hat_{t+1}.
        
        Nếu target_type == 'return':
            P_hat_{t+1} = P_t * (1 + r_hat_{t+1})
        Nếu target_type == 'close':
            P_hat_{t+1} = y_hat
        """
        preds = self.predict(X)
        curr = np.asarray(current_close).ravel()
        
        if self.target_type == "return":
            return curr * (1.0 + preds)
        else:
            return preds

    def predict_direction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        current_close: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Dự báo xu hướng nhị phân: 1 nếu Tăng giá, 0 nếu Giảm/Đi ngang.
        
        Nếu target_type == 'return':
            direction_hat = 1 nếu r_hat_{t+1} > 0, ngược lại 0.
        Nếu target_type == 'close':
            direction_hat = 1 nếu P_hat_{t+1} > P_t, ngược lại 0.
        """
        if self.target_type == "return":
            preds = self.predict(X)
            return (preds > 0).astype(int)
        else:
            if current_close is None:
                raise ValueError("Cần cung cấp current_close khi target_type='close' để xác định xu hướng!")
            preds_close = self.predict(X)
            curr = np.asarray(current_close).ravel()
            return (preds_close > curr).astype(int)


class NaivePersistenceModel(BaseForecaster):
    """
    Mô hình Naive Persistence (Random Walk Benchmark / Không đổi).
    
    Cơ sở kinh tế học:
    Theo giả thuyết thị trường hiệu quả (EMH - Fama 1970) và tính chất Martingale:
        E[Close_{t+1} | Thông tin tại t] = Close_t
    Nghĩa là mức giá tốt nhất để dự đoán cho ngày mai chính là giá của ngày hôm nay.
    
    Trong thực tế phân tích định lượng:
    - Nếu một mô hình AI phức tạp không đạt sai số (RMSE/MAE) tốt hơn Naive Persistence,
      mô hình đó có nguy cơ chỉ đang học nhiễu (noise) của thị trường.
    """
    def __init__(self, target_type: str = "return", mode: str = "last_price"):
        """
        Args:
            target_type: 'return' hoặc 'close'.
            mode: 
                - 'last_price': Dự báo P_hat_{t+1} = P_t (nghĩa là r_hat_{t+1} = 0).
                - 'last_return': Giữ nguyên quán tính return hôm trước (r_hat_{t+1} = r_t).
        """
        super().__init__(name=f"Naive Persistence ({mode})", target_type=target_type)
        if mode not in ["last_price", "last_return"]:
            raise ValueError("mode phải là 'last_price' hoặc 'last_return'!")
        self.mode = mode

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Mô hình Naive không cần tham số tối ưu (non-parametric), fit hoàn thành tức thì."""
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        n_samples = len(X)
        if self.target_type == "return":
            if self.mode == "last_price":
                # Giả định giá ngày mai bằng giá hôm nay -> Tỷ suất sinh lời bằng 0
                return np.zeros(n_samples, dtype=float)
            else:
                # Quán tính: lấy return ngày gần nhất nếu có cột gold_return_lag1 trong X
                if isinstance(X, pd.DataFrame) and "gold_return_lag1" in X.columns:
                    return X["gold_return_lag1"].values
                return np.zeros(n_samples, dtype=float)
        else:
            # target_type == 'close': Cần giá hiện tại
            if isinstance(X, pd.DataFrame) and "gold_close" in X.columns:
                return X["gold_close"].values
            raise ValueError("Khi target_type='close', X cần chứa cột 'gold_close' để trích xuất giá hiện tại!")

    def predict_close(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        current_close: Union[pd.Series, np.ndarray]
    ) -> np.ndarray:
        curr = np.asarray(current_close).ravel()
        if self.mode == "last_price":
            # P_hat_{t+1} = P_t
            return curr.copy()
        else:
            # P_hat_{t+1} = P_t * (1 + r_hat_{t+1})
            preds_ret = self.predict(X)
            return curr * (1.0 + preds_ret)

    def predict_direction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        current_close: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> np.ndarray:
        n_samples = len(X)
        if self.mode == "last_price":
            # Dự báo giá không đổi -> Return = 0 (quy ước direction = 0 vì không tăng)
            return np.zeros(n_samples, dtype=int)
        else:
            preds_ret = self.predict(X)
            return (preds_ret > 0).astype(int)


class LinearRegressionBaseline(BaseForecaster):
    """
    Mô hình Hồi quy Tuyến tính Đa biến (Multivariate Linear & Ridge Regression).
    
    Đặc điểm kỹ thuật:
    - OLS (Ordinary Least Squares): Tối thiểu hóa tổng bình phương phần dư.
    - Ridge Regression (L2 Penalty): Kiểm soát hiện tượng quá khớp (Overfitting) và đa cộng tuyến
      khi tập đặc trưng có 56 biến tương quan mạnh (nhiều đường trung bình SMA, EMA, lợi suất chéo).
    - Cung cấp hàm bóc tách hệ số trọng số (Feature Coefficients) giúp giải thích học thuật.
    """
    def __init__(
        self,
        target_type: str = "return",
        model_type: str = "ridge",
        alpha: float = 1.0,
        fit_intercept: bool = True
    ):
        """
        Args:
            target_type: 'return' hoặc 'close'. Khuyến nghị 'return' để tránh spurious regression.
            model_type: 'linear' (OLS chuẩn) hoặc 'ridge' (L2 regularized).
            alpha: Hệ số phạt L2 của Ridge (mặc định 1.0).
            fit_intercept: Có tính hệ số chặn bias hay không.
        """
        model_name = "Ridge Regression" if model_type.lower() == "ridge" else "Linear Regression (OLS)"
        super().__init__(name=model_name, target_type=target_type)
        
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
        if self.model_type == "ridge":
            self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)
        elif self.model_type in ["linear", "ols"]:
            self.model = LinearRegression(fit_intercept=fit_intercept)
        else:
            raise ValueError("model_type phải là 'linear' hoặc 'ridge'!")
            
        self.feature_names_: Optional[List[str]] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Huấn luyện mô hình hồi quy tuyến tính."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_arr = np.asarray(X)
            
        y_arr = np.asarray(y).ravel()
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện! Hãy gọi fit() trước.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.model.predict(X_arr)

    def get_coefficients(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Trích xuất hệ số hồi quy của từng đặc trưng, sắp xếp theo mức độ ảnh hưởng tuyệt đối.
        
        Returns:
            pd.DataFrame với các cột: ['feature', 'coefficient', 'abs_importance']
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được fit! Không thể trích xuất hệ số.")
            
        names = feature_names if feature_names is not None else self.feature_names_
        if names is None or len(names) != len(self.model.coef_):
            names = [f"feature_{i}" for i in range(len(self.model.coef_))]
            
        coef_df = pd.DataFrame({
            "feature": names,
            "coefficient": self.model.coef_,
            "abs_importance": np.abs(self.model.coef_)
        }).sort_values(by="abs_importance", ascending=False).reset_index(drop=True)
        
        return coef_df


if __name__ == "__main__":
    # Test thực thi và so sánh nhanh các Baseline Models trên tập dữ liệu thực tế
    from src.preprocessor import time_series_split, TimeSeriesScaler
    from src.features import get_feature_columns_for_training

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "featured_gold_data.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy file dữ liệu tại {data_path}. Vui lòng chạy features.py trước!")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    print(f"\n[INFO] Đã đọc dữ liệu đặc trưng: {df.shape[0]} dòng, {df.shape[1]} cột.")

    # 1. Phân tách Train / Val / Test (70% - 15% - 15%)
    train_df, val_df, test_df = time_series_split(df, 0.70, 0.15, 0.15)
    feature_cols = get_feature_columns_for_training(df)
    
    # 2. Chuẩn hóa đặc trưng (Chống rò rỉ dữ liệu)
    scaler = TimeSeriesScaler("standard")
    scaler.fit(train_df, feature_cols)
    train_scaled = scaler.transform(train_df)
    test_scaled = scaler.transform(test_df)

    X_train = train_scaled[feature_cols]
    y_train_ret = train_df["target_return"]
    
    X_test = test_scaled[feature_cols]
    y_test_ret = test_df["target_return"]
    y_test_close = test_df["target_close"]
    curr_close_test = test_df["gold_close"]

    # 3. Khởi tạo 3 mô hình Baseline
    models: List[BaseForecaster] = [
        NaivePersistenceModel(target_type="return", mode="last_price"),
        LinearRegressionBaseline(target_type="return", model_type="linear"),
        LinearRegressionBaseline(target_type="return", model_type="ridge", alpha=10.0),
    ]

    print("\n" + "="*95)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ SƠ BỘ CÁC MÔ HÌNH BASELINE TRÊN TẬP TEST (2024 - NAY)':^95}")
    print("="*95)
    header = f"{'Mô Hình':<32} | {'RMSE ($)':<12} | {'MAE ($)':<12} | {'MAPE (%)':<12} | {'Hit Ratio / DA (%)':<18}"
    print(header)
    print("-" * 95)

    for m in models:
        m.fit(X_train, y_train_ret)
        
        # Dự báo mức giá đóng cửa tương lai P_hat_{t+1}
        pred_close = m.predict_close(X_test, curr_close_test)
        pred_dir = m.predict_direction(X_test, curr_close_test)
        
        # Đánh giá các chỉ số
        actual_close = y_test_close.values
        actual_dir = test_df["target_direction"].values
        
        rmse = np.sqrt(np.mean((pred_close - actual_close) ** 2))
        mae = np.mean(np.abs(pred_close - actual_close))
        mape = np.mean(np.abs((pred_close - actual_close) / actual_close)) * 100.0
        da = np.mean(pred_dir == actual_dir) * 100.0
        
        print(f"{m.name:<32} | {rmse:<12.2f} | {mae:<12.2f} | {mape:<12.2f}% | {da:<18.2f}%")

    print("="*95 + "\n")

    # 4. Trích xuất Top 10 đặc trưng có trọng số lớn nhất trong Ridge Regression
    ridge_model = [m for m in models if isinstance(m, LinearRegressionBaseline) and m.model_type == "ridge"][0]
    coef_df = ridge_model.get_coefficients()
    print("[INFO] TOP 10 ĐẶC TRƯNG ẢNH HƯỞNG LỚN NHẤT ĐẾN TỶ SUẤT SINH LỜI CỦA VÀNG (RIDGE REGRESSION):")
    for idx, row in coef_df.head(10).iterrows():
        sign = "+" if row['coefficient'] > 0 else "-"
        print(f"  {idx+1:>2}. {row['feature']:<25}: {row['coefficient']:>+.5f} ({sign} ảnh hưởng)")
