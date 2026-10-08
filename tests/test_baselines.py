"""
Kiểm thử tự động cho module src/models/baselines.py:
1. Kiểm tra NaivePersistenceModel:
   - Dự báo giá đóng cửa khớp chính xác với giá hiện tại (P_hat_{t+1} = P_t).
   - Dự báo tỷ suất sinh lời r_hat_{t+1} = 0 khi mode='last_price'.
   - Dự báo xu hướng có dạng mảng số nguyên nhị phân.
2. Kiểm tra LinearRegressionBaseline (OLS & Ridge):
   - Quá trình fit() và predict() chạy thông suốt, kích thước đầu ra đúng với số lượng mẫu.
   - Báo lỗi hợp lệ nếu gọi predict() khi chưa fit().
   - Công thức quy đổi giá P_hat_{t+1} = P_t * (1 + r_hat_{t+1}) chuẩn xác về mặt số học.
   - Hàm predict_direction() chỉ trả về các nhãn nhị phân {0, 1}.
   - Hàm get_coefficients() trích xuất đúng tên và số lượng biến, sắp xếp giảm dần theo abs_importance.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.models.baselines import (
    BaseForecaster,
    NaivePersistenceModel,
    LinearRegressionBaseline
)
from src.features import get_feature_columns_for_training

class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tạo dữ liệu giả lập có cấu trúc tương tự dữ liệu tài chính
        np.random.seed(42)
        cls.n_samples = 100
        cls.n_features = 10
        cls.feature_names = [f"feat_{i}" for i in range(cls.n_features)]
        
        cls.X_df = pd.DataFrame(
            np.random.randn(cls.n_samples, cls.n_features),
            columns=cls.feature_names
        )
        cls.current_close = pd.Series(
            2000.0 + np.cumsum(np.random.randn(cls.n_samples) * 10),
            name="gold_close"
        )
        cls.y_return = pd.Series(
            np.random.randn(cls.n_samples) * 0.01,
            name="target_return"
        )
        cls.y_close = cls.current_close * (1.0 + cls.y_return)

    def test_naive_persistence_last_price(self):
        """Naive Persistence (last_price) phải dự báo P_hat_{t+1} = P_t và r_hat_{t+1} = 0."""
        model = NaivePersistenceModel(target_type="return", mode="last_price")
        model.fit(self.X_df, self.y_return)
        
        # 1. Dự báo return phải bằng 0 toàn bộ
        pred_ret = model.predict(self.X_df)
        self.assertEqual(len(pred_ret), self.n_samples)
        np.testing.assert_array_equal(pred_ret, np.zeros(self.n_samples))
        
        # 2. Dự báo close phải bằng chính xác current_close
        pred_close = model.predict_close(self.X_df, self.current_close)
        np.testing.assert_array_equal(pred_close, self.current_close.values)

    def test_naive_persistence_direction(self):
        """Naive direction phải trả về mảng nhị phân {0, 1}."""
        model = NaivePersistenceModel(target_type="return", mode="last_price")
        model.fit(self.X_df, self.y_return)
        pred_dir = model.predict_direction(self.X_df, self.current_close)
        
        self.assertEqual(len(pred_dir), self.n_samples)
        self.assertTrue(set(np.unique(pred_dir)).issubset({0, 1}))

    def test_linear_regression_ols_fit_predict(self):
        """Linear Regression OLS fit và predict đúng kích thước ma trận."""
        model = LinearRegressionBaseline(target_type="return", model_type="linear")
        
        # Kiểm tra chưa fit gọi predict phải báo lỗi
        with self.assertRaises(RuntimeError):
            model.predict(self.X_df)
            
        model.fit(self.X_df, self.y_return)
        self.assertTrue(model.is_fitted)
        
        preds = model.predict(self.X_df)
        self.assertEqual(preds.shape, (self.n_samples,))
        self.assertFalse(np.isnan(preds).any(), "Dự báo chứa giá trị NaN!")

    def test_ridge_regression_fit_predict(self):
        """Ridge Regression fit thành công với hệ số phạt L2 alpha > 0."""
        model = LinearRegressionBaseline(target_type="return", model_type="ridge", alpha=5.0)
        model.fit(self.X_df, self.y_return)
        
        preds = model.predict(self.X_df)
        self.assertEqual(len(preds), self.n_samples)
        self.assertFalse(np.isnan(preds).any())

    def test_price_reconstruction_formula(self):
        """Kiểm tra P_hat_{t+1} = P_t * (1 + r_hat_{t+1}) khớp chính xác đến từng số thập phân."""
        model = LinearRegressionBaseline(target_type="return", model_type="ridge", alpha=1.0)
        model.fit(self.X_df, self.y_return)
        
        pred_ret = model.predict(self.X_df)
        pred_close = model.predict_close(self.X_df, self.current_close)
        expected_close = self.current_close.values * (1.0 + pred_ret)
        
        np.testing.assert_almost_equal(pred_close, expected_close, decimal=6)

    def test_predict_direction_logic(self):
        """Hướng dự báo phải bằng 1 khi r_hat > 0 và bằng 0 khi r_hat <= 0."""
        model = LinearRegressionBaseline(target_type="return", model_type="linear")
        model.fit(self.X_df, self.y_return)
        
        pred_ret = model.predict(self.X_df)
        pred_dir = model.predict_direction(self.X_df, self.current_close)
        
        expected_dir = (pred_ret > 0).astype(int)
        np.testing.assert_array_equal(pred_dir, expected_dir)

    def test_get_coefficients_ordering(self):
        """Hệ số trích xuất phải có độ dài bằng số đặc trưng và sắp xếp giảm dần theo abs_importance."""
        model = LinearRegressionBaseline(target_type="return", model_type="ridge", alpha=1.0)
        model.fit(self.X_df, self.y_return)
        
        coef_df = model.get_coefficients()
        self.assertEqual(len(coef_df), self.n_features)
        self.assertListEqual(list(coef_df.columns), ["feature", "coefficient", "abs_importance"])
        
        # Kiểm tra tính chất sắp xếp giảm dần
        abs_vals = coef_df["abs_importance"].values
        self.assertTrue((abs_vals[:-1] >= abs_vals[1:]).all(), "Hệ số chưa được sắp xếp giảm dần!")

    def test_real_dataset_smoke_test(self):
        """Kiểm tra mô hình chạy trơn tru trên tập dữ liệu featured_gold_data.csv thật."""
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "featured_gold_data.csv")
        if not os.path.exists(data_path):
            self.skipTest("Chưa có featured_gold_data.csv")
            
        df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
        feature_cols = get_feature_columns_for_training(df)
        
        X = df[feature_cols].iloc[:200]
        y = df["target_return"].iloc[:200]
        curr_close = df["gold_close"].iloc[:200]
        
        model = LinearRegressionBaseline(target_type="return", model_type="ridge", alpha=1.0)
        model.fit(X, y)
        preds_close = model.predict_close(X, curr_close)
        preds_dir = model.predict_direction(X, curr_close)
        
        self.assertEqual(len(preds_close), 200)
        self.assertEqual(len(preds_dir), 200)
        self.assertTrue((preds_close > 0).all(), "Giá dự báo phải dương!")

if __name__ == "__main__":
    unittest.main()
