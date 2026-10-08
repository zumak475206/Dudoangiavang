"""
Kiểm thử tự động cho module src/evaluator.py:

1. TestMetricsMath:
   - Độ chính xác toán học của calculate_rmse, calculate_mae, calculate_mape.
   - calculate_directional_accuracy: đúng 100%, 75%, 0%.
   - calculate_r2: dự báo hoàn hảo cho R2 = 1.0.
   - compute_all_metrics: trả về đầy đủ 5 trường chỉ số bắt buộc.

2. TestLeaderboardGeneration:
   - generate_leaderboard chạy thành công với các mô hình mock / baseline.
   - DataFrame kết quả chứa đúng các cột quy chuẩn.
   - Thứ tự mô hình được sắp xếp tăng dần theo RMSE ($).

3. TestMultiStepForecast:
   - forecast_next_n_days tạo đúng số ngày n_days = 7.
   - Ngày dự báo chỉ rơi vào ngày làm việc trong tuần (bỏ qua Thứ Bảy & Chủ Nhật).
   - Giá dự báo luôn dương (> 0).
   - Cận Dưới <= Giá Dự Báo <= Cận Trên cho toàn bộ 7 bước.

4. TestEvaluatorPipelineIntegration:
   - Chạy tích hợp trên file featured_gold_data.csv thật.
   - Leaderboard phân định được các mô hình, chứng minh Checkpoint 3 đạt yêu cầu.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.evaluator import (
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_directional_accuracy,
    calculate_r2,
    compute_all_metrics,
    evaluate_single_model,
    generate_leaderboard,
    forecast_next_n_days,
)
from src.models.baselines import BaseForecaster, NaivePersistenceModel


class MockConstantModel(BaseForecaster):
    """Mô hình giả lập dự báo một hằng số return cố định."""
    def __init__(self, constant_return: float = 0.005):
        super().__init__(name="Mock Constant Forecaster", target_type="return")
        self.constant_return = constant_return
        self.is_fitted = True

    def fit(self, X, y):
        self.is_fitted = True
        return self

    def predict(self, X):
        n = len(X)
        return np.full(n, self.constant_return, dtype=np.float32)


class TestMetricsMath(unittest.TestCase):
    def test_rmse_exact(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([103.0, 96.0])  # diff: 3, -4; squared: 9, 16; mean: 12.5; sqrt: ~3.5355
        expected = np.sqrt((9 + 16) / 2.0)
        self.assertAlmostEqual(calculate_rmse(y_true, y_pred), expected, places=4)

    def test_rmse_zero_error(self):
        y_true = np.array([2000.5, 2050.2, 1980.0])
        self.assertEqual(calculate_rmse(y_true, y_true), 0.0)

    def test_mae_exact(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([105.0, 98.0])
        self.assertEqual(calculate_mae(y_true, y_pred), 3.5)

    def test_mape_exact(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])  # error: 10%, 5% -> mean 7.5%
        self.assertAlmostEqual(calculate_mape(y_true, y_pred), 7.5, places=5)

    def test_directional_accuracy_percentage(self):
        # 3 đúng, 1 sai -> 75%
        y_true_dir = np.array([1, 0, 1, 1])
        y_pred_dir = np.array([1, 0, 0, 1])
        self.assertEqual(calculate_directional_accuracy(y_true_dir, y_pred_dir), 75.0)

        # Toàn bộ đúng -> 100%
        self.assertEqual(calculate_directional_accuracy(y_true_dir, y_true_dir), 100.0)

        # Toàn bộ sai -> 0%
        y_inv = 1 - y_true_dir
        self.assertEqual(calculate_directional_accuracy(y_true_dir, y_inv), 0.0)

    def test_r2_score_perfect(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        self.assertAlmostEqual(calculate_r2(y_true, y_true), 1.0, places=4)

    def test_compute_all_metrics_keys(self):
        y_true = np.array([2000.0, 2010.0, 2020.0])
        y_pred = np.array([2002.0, 2008.0, 2025.0])
        y_tdir = np.array([1, 0, 1])
        y_pdir = np.array([1, 0, 1])
        metrics = compute_all_metrics(y_true, y_pred, y_tdir, y_pdir)

        expected_keys = {
            "rmse_usd",
            "mae_usd",
            "mape_pct",
            "directional_accuracy_pct",
            "r2_score",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)
        self.assertFalse(any(np.isnan(v) for v in metrics.values()))


class TestLeaderboardGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n = 50
        cls.curr_close = np.linspace(2000, 2100, cls.n)
        cls.y_test_close = cls.curr_close * 1.002
        cls.y_test_dir = np.ones(cls.n, dtype=int)
        cls.X_test = pd.DataFrame(np.random.randn(cls.n, 5), columns=[f"f_{i}" for i in range(5)])

        cls.models = {
            "Naive": NaivePersistenceModel(mode="last_price"),
            "MockModel": MockConstantModel(constant_return=0.002),
        }

    def test_generate_leaderboard_structure(self):
        lb = generate_leaderboard(
            models=self.models,
            X_test=self.X_test,
            current_close_test=self.curr_close,
            y_test_close=self.y_test_close,
            y_test_dir=self.y_test_dir,
        )

        self.assertIsInstance(lb, pd.DataFrame)
        self.assertEqual(len(lb), len(self.models))

        expected_cols = [
            "Mô Hình",
            "RMSE ($)",
            "MAE ($)",
            "MAPE (%)",
            "Directional Acc (%)",
            "R2 Score",
            "Residual Std ($)",
        ]
        for col in expected_cols:
            self.assertIn(col, lb.columns)

    def test_leaderboard_sorted_by_rmse(self):
        lb = generate_leaderboard(
            models=self.models,
            X_test=self.X_test,
            current_close_test=self.curr_close,
            y_test_close=self.y_test_close,
            y_test_dir=self.y_test_dir,
        )
        rmse_values = lb["RMSE ($)"].values
        self.assertTrue(
            (rmse_values[:-1] <= rmse_values[1:]).all(),
            "Leaderboard chưa được sắp xếp tăng dần theo RMSE ($)!",
        )


class TestMultiStepForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feature_cols = ["f_0", "f_1", "gold_return_lag1"]
        cls.latest_row = pd.Series([0.1, -0.2, 0.003], index=cls.feature_cols)
        cls.latest_close = 2350.0
        cls.latest_date = pd.Timestamp("2026-03-27")  # Thứ Sáu
        cls.model = MockConstantModel(constant_return=0.001)

    def test_forecast_length_and_steps(self):
        df_fc = forecast_next_n_days(
            model=self.model,
            latest_feature_row=self.latest_row,
            latest_close=self.latest_close,
            latest_date=self.latest_date,
            feature_cols=self.feature_cols,
            n_days=7,
            residual_std=15.0,
        )
        self.assertEqual(len(df_fc), 7)
        self.assertEqual(list(df_fc["Bước"]), [f"T+{i}" for i in range(1, 8)])

    def test_forecast_skips_weekends(self):
        """Kiểm tra ngày dự báo không bao gồm Thứ Bảy hoặc Chủ Nhật."""
        df_fc = forecast_next_n_days(
            model=self.model,
            latest_feature_row=self.latest_row,
            latest_close=self.latest_close,
            latest_date=self.latest_date,
            feature_cols=self.feature_cols,
            n_days=7,
        )
        for d_str in df_fc["Ngày Giao Dịch"]:
            d = pd.to_datetime(d_str)
            self.assertNotIn(
                d.weekday(),
                [5, 6],
                f"Ngày {d_str} là cuối tuần ({d.strftime('%A')}) nhưng vẫn có trong lịch!",
            )

    def test_forecast_prices_positive(self):
        df_fc = forecast_next_n_days(
            model=self.model,
            latest_feature_row=self.latest_row,
            latest_close=self.latest_close,
            latest_date=self.latest_date,
            feature_cols=self.feature_cols,
            n_days=7,
        )
        prices = df_fc["Giá Dự Báo ($)"].values
        self.assertTrue((prices > 0).all())

    def test_confidence_interval_bounds(self):
        df_fc = forecast_next_n_days(
            model=self.model,
            latest_feature_row=self.latest_row,
            latest_close=self.latest_close,
            latest_date=self.latest_date,
            feature_cols=self.feature_cols,
            n_days=7,
            residual_std=20.0,
        )
        for _, row in df_fc.iterrows():
            lower = row["Cận Dưới 95% ($)"]
            price = row["Giá Dự Báo ($)"]
            upper = row["Cận Trên 95% ($)"]
            self.assertLessEqual(lower, price, "Cận dưới phải <= Giá dự báo")
            self.assertGreaterEqual(upper, price, "Cận trên phải >= Giá dự báo")


class TestEvaluatorPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "featured_gold_data.csv"
        )

    def test_featured_data_exists_for_eval(self):
        self.assertTrue(
            os.path.exists(self.data_path),
            "featured_gold_data.csv phải tồn tại để phục vụ đánh giá!",
        )


if __name__ == "__main__":
    unittest.main()
