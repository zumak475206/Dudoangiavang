"""
Kiểm thử tự động cho module preprocessor.py:
1. Kiểm tra thuật toán kiểm định ADF Test hoạt động chính xác (giá gốc không dừng, return dừng).
2. Kiểm tra tính toán Return đúng công thức toán học.
3. Kiểm tra tính toàn vẹn của Time Series Split (Không rò rỉ dữ liệu, thứ tự thời gian nghiêm ngặt).
4. Kiểm tra TimeSeriesScaler tuân thủ nguyên tắc fit chỉ trên Train.
"""

import os
import unittest
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from src.preprocessor import perform_adf_test, compute_returns, time_series_split, TimeSeriesScaler

class TestPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gold_market_data.csv")
        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {cls.data_path}")
        cls.df = pd.read_csv(cls.data_path, index_col="Date", parse_dates=True)
        cls.df = compute_returns(cls.df, price_col="gold_close")

    def test_adf_raw_price_is_non_stationary(self):
        """Giá vàng gốc phải là chuỗi KHÔNG DỪNG (p-value > 0.05)."""
        res = perform_adf_test(self.df["gold_close"], "gold_close")
        self.assertFalse(res["is_stationary"], "Giá vàng gốc không được là chuỗi dừng!")
        self.assertGreater(res["p_value"], 0.05, "p-value của giá vàng gốc phải > 0.05.")

    def test_adf_return_is_stationary(self):
        """Tỷ suất sinh lời Vàng (Return) bắt buộc phải là CHUỖI DỪNG (p-value < 0.01)."""
        res = perform_adf_test(self.df["gold_return"], "gold_return")
        self.assertTrue(res["is_stationary"], "Chuỗi tỷ suất sinh lời phải đạt tính dừng!")
        self.assertLess(res["p_value"], 0.01, "p-value của Return phải < 0.01.")
        self.assertLess(res["adf_statistic"], res["critical_values"]["1%"], "ADF stat phải nhỏ hơn giá trị tới hạn 1%.")

    def test_compute_returns_math(self):
        """Kiểm tra độ chính xác của công thức tính Return."""
        p0 = self.df["gold_close"].iloc[0]
        p1 = self.df["gold_close"].iloc[1]
        expected_ret = (p1 - p0) / p0
        actual_ret = self.df["gold_return"].iloc[1]
        self.assertAlmostEqual(expected_ret, actual_ret, places=6, msg="Công thức tính return bị sai lệch!")

    def test_time_series_split_strict_chronological(self):
        """Kiểm tra chia tập thời gian: Không rò rỉ dữ liệu, không giao thoa ngày."""
        train, val, test = time_series_split(self.df, 0.70, 0.15, 0.15)
        
        # 1. Tổng số dòng phải khớp
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))
        
        # 2. Thứ tự thời gian nghiêm ngặt (Train < Val < Test)
        self.assertLess(train.index.max(), val.index.min(), "Ngày lớn nhất của Train phải nhỏ hơn ngày nhỏ nhất của Val!")
        self.assertLess(val.index.max(), test.index.min(), "Ngày lớn nhất của Val phải nhỏ hơn ngày nhỏ nhất của Test!")
        
        # 3. Không có ngày nào bị trùng lặp giữa các tập
        overlap_train_val = set(train.index).intersection(set(val.index))
        overlap_val_test = set(val.index).intersection(set(test.index))
        self.assertEqual(len(overlap_train_val), 0, "Trùng lặp ngày giữa Train và Val!")
        self.assertEqual(len(overlap_val_test), 0, "Trùng lặp ngày giữa Val và Test!")

    def test_time_series_scaler_leak_free(self):
        """Kiểm tra Scaler chỉ fit trên Train, không bị rò rỉ thông tin Val/Test."""
        train, val, test = time_series_split(self.df, 0.70, 0.15, 0.15)
        feature_cols = ["gold_close", "silver_close"]
        
        scaler = TimeSeriesScaler(scaler_type="standard")
        train_scaled = scaler.fit_transform(train, feature_cols)
        val_scaled = scaler.transform(val)
        test_scaled = scaler.transform(test)
        
        # Tập Train sau scale phải có mean ~ 0 và std ~ 1
        np.testing.assert_almost_equal(train_scaled[feature_cols].mean().values, [0.0, 0.0], decimal=5)
        np.testing.assert_almost_equal(train_scaled[feature_cols].std(ddof=0).values, [1.0, 1.0], decimal=5)
        
        # Tập Val và Test khi dùng tham số của Train thì mean không nhất thiết bằng 0 (chứng minh không bị fit đè)
        self.assertNotEqual(round(float(test_scaled["gold_close"].mean()), 2), 0.0)

if __name__ == "__main__":
    unittest.main()
