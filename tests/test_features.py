"""
Kiểm thử tự động cho module features.py:
1. Xác nhận đầy đủ 18 chỉ báo kỹ thuật tài chính thuộc 4 nhóm (Trend, Momentum, Volatility, Volume).
2. Kiểm tra giới hạn giá trị logic của RSI (0 <= RSI <= 100) và Stochastic Oscillator.
3. Kiểm tra tính trật tự của dải Bollinger Bands (bb_high >= bb_mid >= bb_low).
4. Kiểm tra ATR (Average True Range) luôn không âm (>= 0).
5. Kiểm tra toàn bộ pipeline đặc trưng đầy đủ (build_full_feature_pipeline):
   - Đủ > 50 cột đặc trưng.
   - Sạch 100% không còn giá trị NaN sau khi cắt warm-up.
   - Các biến mục tiêu target_return, target_close, target_direction đồng nhất logic.
   - Đặc trưng trễ (Lag 1) khớp chính xác với giá trị ngày hôm trước (Không rò rỉ dữ liệu).
"""

import os
import unittest
import numpy as np
import pandas as pd
from src.features import (
    build_technical_features,
    build_full_feature_pipeline,
    get_technical_feature_names,
    get_feature_columns_for_training
)

class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_gold_data.csv")
        cls.raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gold_market_data.csv")
        cls.input_path = cls.cleaned_path if os.path.exists(cls.cleaned_path) else cls.raw_path
        
        if not os.path.exists(cls.input_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {cls.input_path}")
            
        cls.df_input = pd.read_csv(cls.input_path, index_col="Date", parse_dates=True)
        cls.df_technical = build_technical_features(cls.df_input, drop_warmup=False)
        cls.df_full = build_full_feature_pipeline(cls.df_input, drop_warmup=True)

    def test_all_technical_indicator_columns_present(self):
        """Xác nhận tất cả 18 cột chỉ báo kỹ thuật đều tồn tại trong DataFrame."""
        categories = get_technical_feature_names()
        for cat_name, cols in categories.items():
            for col in cols:
                self.assertIn(col, self.df_technical.columns, f"Thiếu chỉ báo {col} thuộc nhóm {cat_name}!")

    def test_rsi_bounds(self):
        """RSI bắt buộc phải nằm trong miền giá trị [0, 100]."""
        valid_rsi = self.df_technical["rsi_14"].dropna()
        self.assertTrue((valid_rsi >= 0.0).all(), "Tồn tại giá trị RSI âm (< 0)!")
        self.assertTrue((valid_rsi <= 100.0).all(), "Tồn tại giá trị RSI vượt quá 100!")

    def test_bollinger_bands_ordering(self):
        """Dải Bollinger phải thỏa mãn: bb_high >= bb_mid >= bb_low trên toàn bộ chuỗi."""
        valid_bb = self.df_technical[["bb_high", "bb_mid", "bb_low"]].dropna()
        self.assertTrue((valid_bb["bb_high"] >= valid_bb["bb_mid"]).all(), "Dải trên Bollinger nhỏ hơn dải giữa!")
        self.assertTrue((valid_bb["bb_mid"] >= valid_bb["bb_low"]).all(), "Dải giữa Bollinger nhỏ hơn dải dưới!")
        self.assertTrue((self.df_technical["bb_width"].dropna() >= 0.0).all(), "Độ rộng dải Bollinger (Bandwidth) bị âm!")

    def test_stochastic_oscillator_bounds(self):
        """Stochastic %K và %D phải nằm trong đoạn [0, 100]."""
        stoch_k = self.df_technical["stoch_k"].dropna()
        stoch_d = self.df_technical["stoch_d"].dropna()
        self.assertTrue((stoch_k >= 0.0).all() and (stoch_k <= 100.0).all(), "%K nằm ngoài khoảng [0, 100]!")
        self.assertTrue((stoch_d >= 0.0).all() and (stoch_d <= 100.0).all(), "%D nằm ngoài khoảng [0, 100]!")

    def test_atr_non_negative(self):
        """ATR đo độ biến động giá thực tế nên phải luôn không âm (>= 0) và trung bình dương."""
        valid_atr = self.df_technical["atr_14"].dropna()
        self.assertTrue((valid_atr >= 0.0).all(), "Tồn tại giá trị ATR bị âm (< 0)!")
        self.assertGreater(valid_atr.mean(), 0.0, "Giá trị trung bình ATR phải lớn hơn 0!")

    def test_full_pipeline_shape_and_no_nans(self):
        """Tập dữ liệu hoàn chỉnh phải sạch 100% (0 NaN) và có > 25 đặc trưng theo chuẩn Checkpoint 2."""
        self.assertEqual(self.df_full.isna().sum().sum(), 0, "Tập dữ liệu sau pipeline vẫn còn giá trị NaN!")
        self.assertGreater(self.df_full.shape[1], 25, f"Số lượng cột quá ít: {self.df_full.shape[1]} (Kỳ vọng > 25 cột)")
        self.assertGreater(len(self.df_full), 2000, "Số lượng dòng sau cắt warm-up quá ít!")

    def test_target_variables_consistency(self):
        """Kiểm tra tính nhất quán logic giữa 3 biến mục tiêu."""
        targets = ["target_return", "target_close", "target_direction"]
        for t in targets:
            self.assertIn(t, self.df_full.columns, f"Thiếu biến mục tiêu: {t}")
            
        # 1. target_return phải khớp với (target_close - gold_close) / gold_close
        expected_ret = (self.df_full["target_close"] - self.df_full["gold_close"]) / self.df_full["gold_close"]
        np.testing.assert_almost_equal(self.df_full["target_return"].values, expected_ret.values, decimal=5)
        
        # 2. target_direction phải là nhị phân {0, 1}
        self.assertTrue(set(self.df_full["target_direction"].unique()).issubset({0, 1}))
        
        # 3. target_direction == 1 khi target_return > 0
        expected_dir = (self.df_full["target_return"] > 0).astype(int)
        self.assertTrue((self.df_full["target_direction"] == expected_dir).all(), "target_direction không khớp với target_return > 0!")

    def test_lag_features_strictly_past(self):
        """Kiểm tra đặc trưng trễ Lag 1 của ngày hôm nay khớp chính xác với Return của ngày hôm qua."""
        # Lấy dòng thứ 10 và dòng thứ 9
        row_today = self.df_full.iloc[10]
        row_yesterday = self.df_full.iloc[9]
        self.assertAlmostEqual(row_today["gold_return_lag1"], row_yesterday["gold_return"], places=6,
                               msg="gold_return_lag1 hôm nay không khớp với gold_return hôm qua!")

    def test_training_feature_columns_exclude_targets(self):
        """Danh sách đặc trưng huấn luyện (X) tuyệt đối không được chứa các cột target (chống Data Leakage)."""
        feature_cols = get_feature_columns_for_training(self.df_full)
        for t in ["target_return", "target_close", "target_direction"]:
            self.assertNotIn(t, feature_cols, f"Phát hiện rò rỉ dữ liệu: {t} nằm trong danh sách X features!")
        self.assertGreater(len(feature_cols), 25, "Số lượng X features phải > 25 đặc trưng!")

if __name__ == "__main__":
    unittest.main()
