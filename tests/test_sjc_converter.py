"""
Kiểm thử tự động cho module src/sjc_converter.py:

1. TestConstantsAndInitialization:
   - Hệ số chuyển đổi 37.5 / 31.1034768 ~= 1.205653.
   - Kiểm tra chặn lỗi tham số không hợp lệ (tỷ giá <= 0, thuế < 0, phí vận chuyển < 0).

2. TestScalarConversion:
   - Độ chính xác toán học của công thức quy đổi giá cơ sở và giá SJC.
   - Đơn vị triệu VNĐ khớp chính xác với VNĐ chia 1,000,000.
   - Chặn lỗi khi giá vàng thế giới <= 0 USD.

3. TestVectorAndSeriesConversion:
   - Quy đổi mảng numpy trả về DataFrame đúng số dòng.
   - Quy đổi pd.Series giữ nguyên DateTime index hoặc custom index.

4. TestForecastDataFrameConversion:
   - Bổ sung đúng các cột SJC vào DataFrame dự báo đa bước.
   - Thứ tự Cận Dưới SJC <= Giá SJC <= Cận Trên SJC được bảo toàn.
   - Báo lỗi KeyError nếu thiếu cột giá USD bắt buộc.

5. TestSpreadCalculation:
   - Tính toán chính xác độ chênh lệch tuyệt đối (VNĐ), chênh lệch triệu VNĐ và phần trăm chênh lệch.
"""

import unittest
import numpy as np
import pandas as pd

from src.sjc_converter import (
    SJCConverter,
    CONVERSION_FACTOR,
    DEFAULT_USD_VND_RATE,
    DEFAULT_SJC_PREMIUM_VND,
    convert_world_to_sjc,
)


class TestConstantsAndInitialization(unittest.TestCase):
    def test_conversion_factor_exact(self):
        expected = 37.5 / 31.1034768
        self.assertAlmostEqual(CONVERSION_FACTOR, expected, places=6)
        self.assertAlmostEqual(CONVERSION_FACTOR, 1.205653, places=5)

    def test_invalid_usd_rate_raises(self):
        with self.assertRaises(ValueError):
            SJCConverter(usd_vnd_rate=0)
        with self.assertRaises(ValueError):
            SJCConverter(usd_vnd_rate=-25000)

    def test_invalid_tax_rate_raises(self):
        with self.assertRaises(ValueError):
            SJCConverter(import_tax_rate=-0.05)

    def test_invalid_shipping_fee_raises(self):
        with self.assertRaises(ValueError):
            SJCConverter(shipping_fee_usd=-10.0)


class TestScalarConversion(unittest.TestCase):
    def test_conversion_zero_tax_zero_shipping(self):
        converter = SJCConverter(
            usd_vnd_rate=25000.0,
            sjc_premium_vnd=0.0,
            import_tax_rate=0.0,
            shipping_fee_usd=0.0,
        )
        res = converter.convert_price(2000.0)

        # 2000 * (37.5 / 31.1034768) * 25000
        expected_vnd = 2000.0 * (37.5 / 31.1034768) * 25000.0
        self.assertAlmostEqual(res["base_price_vnd"], expected_vnd, delta=1.0)
        self.assertEqual(res["base_price_vnd"], res["sjc_price_vnd"])
        self.assertAlmostEqual(res["sjc_price_trieu"], res["sjc_price_vnd"] / 1_000_000.0, places=3)

    def test_premium_added_correctly(self):
        premium = 6_000_000.0
        converter = SJCConverter(
            usd_vnd_rate=25000.0,
            sjc_premium_vnd=premium,
            import_tax_rate=0.0,
            shipping_fee_usd=0.0,
        )
        res = converter.convert_price(2500.0)
        self.assertEqual(res["sjc_price_vnd"], res["base_price_vnd"] + premium)
        self.assertAlmostEqual(res["sjc_premium_trieu"], 6.0, places=3)

    def test_negative_world_price_raises(self):
        converter = SJCConverter()
        with self.assertRaises(ValueError):
            converter.convert_price(0.0)
        with self.assertRaises(ValueError):
            converter.convert_price(-1500.0)

    def test_convenience_function(self):
        res = convert_world_to_sjc(2400.0, usd_vnd_rate=25000.0, sjc_premium_vnd=5_000_000.0)
        self.assertIn("sjc_price_trieu", res)
        self.assertIn("base_price_trieu", res)
        self.assertGreater(res["sjc_price_vnd"], res["base_price_vnd"])


class TestVectorAndSeriesConversion(unittest.TestCase):
    def setUp(self):
        self.converter = SJCConverter(usd_vnd_rate=25000.0, sjc_premium_vnd=5_000_000.0)

    def test_numpy_array_conversion(self):
        prices = np.array([2000.0, 2200.0, 2500.0])
        df_res = self.converter.convert_price(prices)

        self.assertIsInstance(df_res, pd.DataFrame)
        self.assertEqual(len(df_res), 3)
        self.assertIn("sjc_price_trieu", df_res.columns)
        self.assertIn("base_price_trieu", df_res.columns)

        # Giá tăng dần
        sjc_vals = df_res["sjc_price_trieu"].values
        self.assertTrue((sjc_vals[:-1] < sjc_vals[1:]).all())

    def test_pandas_series_preserves_index(self):
        idx = pd.date_range("2026-01-01", periods=4, freq="D")
        s = pd.Series([2100.0, 2150.0, 2120.0, 2180.0], index=idx)
        df_res = self.converter.convert_price(s)

        self.assertEqual(list(df_res.index), list(idx))
        self.assertEqual(len(df_res), 4)


class TestForecastDataFrameConversion(unittest.TestCase):
    def setUp(self):
        self.converter = SJCConverter(usd_vnd_rate=25000.0, sjc_premium_vnd=5_000_000.0)
        self.mock_forecast = pd.DataFrame({
            "Bước": ["T+1", "T+2"],
            "Giá Dự Báo ($)": [2400.0, 2450.0],
            "Cận Dưới 95% ($)": [2350.0, 2400.0],
            "Cận Trên 95% ($)": [2450.0, 2500.0],
        })

    def test_forecast_dataframe_extension(self):
        df_out = self.converter.convert_forecast_dataframe(self.mock_forecast)

        expected_cols = [
            "Giá SJC (Triệu/lượng)",
            "Giá Thế Giới Quy Đổi (Triệu/lượng)",
            "Cận Dưới SJC (Triệu/lượng)",
            "Cận Trên SJC (Triệu/lượng)",
        ]
        for col in expected_cols:
            self.assertIn(col, df_out.columns)

        # Kiểm tra bất đẳng thức cận: Cận Dưới <= Giá <= Cận Trên
        for _, row in df_out.iterrows():
            lower = row["Cận Dưới SJC (Triệu/lượng)"]
            price = row["Giá SJC (Triệu/lượng)"]
            upper = row["Cận Trên SJC (Triệu/lượng)"]
            self.assertLessEqual(lower, price)
            self.assertGreaterEqual(upper, price)

    def test_missing_price_column_raises(self):
        bad_df = pd.DataFrame({"Bước": ["T+1"], "Wrong_Col": [2000.0]})
        with self.assertRaises(KeyError):
            self.converter.convert_forecast_dataframe(bad_df)


class TestSpreadCalculation(unittest.TestCase):
    def test_spread_calculation(self):
        converter = SJCConverter(
            usd_vnd_rate=25000.0,
            sjc_premium_vnd=0.0,
            import_tax_rate=0.0,
            shipping_fee_usd=0.0,
        )
        world_usd = 2000.0
        # Giá gốc = 2000 * 1.205653 * 25000 = 60,282,650 VNĐ
        # Giả sử ngoài thị trường bán 75,000,000 VNĐ
        market_sjc = 75_000_000.0
        spread_info = converter.calculate_spread(market_sjc, world_usd)

        expected_spread_vnd = market_sjc - spread_info["base_world_vnd"]
        self.assertAlmostEqual(spread_info["spread_vnd"], expected_spread_vnd, delta=1.0)
        self.assertGreater(spread_info["spread_pct"], 20.0)
        self.assertAlmostEqual(
            spread_info["spread_trieu"],
            spread_info["spread_vnd"] / 1_000_000.0,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
