"""
Kiểm thử tự động cho module data_loader.py
Đảm bảo dữ liệu tải về từ Yahoo Finance đáp ứng các tiêu chuẩn chất lượng (Data Quality Gates):
1. File CSV tồn tại và không rỗng.
2. Không chứa bất kỳ giá trị NaN / Null nào.
3. Đủ 10 cột dữ liệu (OHLCV của Vàng + 5 biến vĩ mô).
4. Số lượng mẫu (dòng) đủ lớn (> 2000 dòng).
5. Trục thời gian (Date) được sắp xếp tăng dần liên tục.
"""

import os
import unittest
import pandas as pd

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gold_market_data.csv")
        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Chưa tìm thấy file dữ liệu tại: {cls.data_path}. Hãy chạy data_loader.py trước!")
        cls.df = pd.read_csv(cls.data_path, index_col="Date", parse_dates=True)

    def test_file_exists_and_not_empty(self):
        """Kiểm tra file dữ liệu tồn tại và có số lượng dòng > 2000."""
        self.assertGreater(len(self.df), 2000, f"Dữ liệu quá ít: {len(self.df)} dòng (kỳ vọng > 2000 dòng).")

    def test_no_null_values(self):
        """Kiểm tra không còn bất kỳ giá trị NaN / Missing nào."""
        null_count = self.df.isna().sum().sum()
        self.assertEqual(null_count, 0, f"Còn {null_count} giá trị NaN trong tập dữ liệu!")

    def test_required_columns_present(self):
        """Kiểm tra có đầy đủ 10 cột đặc trưng bắt buộc."""
        required_cols = [
            "gold_open", "gold_high", "gold_low", "gold_close", "gold_volume",
            "silver_close", "dxy_close", "oil_close", "sp500_close", "us10y_close"
        ]
        for col in required_cols:
            self.assertIn(col, self.df.columns, f"Thiếu cột bắt buộc: {col}")

    def test_chronological_order(self):
        """Kiểm tra chuỗi thời gian tăng dần đều (không bị đảo lộn thời gian)."""
        self.assertTrue(self.df.index.is_monotonic_increasing, "Chỉ số thời gian (Date) không tăng dần đơn điệu!")

    def test_data_ranges_logical(self):
        """Kiểm tra logic tài chính: Giá High >= Low, Giá Vàng > 0."""
        self.assertTrue((self.df["gold_high"] >= self.df["gold_low"]).all(), "Tồn tại ngày có giá High < Low!")
        self.assertTrue((self.df["gold_close"] > 500).all(), "Giá vàng đóng cửa có giá trị bất thường (< 500 USD)!")

if __name__ == "__main__":
    unittest.main()
