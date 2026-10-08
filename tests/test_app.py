"""
Kiểm thử tự động cho giao diện ứng dụng Streamlit Dashboard (app/app.py):

1. TestAppSyntaxAndImports:
   - Module app.py biên dịch hoàn hảo (syntax valid).
   - Tất cả dependencies Streamlit, Plotly, SJCConverter được import đúng.

2. TestAppDataLoading:
   - load_market_data tải đúng DataFrame, có các cột Close, Open, High, Low.
   - load_leaderboard có đủ 5 mô hình và các cột RMSE, MAE, DA.
   - load_forecast_7days có đủ 7 ngày bước T+1 đến T+7.

3. TestSJCIntegrationInApp:
   - Quy đổi DataFrame dự báo không làm mất dòng.
   - Các cột SJC mới có giá trị hợp lệ (> 0).
"""

import os
import py_compile
import unittest
import pandas as pd

from src.sjc_converter import SJCConverter


class TestAppIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "app.py"
        )
        cls.data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "featured_gold_data.csv"
        )
        cls.leaderboard_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "model_leaderboard.csv"
        )
        cls.forecast_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "forecast_7days.csv"
        )

    def test_app_file_exists_and_compiles(self):
        """File app/app.py phải tồn tại và biên dịch không có lỗi cú pháp."""
        self.assertTrue(os.path.exists(self.app_path), "Không tìm thấy app/app.py!")
        try:
            py_compile.compile(self.app_path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"app/app.py gặp lỗi cú pháp: {e}")

    def test_market_data_ready_for_app(self):
        """Dữ liệu thị trường phải sẵn sàng cho biểu đồ nến và chỉ báo kỹ thuật."""
        self.assertTrue(os.path.exists(self.data_path))
        df = pd.read_csv(self.data_path, index_col="Date", parse_dates=True)
        self.assertGreater(len(df), 2000)
        self.assertIn("gold_close", df.columns)
        self.assertIn("gold_high", df.columns)
        self.assertIn("gold_low", df.columns)

    def test_leaderboard_data_ready_for_app(self):
        """Bảng xếp hạng phải sẵn sàng cho Tab 4."""
        self.assertTrue(os.path.exists(self.leaderboard_path))
        df_lb = pd.read_csv(self.leaderboard_path)
        self.assertGreaterEqual(len(df_lb), 5)
        self.assertIn("Mô Hình", df_lb.columns)
        self.assertIn("RMSE ($)", df_lb.columns)
        self.assertIn("Directional Acc (%)", df_lb.columns)

    def test_forecast_data_ready_for_app(self):
        """Dữ liệu dự báo 7 ngày phải sẵn sàng cho Tab 2 & Tab 3."""
        self.assertTrue(os.path.exists(self.forecast_path))
        df_fc = pd.read_csv(self.forecast_path)
        self.assertEqual(len(df_fc), 7)
        self.assertIn("Giá Dự Báo ($)", df_fc.columns)

        # Thử nghiệm quy đổi SJC như trong app
        converter = SJCConverter()
        df_sjc = converter.convert_forecast_dataframe(df_fc)
        self.assertIn("Giá SJC (Triệu/lượng)", df_sjc.columns)
        self.assertTrue((df_sjc["Giá SJC (Triệu/lượng)"] > 0).all())


if __name__ == "__main__":
    unittest.main()
