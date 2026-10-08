"""
Kiểm thử tự động cho module data_cleaner.py:
1. Xác nhận tệp dữ liệu thô (raw_uncleaned_data.csv) có chứa NaN thực tế.
2. Xác nhận tệp dữ liệu đã làm sạch (cleaned_gold_data.csv) sạch 100%, không còn NaN.
3. Xác nhận cột đặc trưng outlier_flag được tạo chính xác (chỉ nhận giá trị 0 hoặc 1).
4. Xác nhận file báo cáo kiểm toán đối chứng reports/data_cleaning_audit.md được tạo thành công.
"""

import os
import unittest
import pandas as pd

class TestDataCleaner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_uncleaned_data.csv")
        cls.cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_gold_data.csv")
        cls.audit_path = os.path.join(os.path.dirname(__file__), "..", "reports", "data_cleaning_audit.md")
        
        if not os.path.exists(cls.raw_path):
            raise FileNotFoundError(f"Chưa có file raw tại {cls.raw_path}")
        if not os.path.exists(cls.cleaned_path):
            raise FileNotFoundError(f"Chưa có file cleaned tại {cls.cleaned_path}")
        if not os.path.exists(cls.audit_path):
            raise FileNotFoundError(f"Chưa có file audit report tại {cls.audit_path}")
            
        cls.df_raw = pd.read_csv(cls.raw_path, index_col="Date", parse_dates=True)
        cls.df_cleaned = pd.read_csv(cls.cleaned_path, index_col="Date", parse_dates=True)

    def test_raw_data_contains_real_missing_values(self):
        """Dữ liệu thô ban đầu phải chứa các giá trị NaN thực tế do lệch lịch nghỉ lễ."""
        total_nans = self.df_raw.isna().sum().sum()
        self.assertGreater(total_nans, 0, f"Dữ liệu thô phải có NaN để chứng minh quy trình làm sạch (hiện có: {total_nans})")

    def test_cleaned_data_has_zero_nans(self):
        """Dữ liệu sau khi làm sạch bắt buộc phải sạch 100% (0 giá trị NaN)."""
        clean_nans = self.df_cleaned.isna().sum().sum()
        self.assertEqual(clean_nans, 0, f"Dữ liệu đã làm sạch vẫn còn {clean_nans} giá trị NaN!")

    def test_outlier_flag_present_and_binary(self):
        """Cột đặc trưng outlier_flag phải tồn tại và chỉ chứa các giá trị 0 hoặc 1."""
        self.assertIn("outlier_flag", self.df_cleaned.columns, "Thiếu cột outlier_flag!")
        unique_vals = set(self.df_cleaned["outlier_flag"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}), f"outlier_flag chứa giá trị không hợp lệ: {unique_vals}")
        # Phải có ít nhất một số phiên được gắn cờ biến động mạnh
        self.assertGreater(self.df_cleaned["outlier_flag"].sum(), 0, "Không có phiên nào được gắn cờ outlier!")

    def test_audit_report_content_integrity(self):
        """File báo cáo kiểm toán phải chứa các tiêu mục đối chứng cốt lõi."""
        with open(self.audit_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("BÁO CÁO KIỂM TOÁN TIỀN XỬ LÝ DỮ LIỆU", content)
        self.assertIn("BẢNG SO SÁNH ĐỐI CHỨNG TRƯỚC VÀ SAU XỬ LÝ", content)
        self.assertIn("Winsorization", content)

    def test_chronological_continuity(self):
        """Số dòng dữ liệu không bị hao hụt và thứ tự ngày tăng dần đơn điệu."""
        self.assertEqual(len(self.df_raw), len(self.df_cleaned), "Số dòng bị thay đổi sau khi làm sạch!")
        self.assertTrue(self.df_cleaned.index.is_monotonic_increasing, "Date index bị đảo lộn!")

if __name__ == "__main__":
    unittest.main()
