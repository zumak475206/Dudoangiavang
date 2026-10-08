"""
Module sjc_converter.py:
Quy đổi giá vàng thế giới (USD/oz) sang giá vàng miếng SJC Việt Nam (Triệu VNĐ/lượng).

Cơ sở lý thuyết & Quy chuẩn đo lường tài chính:
1. Đơn vị khối lượng quốc tế vs Việt Nam:
   - 1 Ounce Troy (oz troy) vàng quốc tế = 31.1034768 grams.
   - 1 Lượng vàng (Cây vàng) Việt Nam = 10 Chỉ = 37.5000000 grams.
   - Hệ số chuyển đổi khối lượng:
     Hệ số = 37.5 / 31.1034768 ~= 1.205653 (tức 1 Lượng = 1.205653 Ounce).

2. Công thức quy đổi giá vàng nhập khẩu cơ sở (Base Parity Price):
   P_base_VND = (P_world_USD + Phí_vận_chuyển_bảo_hiểm) * Hệ_số * Tỷ_giá_USD/VND * (1 + Thuế_nhập_khẩu)

3. Cơ chế chênh lệch giá vàng SJC nội địa (Domestic SJC Premium/Spread):
   Tại Việt Nam, do chính sách quản lý vàng miếng độc quyền của Ngân hàng Nhà nước (Nghị định 24/2012),
   giá vàng miếng SJC thường cao hơn giá vàng thế giới quy đổi một khoảng chênh lệch (SJC Premium)
   dao động từ 3 đến 15 triệu VNĐ/lượng tùy từng thời kỳ.
   P_SJC_VND = P_base_VND + SJC_Premium_VND

4. Đơn vị tiền tệ hiển thị:
   - 1 Triệu VNĐ = 1,000,000 VNĐ.
   - Giá SJC thông thường được yết theo: "XX.XX Triệu VNĐ/lượng" (Ví dụ: 85.50 triệu VNĐ/lượng).
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# ==============================================================================
# HẰNG SỐ QUY CHUẨN ĐO LƯỜNG VÀ THỊ TRƯỜNG
# ==============================================================================

# Khối lượng
TROY_OUNCE_TO_GRAMS: float = 31.1034768
LUONG_TO_GRAMS: float = 37.5000000
CONVERSION_FACTOR: float = LUONG_TO_GRAMS / TROY_OUNCE_TO_GRAMS  # ~= 1.205653

# Tham số thị trường mặc định
DEFAULT_USD_VND_RATE: float = 25450.0          # Tỷ giá bán ra bình quân USD/VND (2024-2026)
DEFAULT_IMPORT_TAX_RATE: float = 0.01          # Thuế nhập khẩu & phụ phí hải quan (1%)
DEFAULT_SHIPPING_INSURANCE_USD: float = 2.0    # Phí vận chuyển, bảo hiểm quốc tế ($2/oz)
DEFAULT_SJC_PREMIUM_VND: float = 5_000_000.0   # Chênh lệch cung-cầu vàng miếng SJC (5 triệu VNĐ/lượng)


class SJCConverter:
    """
    Bộ quy đổi giá vàng thế giới (USD/oz) sang giá vàng miếng SJC (VNĐ/lượng & Triệu VNĐ/lượng).
    Hỗ trợ xử lý cả giá trị đơn lẻ (scalar) lẫn mảng/chuỗi thời gian/bảng dự báo.
    """

    def __init__(
        self,
        usd_vnd_rate: float = DEFAULT_USD_VND_RATE,
        sjc_premium_vnd: float = DEFAULT_SJC_PREMIUM_VND,
        import_tax_rate: float = DEFAULT_IMPORT_TAX_RATE,
        shipping_fee_usd: float = DEFAULT_SHIPPING_INSURANCE_USD,
    ):
        """
        Args:
            usd_vnd_rate: Tỷ giá quy đổi USD/VND (Ví dụ: 25450.0).
            sjc_premium_vnd: Độ chênh lệch giá vàng miếng SJC so với giá thế giới quy đổi (VNĐ).
            import_tax_rate: Tỷ lệ thuế nhập khẩu và phụ phí hải quan (mặc định 0.01 = 1%).
            shipping_fee_usd: Chi phí vận chuyển và bảo hiểm quốc tế ($/oz).
        """
        if usd_vnd_rate <= 0:
            raise ValueError("Tỷ giá USD/VND phải lớn hơn 0!")
        if import_tax_rate < 0:
            raise ValueError("Thuế suất nhập khẩu không thể âm!")
        if shipping_fee_usd < 0:
            raise ValueError("Phí vận chuyển không thể âm!")

        self.usd_vnd_rate = float(usd_vnd_rate)
        self.sjc_premium_vnd = float(sjc_premium_vnd)
        self.import_tax_rate = float(import_tax_rate)
        self.shipping_fee_usd = float(shipping_fee_usd)

    def convert_price(
        self,
        world_price_usd: Union[float, int, np.ndarray, pd.Series],
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """
        Quy đổi một mức giá hoặc một danh sách giá vàng thế giới sang giá SJC.

        Args:
            world_price_usd: Giá vàng thế giới tính theo USD/Ounce.

        Returns:
            - Nếu đầu vào là số đơn lẻ (scalar): Trả về dict chứa chi tiết các thành phần giá.
            - Nếu đầu vào là Series / ndarray: Trả về DataFrame chứa các cột quy đổi tương ứng.
        """
        is_scalar = isinstance(world_price_usd, (int, float, np.floating, np.integer))

        if is_scalar:
            p_usd = float(world_price_usd)
            if p_usd <= 0:
                raise ValueError("Giá vàng thế giới phải lớn hơn 0 USD!")

            # 1. Giá nhập khẩu cơ sở (chưa tính chênh lệch thương hiệu SJC)
            base_vnd = (
                (p_usd + self.shipping_fee_usd)
                * CONVERSION_FACTOR
                * self.usd_vnd_rate
                * (1.0 + self.import_tax_rate)
            )

            # 2. Giá vàng miếng SJC nội địa
            sjc_vnd = base_vnd + self.sjc_premium_vnd

            return {
                "world_price_usd": round(p_usd, 2),
                "usd_vnd_rate": round(self.usd_vnd_rate, 2),
                "base_price_vnd": round(base_vnd, 0),
                "base_price_trieu": round(base_vnd / 1_000_000.0, 3),
                "sjc_premium_vnd": round(self.sjc_premium_vnd, 0),
                "sjc_premium_trieu": round(self.sjc_premium_vnd / 1_000_000.0, 3),
                "sjc_price_vnd": round(sjc_vnd, 0),
                "sjc_price_trieu": round(sjc_vnd / 1_000_000.0, 3),
            }
        else:
            p_arr = np.asarray(world_price_usd, dtype=np.float64)
            base_vnd_arr = (
                (p_arr + self.shipping_fee_usd)
                * CONVERSION_FACTOR
                * self.usd_vnd_rate
                * (1.0 + self.import_tax_rate)
            )
            sjc_vnd_arr = base_vnd_arr + self.sjc_premium_vnd

            df_res = pd.DataFrame({
                "world_price_usd": np.round(p_arr, 2),
                "base_price_trieu": np.round(base_vnd_arr / 1_000_000.0, 3),
                "sjc_price_trieu": np.round(sjc_vnd_arr / 1_000_000.0, 3),
                "sjc_price_vnd": np.round(sjc_vnd_arr, 0),
            })
            if isinstance(world_price_usd, pd.Series):
                df_res.index = world_price_usd.index
            return df_res

    def convert_forecast_dataframe(
        self,
        forecast_df: pd.DataFrame,
        price_col: str = "Giá Dự Báo ($)",
        lower_col: Optional[str] = "Cận Dưới 95% ($)",
        upper_col: Optional[str] = "Cận Trên 95% ($)",
    ) -> pd.DataFrame:
        """
        Mở rộng bảng dự báo đa bước (từ evaluator.py) bằng cách bổ sung các cột giá SJC.

        Args:
            forecast_df: DataFrame dự báo 7 ngày có chứa cột giá USD.
            price_col: Tên cột giá dự báo chính.
            lower_col: Tên cột cận dưới khoảng tin cậy.
            upper_col: Tên cột cận trên khoảng tin cậy.

        Returns:
            pd.DataFrame mới có thêm các cột SJC (Triệu VNĐ/lượng).
        """
        if price_col not in forecast_df.columns:
            raise KeyError(f"Không tìm thấy cột '{price_col}' trong DataFrame dự báo!")

        df_out = forecast_df.copy()

        # Quy đổi giá chính
        prices_usd = df_out[price_col].values
        converted_main = self.convert_price(prices_usd)
        df_out["Giá SJC (Triệu/lượng)"] = converted_main["sjc_price_trieu"].values
        df_out["Giá Thế Giới Quy Đổi (Triệu/lượng)"] = converted_main["base_price_trieu"].values

        # Quy đổi cận dưới nếu có
        if lower_col and lower_col in df_out.columns:
            converted_lower = self.convert_price(df_out[lower_col].values)
            df_out["Cận Dưới SJC (Triệu/lượng)"] = converted_lower["sjc_price_trieu"].values

        # Quy đổi cận trên nếu có
        if upper_col and upper_col in df_out.columns:
            converted_upper = self.convert_price(df_out[upper_col].values)
            df_out["Cận Trên SJC (Triệu/lượng)"] = converted_upper["sjc_price_trieu"].values

        return df_out

    def calculate_spread(
        self,
        sjc_actual_vnd: float,
        world_price_usd: float,
    ) -> Dict[str, float]:
        """
        Tính toán mức chênh lệch thực tế (Spread/Premium) giữa giá niêm yết SJC
        ngoài thị trường và giá vàng thế giới quy đổi tại cùng thời điểm.

        Args:
            sjc_actual_vnd: Giá niêm yết SJC thực tế (VNĐ/lượng, ví dụ 85,500,000).
            world_price_usd: Giá vàng thế giới tại cùng thời điểm (USD/oz, ví dụ 2350.0).

        Returns:
            Dict chứa chênh lệch tuyệt đối (VNĐ), chênh lệch (triệu VNĐ) và tỷ lệ % chênh lệch.
        """
        conv = self.convert_price(world_price_usd)
        base_vnd = conv["base_price_vnd"]

        spread_vnd = sjc_actual_vnd - base_vnd
        spread_trieu = spread_vnd / 1_000_000.0
        spread_pct = (spread_vnd / base_vnd) * 100.0 if base_vnd > 0 else 0.0

        return {
            "sjc_actual_vnd": round(sjc_actual_vnd, 0),
            "sjc_actual_trieu": round(sjc_actual_vnd / 1_000_000.0, 3),
            "base_world_vnd": round(base_vnd, 0),
            "base_world_trieu": round(base_vnd / 1_000_000.0, 3),
            "spread_vnd": round(spread_vnd, 0),
            "spread_trieu": round(spread_trieu, 3),
            "spread_pct": round(spread_pct, 2),
        }


# ==============================================================================
# HÀM CONVENIENCE STANDALONE
# ==============================================================================

def convert_world_to_sjc(
    world_price_usd: float,
    usd_vnd_rate: float = DEFAULT_USD_VND_RATE,
    sjc_premium_vnd: float = DEFAULT_SJC_PREMIUM_VND,
) -> Dict[str, float]:
    """Hàm nhanh quy đổi giá vàng thế giới sang giá SJC."""
    converter = SJCConverter(usd_vnd_rate=usd_vnd_rate, sjc_premium_vnd=sjc_premium_vnd)
    return converter.convert_price(world_price_usd)


def process_forecast_sjc(
    forecast_path: Optional[str] = None,
    output_path: Optional[str] = None,
    usd_vnd_rate: float = DEFAULT_USD_VND_RATE,
    sjc_premium_vnd: float = DEFAULT_SJC_PREMIUM_VND,
) -> pd.DataFrame:
    """
    Đọc bảng dự báo forecast_7days.csv, bổ sung giá SJC và lưu kết quả.
    """
    if forecast_path is None:
        forecast_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "forecast_7days.csv"
        )
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "reports", "sjc_forecast_7days.csv"
        )

    if not os.path.exists(forecast_path):
        raise FileNotFoundError(f"Không tìm thấy file dự báo: {forecast_path}")

    df_fc = pd.read_csv(forecast_path)
    converter = SJCConverter(usd_vnd_rate=usd_vnd_rate, sjc_premium_vnd=sjc_premium_vnd)
    df_sjc = converter.convert_forecast_dataframe(df_fc)

    df_sjc.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df_sjc


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"{'MODULE QUY ĐỔI GIÁ VÀNG MIẾNG SJC (TASK 10)':^80}")
    print("=" * 80)

    converter = SJCConverter(
        usd_vnd_rate=25450.0,
        sjc_premium_vnd=5_000_000.0,
    )

    # Thử nghiệm với một mức giá điển hình
    test_usd = 4187.10
    res = converter.convert_price(test_usd)

    print(f"  • Giá Vàng Thế Giới        : ${res['world_price_usd']:,.2f} USD/oz")
    print(f"  • Tỷ giá USD/VND           : {res['usd_vnd_rate']:,.0f} VND/USD")
    print(f"  • Giá Thế Giới Quy Đổi     : {res['base_price_trieu']:,.3f} Triệu VNĐ/lượng ({res['base_price_vnd']:,.0f} VNĐ)")
    print(f"  • Chênh Lệch SJC (Premium) : +{res['sjc_premium_trieu']:,.3f} Triệu VNĐ/lượng")
    print(f"  • GIÁ VÀNG MIẾNG SJC       : {res['sjc_price_trieu']:,.3f} TRIỆU VNĐ/LƯỢNG ({res['sjc_price_vnd']:,.0f} VNĐ)")
    print("-" * 80)

    # Thử nghiệm tính Spread thực tế
    sjc_market = 135_000_000.0  # Ví dụ 135 triệu
    spread_res = converter.calculate_spread(sjc_market, test_usd)
    print(f"  • Khảo sát chênh lệch thực tế:")
    print(f"    - Giá SJC thị trường    : {spread_res['sjc_actual_trieu']} Triệu VNĐ/lượng")
    print(f"    - Giá nhập khẩu gốc     : {spread_res['base_world_trieu']} Triệu VNĐ/lượng")
    print(f"    - Chênh lệch nội địa    : +{spread_res['spread_trieu']} Triệu VNĐ (+{spread_res['spread_pct']}%)")
    print("=" * 80)

    # Quy đổi toàn bộ bảng dự báo 7 ngày từ reports/forecast_7days.csv
    try:
        df_sjc_fc = process_forecast_sjc()
        print(f"\n  [KẾT QUẢ QUY ĐỔI BẢNG DỰ BÁO 7 NGÀY SANG GIÁ SJC]:")
        view_cols = [
            "Bước",
            "Ngày Giao Dịch",
            "Giá Dự Báo ($)",
            "Giá SJC (Triệu/lượng)",
            "Cận Dưới SJC (Triệu/lượng)",
            "Cận Trên SJC (Triệu/lượng)",
        ]
        print(df_sjc_fc[view_cols].to_string(index=False))
        print(f"\n  [Lưu Trữ] Đã lưu bảng dự báo SJC vào: reports/sjc_forecast_7days.csv")
    except Exception as e:
        print(f"  [Lưu ý] Không thể quy đổi forecast_7days.csv: {e}")
