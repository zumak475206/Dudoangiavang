"""
Module data_loader.py:
Chịu trách nhiệm tải dữ liệu giá vàng và các chỉ số kinh tế vĩ mô liên thị trường (Inter-market)
từ Yahoo Finance, sau đó làm sạch và hợp nhất thành một bộ dữ liệu đồng bộ theo mốc thời gian.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from typing import List, Optional
import os
import pandas as pd
import yfinance as yf

# Danh mục các mã tài sản tương quan vĩ mô với vàng
DEFAULT_TICKERS = {
    "GOLD": "GC=F",          # Giá vàng tương lai (Gold Futures)
    "SILVER": "SI=F",        # Giá bạc tương lai (Silver Futures)
    "DXY": "DX-Y.NYB",       # Chỉ số sức mạnh đồng USD (US Dollar Index)
    "OIL": "CL=F",           # Giá dầu thô WTI (Crude Oil)
    "SP500": "^GSPC",        # Chỉ số chứng khoán Mỹ S&P 500
    "US10Y": "^TNX",         # Lợi suất trái phiếu chính phủ Mỹ kỳ hạn 10 năm
}

def fetch_financial_data(
    tickers: Optional[dict] = None,
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Tải dữ liệu giá đóng cửa (Close/Adj Close) và khối lượng (Volume) của Vàng cùng các tài sản vĩ mô.
    
    Args:
        tickers: Dict chứa tên gợi nhớ và mã ticker (VD: {"GOLD": "GC=F"}).
        start_date: Ngày bắt đầu lấy dữ liệu (YYYY-MM-DD).
        end_date: Ngày kết thúc (Mặc định None = ngày gần nhất).
        save_path: Đường dẫn lưu file CSV nếu muốn lưu.
        
    Returns:
        pd.DataFrame: Bảng dữ liệu đã xử lý các ngày trùng khớp và nội suy missing.
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    ticker_symbols = list(tickers.values())
    print(f"[INFO] Bắt đầu tải dữ liệu cho các mã: {ticker_symbols} từ {start_date}...")

    # Tải dữ liệu hàng loạt từ yfinance
    raw_data = yf.download(
        ticker_symbols,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=True
    )

    df_combined = pd.DataFrame(index=raw_data.index)

    for name, symbol in tickers.items():
        try:
            if isinstance(raw_data.columns, pd.MultiIndex):
                if symbol in raw_data.columns.levels[0]:
                    sub_df = raw_data[symbol]
                elif symbol in raw_data.columns.levels[1]:
                    sub_df = raw_data.xs(symbol, level=1, axis=1)
                else:
                    print(f"[WARN] Không tìm thấy dữ liệu cho mã: {symbol}")
                    continue
            else:
                sub_df = raw_data

            if name == "GOLD":
                df_combined["gold_open"] = sub_df["Open"]
                df_combined["gold_high"] = sub_df["High"]
                df_combined["gold_low"] = sub_df["Low"]
                df_combined["gold_close"] = sub_df["Close"]
                df_combined["gold_volume"] = sub_df["Volume"]
            else:
                df_combined[f"{name.lower()}_close"] = sub_df["Close"]
        except Exception as e:
            print(f"[ERROR] Lỗi khi trích xuất mã {symbol} ({name}): {e}")

    # 1. Lưu bản dữ liệu thô NGUYÊN BẢN (chứa NaN thực tế do lệch lịch nghỉ lễ các sàn)
    raw_uncleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "raw_uncleaned_data.csv")
    os.makedirs(os.path.dirname(raw_uncleaned_path), exist_ok=True)
    df_combined.to_csv(raw_uncleaned_path)
    nan_counts = df_combined.isna().sum().to_dict()
    print(f"[INFO] Đã lưu dữ liệu thô nguyên bản: {raw_uncleaned_path} (chứa tổng cộng {df_combined.isna().sum().sum()} giá trị NaN do lệch sàn giao dịch)")

    # 2. Xử lý các ngày thị trường nghỉ giao dịch lệch pha (Forward-fill rồi Backward-fill)
    df_combined = df_combined.ffill().bfill()
    df_combined.dropna(inplace=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_combined.to_csv(save_path)
        print(f"[INFO] Đã lưu dữ liệu vào: {save_path} với kích thước {df_combined.shape}")

    return df_combined

if __name__ == "__main__":
    # Test thu thập dữ liệu mẫu
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gold_market_data.csv")
    df = fetch_financial_data(save_path=data_path)
    print("\n--- 5 DÒNG ĐẦU DỮ LIỆU ---")
    print(df.head())
    print("\n--- THÔNG TIN CÁC CỘT DỮ LIỆU ---")
    print(df.info())
