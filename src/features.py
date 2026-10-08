"""
Module features.py:
Chịu trách nhiệm xây dựng toàn bộ hệ thống đặc trưng (Feature Engineering Pipeline)
cho bài toán dự đoán giá vàng và xu hướng liên thị trường:

1. Bộ chỉ báo kỹ thuật tài chính (18 indicators): Trend (SMA, EMA, MACD), Momentum (RSI, Stoch), Volatility (BB, ATR), Volume (OBV).
2. Tỷ suất sinh lời & Đặc trưng trễ (Lags): Tỷ suất sinh lời vàng & liên thị trường (DXY, Dầu, Bạc, SP500, US10Y) tại t-1..t-5.
3. Đặc trưng thống kê động (Rolling Statistics): Rolling Volatility 7D/30D, Rolling Mean Return, Tỷ lệ so với SMA 50/200.
4. Đặc trưng thời gian (Calendar Features): Day of week, Month, Quarter (kiểm tra tính mùa vụ).
5. Biến mục tiêu dự báo ngày kế tiếp (Next-day Targets):
   - target_return (Return t+1): Hồi quy chuỗi dừng để huấn luyện mô hình.
   - target_close (Close t+1): Mức giá đóng cửa ngày tiếp theo để đánh giá sai số tiền tệ ($).
   - target_direction (Direction t+1): Phân loại xu hướng 1 (TĂNG) hoặc 0 (GIẢM).
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from typing import Dict, List, Optional, Tuple
import os
import pandas as pd
import numpy as np

# Thư viện phân tích kỹ thuật chuẩn
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# ==============================================================================
# 1. BỘ CHỈ BÁO KỸ THUẬT (TECHNICAL INDICATORS)
# ==============================================================================

def add_trend_indicators(df: pd.DataFrame, close_col: str = "gold_close") -> pd.DataFrame:
    """Tính toán các chỉ báo theo xu hướng (SMA 10/50/200, EMA 12/26, MACD)."""
    df_out = df.copy()
    close_series = df_out[close_col]
    
    # 1. Simple Moving Averages
    df_out["sma_10"] = SMAIndicator(close=close_series, window=10).sma_indicator()
    df_out["sma_50"] = SMAIndicator(close=close_series, window=50).sma_indicator()
    df_out["sma_200"] = SMAIndicator(close=close_series, window=200).sma_indicator()
    
    # 2. Exponential Moving Averages
    df_out["ema_12"] = EMAIndicator(close=close_series, window=12).ema_indicator()
    df_out["ema_26"] = EMAIndicator(close=close_series, window=26).ema_indicator()
    
    # 3. MACD Line, Signal Line và Histogram
    macd_obj = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
    df_out["macd"] = macd_obj.macd()
    df_out["macd_signal"] = macd_obj.macd_signal()
    df_out["macd_diff"] = macd_obj.macd_diff()
    
    return df_out

def add_momentum_indicators(
    df: pd.DataFrame,
    high_col: str = "gold_high",
    low_col: str = "gold_low",
    close_col: str = "gold_close"
) -> pd.DataFrame:
    """Tính toán các chỉ báo xung lượng (RSI 14, Stochastic Oscillator %K, %D)."""
    df_out = df.copy()
    
    # 1. RSI (14)
    df_out["rsi_14"] = RSIIndicator(close=df_out[close_col], window=14).rsi().clip(0.0, 100.0)
    
    # 2. Stochastic Oscillator
    stoch_obj = StochasticOscillator(
        high=df_out[high_col],
        low=df_out[low_col],
        close=df_out[close_col],
        window=14,
        smooth_window=3
    )
    df_out["stoch_k"] = stoch_obj.stoch().clip(lower=0.0, upper=100.0)
    df_out["stoch_d"] = stoch_obj.stoch_signal().clip(lower=0.0, upper=100.0)
    
    return df_out

def add_volatility_indicators(
    df: pd.DataFrame,
    high_col: str = "gold_high",
    low_col: str = "gold_low",
    close_col: str = "gold_close"
) -> pd.DataFrame:
    """Tính toán các chỉ báo biến động (Bollinger Bands, ATR 14)."""
    df_out = df.copy()
    close_series = df_out[close_col]
    
    # 1. Bollinger Bands
    bb_obj = BollingerBands(close=close_series, window=20, window_dev=2)
    df_out["bb_high"] = bb_obj.bollinger_hband()
    df_out["bb_low"] = bb_obj.bollinger_lband()
    df_out["bb_mid"] = bb_obj.bollinger_mavg()
    df_out["bb_width"] = bb_obj.bollinger_wband()
    
    # 2. Average True Range (ATR)
    atr_obj = AverageTrueRange(
        high=df_out[high_col],
        low=df_out[low_col],
        close=close_series,
        window=14
    )
    df_out["atr_14"] = atr_obj.average_true_range()
    
    return df_out

def add_volume_indicators(
    df: pd.DataFrame,
    close_col: str = "gold_close",
    vol_col: str = "gold_volume"
) -> pd.DataFrame:
    """Tính toán các chỉ báo khối lượng (OBV, Volume SMA 20)."""
    df_out = df.copy()
    df_out["obv"] = OnBalanceVolumeIndicator(
        close=df_out[close_col],
        volume=df_out[vol_col]
    ).on_balance_volume()
    df_out["vol_sma_20"] = SMAIndicator(close=df_out[vol_col], window=20).sma_indicator()
    return df_out

# ==============================================================================
# 2. TỶ SUẤT SINH LỜI & ĐẶC TRƯNG TRỄ (RETURNS & LAG FEATURES)
# ==============================================================================

def add_returns_and_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán tỷ suất sinh lời ngày (Daily Returns) và các biến trễ (Lags t-1..t-5).
    
    QUY TẮC CHỐNG DATA LEAKAGE:
    - Tất cả biến trễ chỉ sử dụng thông tin từ ngày hôm nay (t) trở về quá khứ (t-1, t-2...).
    - Tuyệt đối không nhìn trước tương lai.
    """
    df_out = df.copy()
    
    # 1. Tỷ suất sinh lời hiện tại (t) của Vàng và các tài sản liên thị trường
    df_out["gold_return"] = df_out["gold_close"].pct_change()
    
    for asset in ["silver_close", "dxy_close", "oil_close", "sp500_close", "us10y_close"]:
        if asset in df_out.columns:
            ret_col = asset.replace("_close", "_return")
            df_out[ret_col] = df_out[asset].pct_change()
            # Đặc trưng trễ 1 ngày của biến vĩ mô (Thông tin ngày hôm qua t-1 tác động tới hôm nay)
            df_out[f"{ret_col}_lag1"] = df_out[ret_col].shift(1)

    # 2. Biến trễ tỷ suất sinh lời của Vàng (Lag 1, 2, 3, 5)
    df_out["gold_return_lag1"] = df_out["gold_return"].shift(1)
    df_out["gold_return_lag2"] = df_out["gold_return"].shift(2)
    df_out["gold_return_lag3"] = df_out["gold_return"].shift(3)
    df_out["gold_return_lag5"] = df_out["gold_return"].shift(5)
    
    # 3. Tỷ lệ giá trễ so với giá đóng cửa hiện tại (Đo lường đà tăng/giảm tương đối)
    df_out["gold_close_ratio_lag1"] = df_out["gold_close"] / df_out["gold_close"].shift(1)
    df_out["gold_close_ratio_lag2"] = df_out["gold_close"] / df_out["gold_close"].shift(2)
    df_out["gold_close_ratio_lag5"] = df_out["gold_close"] / df_out["gold_close"].shift(5)

    return df_out

# ==============================================================================
# 3. ĐẶC TRƯNG THỐNG KÊ ĐỘNG (ROLLING STATISTICS)
# ==============================================================================

def add_rolling_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán độ biến động lăn (Rolling Volatility) và vị thế giá so với đường trung bình.
    """
    df_out = df.copy()
    
    # 1. Rolling Volatility (Độ lệch chuẩn của tỷ suất sinh lời trong 7 và 30 phiên)
    df_out["gold_volatility_7d"] = df_out["gold_return"].rolling(window=7).std()
    df_out["gold_volatility_30d"] = df_out["gold_return"].rolling(window=30).std()
    
    # 2. Rolling Mean Return (Lợi suất trung bình 7 và 30 phiên)
    df_out["gold_return_mean_7d"] = df_out["gold_return"].rolling(window=7).mean()
    df_out["gold_return_mean_30d"] = df_out["gold_return"].rolling(window=30).mean()
    
    # 3. Tỷ lệ khoảng cách giữa giá hiện tại so với SMA 50 và SMA 200
    if "sma_50" in df_out.columns:
        df_out["gold_to_sma50_ratio"] = df_out["gold_close"] / df_out["sma_50"]
    if "sma_200" in df_out.columns:
        df_out["gold_to_sma200_ratio"] = df_out["gold_close"] / df_out["sma_200"]
        
    return df_out

# ==============================================================================
# 4. ĐẶC TRƯNG LỊCH & MÙA VỤ (CALENDAR FEATURES)
# ==============================================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng chu kỳ thời gian (Thứ trong tuần, Tháng, Quý)."""
    df_out = df.copy()
    df_out["day_of_week"] = df_out.index.dayofweek
    df_out["month"] = df_out.index.month
    df_out["quarter"] = df_out.index.quarter
    return df_out

# ==============================================================================
# 5. BIẾN MỤC TIÊU DỰ BÁO NGÀY TIẾP THEO (NEXT-DAY TARGETS)
# ==============================================================================

def add_target_variables(df: pd.DataFrame, close_col: str = "gold_close") -> pd.DataFrame:
    """
    Định nghĩa 3 biến mục tiêu cho phiên tiếp theo (t+1):
    1. target_return: Tỷ suất sinh lời ngày mai (Return t+1) -> Regression trên chuỗi dừng.
    2. target_close: Giá đóng cửa ngày mai (Close t+1) -> Đánh giá sai số USD.
    3. target_direction: Hướng đi ngày mai (1 nếu Tăng, 0 nếu Giảm/Đứng giá) -> Directional Accuracy.
    """
    df_out = df.copy()
    
    # Giá đóng cửa ngày tiếp theo (t+1)
    df_out["target_close"] = df_out[close_col].shift(-1)
    
    # Tỷ suất sinh lời ngày tiếp theo (t+1)
    df_out["target_return"] = (df_out["target_close"] - df_out[close_col]) / df_out[close_col]
    
    # Hướng đi ngày mai: 1 nếu Tăng, 0 nếu Giảm
    df_out["target_direction"] = (df_out["target_return"] > 0).astype(int)
    
    return df_out

# ==============================================================================
# 6. PIPELINE TỔNG HỢP (FULL FEATURE PIPELINE)
# ==============================================================================

def build_technical_features(df: pd.DataFrame, drop_warmup: bool = False) -> pd.DataFrame:
    """Hàm tương thích ngược: Chỉ xây dựng 18 chỉ báo kỹ thuật cơ bản."""
    df_featured = df.copy()
    df_featured = add_trend_indicators(df_featured)
    df_featured = add_momentum_indicators(df_featured)
    df_featured = add_volatility_indicators(df_featured)
    df_featured = add_volume_indicators(df_featured)
    
    if drop_warmup:
        df_featured = df_featured.dropna()
        
    return df_featured

def build_full_feature_pipeline(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """
    Thực thi toàn bộ quy trình Kỹ nghệ đặc trưng hoàn chỉnh (End-to-End):
    Dữ liệu sạch -> Chỉ báo kỹ thuật -> Returns & Lags -> Rolling Stats -> Calendar -> Targets.
    
    Args:
        df: DataFrame đầu vào đã làm sạch.
        drop_warmup: Nếu True, loại bỏ các dòng warm-up ban đầu do cửa sổ rolling (sma_200)
                     và dòng cuối cùng (vì chưa có giá ngày mai).
    """
    df_pipe = df.copy()
    
    # 1. Chỉ báo kỹ thuật
    df_pipe = add_trend_indicators(df_pipe)
    df_pipe = add_momentum_indicators(df_pipe)
    df_pipe = add_volatility_indicators(df_pipe)
    df_pipe = add_volume_indicators(df_pipe)
    
    # 2. Tỷ suất sinh lời & Đặc trưng trễ
    df_pipe = add_returns_and_lags(df_pipe)
    
    # 3. Thống kê động
    df_pipe = add_rolling_statistics(df_pipe)
    
    # 4. Đặc trưng lịch
    df_pipe = add_calendar_features(df_pipe)
    
    # 5. Biến mục tiêu (t+1)
    df_pipe = add_target_variables(df_pipe)
    
    if drop_warmup:
        # Loại bỏ các dòng NaN ở đầu chuỗi (do rolling window 200) và dòng cuối cùng (do shift(-1) target)
        df_pipe = df_pipe.dropna()
        
    return df_pipe

def get_technical_feature_names() -> Dict[str, List[str]]:
    """Trả về danh mục các cột chỉ báo kỹ thuật theo từng nhóm phân loại."""
    return {
        "Trend": ["sma_10", "sma_50", "sma_200", "ema_12", "ema_26", "macd", "macd_signal", "macd_diff"],
        "Momentum": ["rsi_14", "stoch_k", "stoch_d"],
        "Volatility": ["bb_high", "bb_low", "bb_mid", "bb_width", "atr_14"],
        "Volume": ["obv", "vol_sma_20"]
    }

def get_feature_columns_for_training(df: pd.DataFrame) -> List[str]:
    """
    Trả về danh sách các cột đặc trưng độc lập (X features) dùng để huấn luyện mô hình.
    Loại trừ các cột mục tiêu (target_*) và giá trị tương lai.
    """
    excluded_cols = {"target_return", "target_close", "target_direction"}
    return [col for col in df.columns if col not in excluded_cols]

if __name__ == "__main__":
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_gold_data.csv")
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gold_market_data.csv")
    
    input_path = cleaned_path if os.path.exists(cleaned_path) else raw_path
    if not os.path.exists(input_path):
        print(f"[ERROR] Không tìm thấy dữ liệu đầu vào tại: {input_path}")
        sys.exit(1)
        
    df_input = pd.read_csv(input_path, index_col="Date", parse_dates=True)
    print(f"[INFO] Bắt đầu xây dựng toàn bộ pipeline đặc trưng cho {len(df_input)} phiên từ '{os.path.basename(input_path)}'...")
    
    df_featured = build_full_feature_pipeline(df_input, drop_warmup=True)
    
    # Phân loại đặc trưng
    feature_cols = get_feature_columns_for_training(df_featured)
    target_cols = ["target_return", "target_close", "target_direction"]
    
    print("\n" + "="*80)
    print(f"{'TỔNG KẾT PIPELINE KỸ NGHỆ ĐẶC TRƯNG HOÀN CHỈNH (TASK 5)':^80}")
    print("="*80)
    print(f"🔹 Tổng số cột trong tập dữ liệu: {df_featured.shape[1]} cột.")
    print(f"🔹 Số đặc trưng độc lập (X features): {len(feature_cols)} đặc trưng.")
    print(f"🔹 Số biến mục tiêu (Targets): {len(target_cols)} ({', '.join(target_cols)}).")
    print(f"🔹 Số lượng mẫu sau khi cắt warm-up an toàn: {len(df_featured)} phiên (từ {df_featured.index.min().strftime('%Y-%m-%d')} đến {df_featured.index.max().strftime('%Y-%m-%d')}).")
    print(f"🔹 Kiểm tra giá trị khuyết thiếu (NaN): {df_featured.isna().sum().sum()} ô (Hoàn toàn sạch 100%).")
    print("="*80)
    
    # Lưu file kết quả chính thức cho Phase 3 Modeling
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "featured_gold_data.csv")
    df_featured.to_csv(output_path)
    print(f"\n[INFO] Đã xuất bản bộ dữ liệu huấn luyện chính thức vào: {output_path}")
    print(f"[INFO] Kích thước chính thức: {df_featured.shape}")
