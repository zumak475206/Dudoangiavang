"""
Module preprocessor.py:
Chịu trách nhiệm:
1. Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF Test) để xác định tính dừng (Stationarity).
2. Chuyển đổi dữ liệu chuỗi thời gian sang dạng dừng (Percentage Return & Log Return).
3. Phân tách tập dữ liệu Train / Validation / Test theo trình tự thời gian nghiêm ngặt (Strict Chronological Split - Không rò rỉ dữ liệu).
4. Chuẩn hóa đặc trưng (Scaling) tuân thủ nguyên tắc chống Data Leakage (Fit chỉ trên Train).
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def perform_adf_test(series: pd.Series, name: str = "") -> Dict[str, Union[float, int, bool, str, dict]]:
    """
    Thực hiện kiểm định Augmented Dickey-Fuller (ADF Test) trên một chuỗi thời gian.
    
    Giả thuyết thống kê:
    - H0 (Null Hypothesis): Chuỗi có nghiệm đơn vị (Non-stationary - Không dừng, có xu hướng ngẫu nhiên).
    - H1 (Alternative Hypothesis): Chuỗi không có nghiệm đơn vị (Stationary - Dừng, phương sai và kỳ vọng ổn định).
    
    Quy tắc bác bỏ:
    - Nếu p-value < 0.05 và ADF Statistic < Critical Value 5%: Bác bỏ H0 -> Chuỗi dừng.
    - Nếu p-value >= 0.05: Không đủ cơ sở bác bỏ H0 -> Chuỗi không dừng.
    
    Args:
        series: Chuỗi giá trị 1 chiều (pd.Series).
        name: Tên biến để hiển thị báo cáo.
        
    Returns:
        Dict chứa các chỉ số thống kê và kết luận.
    """
    clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    try:
        result = adfuller(clean_series, autolag="AIC", result_object=False)
    except TypeError:
        result = adfuller(clean_series, autolag="AIC")
    
    adf_stat = float(result[0])
    p_value = float(result[1])
    used_lag = int(result[2])
    n_obs = int(result[3])
    crit_values = {k: float(v) for k, v in result[4].items()}
    
    is_stationary = (p_value < 0.05) and (adf_stat < crit_values["5%"])
    
    if is_stationary:
        conclusion = f"Bác bỏ H0 ở mức ý nghĩa 5% (p={p_value:.4e}). Chuỗi đạt TÍNH DỪNG (Stationary)."
    else:
        conclusion = f"Không thể bác bỏ H0 (p={p_value:.4f} >= 0.05). Chuỗi KHÔNG DỪNG (Non-Stationary)."
        
    return {
        "name": name if name else series.name,
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_obs": n_obs,
        "critical_values": crit_values,
        "is_stationary": is_stationary,
        "conclusion": conclusion
    }

def print_adf_report(adf_results: List[Dict]) -> None:
    """In bảng tổng hợp kết quả kiểm định ADF ra màn hình dạng chuẩn học thuật."""
    print("\n" + "="*85)
    print(f"{'BẢNG BÁO CÁO KIỂM ĐỊNH TÍNH DỪNG (AUGMENTED DICKEY-FULLER TEST)':^85}")
    print("="*85)
    header = f"{'Tên Biến':<22} | {'ADF Stat':<10} | {'p-value':<12} | {'Crit Val 5%':<12} | {'Kết Luận':<15}"
    print(header)
    print("-" * 85)
    for res in adf_results:
        status = "DỪNG (Stationary)" if res["is_stationary"] else "KHÔNG DỪNG"
        p_val_str = f"{res['p_value']:.4e}" if res['p_value'] < 0.0001 else f"{res['p_value']:.4f}"
        print(f"{res['name']:<22} | {res['adf_statistic']:<10.4f} | {p_val_str:<12} | {res['critical_values']['5%']:<12.4f} | {status:<15}")
    print("="*85 + "\n")

def compute_returns(df: pd.DataFrame, price_col: str = "gold_close") -> pd.DataFrame:
    """
    Tính tỷ suất sinh lời phần trăm (Percentage Return) và tỷ suất sinh lời log (Log Return).
    
    Công thức:
    - Return_t = (P_t - P_{t-1}) / P_{t-1}
    - LogReturn_t = ln(P_t / P_{t-1})
    """
    df_out = df.copy()
    prefix = price_col.replace("_close", "")
    
    # 1. Percentage Return (Tỷ suất sinh lời phần trăm)
    df_out[f"{prefix}_return"] = df_out[price_col].pct_change()
    
    # 2. Log Return (Xử lý an toàn trường hợp giá <= 0 như giá Dầu thô tháng 4/2020)
    p_curr = df_out[price_col]
    p_prev = df_out[price_col].shift(1)
    
    valid_mask = (p_curr > 0) & (p_prev > 0)
    log_ret = pd.Series(np.nan, index=df_out.index)
    log_ret[valid_mask] = np.log(p_curr[valid_mask] / p_prev[valid_mask])
    
    # Với các giá trị không dương, fallback về pct_change
    df_out[f"{prefix}_log_return"] = log_ret.fillna(df_out[f"{prefix}_return"])
    
    return df_out

def time_series_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Phân tách dữ liệu chuỗi thời gian theo thứ tự thời gian nghiêm ngặt (Chronological Split).
    
    QUY TẮC CỐT LÕI (CHỐNG DATA LEAKAGE):
    - Tuyệt đối không dùng shuffle=True (xáo trộn dữ liệu).
    - Mốc thời gian tập Train luôn đi trước Validation, và Validation luôn đi trước Test:
      max(Train_Date) < min(Val_Date) <= max(Val_Date) < min(Test_Date)
      
    Args:
        df: DataFrame đầu vào có index là DatetimeIndex đã được sort tăng dần.
        train_ratio: Tỷ lệ tập huấn luyện (mặc định 70%).
        val_ratio: Tỷ lệ tập thẩm định (mặc định 15%).
        test_ratio: Tỷ lệ tập kiểm thử (mặc định 15%).
        
    Returns:
        (train_df, val_df, test_df)
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Tổng 3 tỷ lệ phải bằng 1.0!"
    assert df.index.is_monotonic_increasing, "DataFrame phải được sắp xếp tăng dần theo thời gian!"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

class TimeSeriesScaler:
    """
    Bộ chuẩn hóa dữ liệu chuỗi thời gian đảm bảo không rò rỉ phân phối (Data Leakage).
    Fit scaler DUY NHẤT trên tập Train, sau đó dùng các tham số (mean, std) đó
    để transform cho tập Val và Test.
    """
    def __init__(self, scaler_type: str = "standard"):
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type phải là 'standard' hoặc 'minmax'")
        self.scaler_type = scaler_type
        self.feature_cols: Optional[List[str]] = None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.scaler.fit(train_df[feature_cols])
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Scaler chưa được fit! Hãy gọi fit trên Train trước."
        assert self.feature_cols is not None
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df_scaled

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        self.fit(train_df, feature_cols)
        return self.transform(train_df)

if __name__ == "__main__":
    # Test thực thi trực tiếp module
    cleaned_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_gold_data.csv")
    raw_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "gold_market_data.csv")
    
    data_to_read = cleaned_path if os.path.exists(cleaned_path) else raw_data_path
    if not os.path.exists(data_to_read):
        print(f"[ERROR] Không tìm thấy file dữ liệu tại {data_to_read}. Hãy chạy data_cleaner.py hoặc data_loader.py trước!")
        sys.exit(1)

    df = pd.read_csv(data_to_read, index_col="Date", parse_dates=True)
    print(f"[INFO] Đã đọc dữ liệu từ '{os.path.basename(data_to_read)}': {df.shape[0]} dòng, {df.shape[1]} cột.")

    # 1. Tính toán Return của Vàng, Bạc, USD, Dầu
    for col in ["gold_close", "silver_close", "dxy_close", "oil_close"]:
        df = compute_returns(df, price_col=col)

    # 2. Chạy kiểm định tính dừng ADF Test
    tests_to_run = [
        ("Giá Vàng gốc (gold_close)", df["gold_close"]),
        ("Tỷ suất sinh lời Vàng (gold_return)", df["gold_return"]),
        ("Giá Bạc gốc (silver_close)", df["silver_close"]),
        ("Tỷ suất sinh lời Bạc (silver_return)", df["silver_return"]),
        ("Chỉ số USD gốc (dxy_close)", df["dxy_close"]),
        ("Tỷ suất sinh lời USD (dxy_return)", df["dxy_return"]),
        ("Giá Dầu thô gốc (oil_close)", df["oil_close"]),
        ("Tỷ suất sinh lời Dầu (oil_return)", df["oil_return"]),
    ]

    results = [perform_adf_test(s, name) for name, s in tests_to_run]
    print_adf_report(results)

    # 3. Phân tách Train / Val / Test mẫu
    train, val, test = time_series_split(df, 0.70, 0.15, 0.15)
    print(f"[INFO] Phân tách chuỗi thời gian (No Leakage):")
    print(f"  - Tập Train: {len(train)} mẫu ({train.index.min().strftime('%Y-%m-%d')} -> {train.index.max().strftime('%Y-%m-%d')})")
    print(f"  - Tập Val:   {len(val)} mẫu ({val.index.min().strftime('%Y-%m-%d')} -> {val.index.max().strftime('%Y-%m-%d')})")
    print(f"  - Tập Test:  {len(test)} mẫu ({test.index.min().strftime('%Y-%m-%d')} -> {test.index.max().strftime('%Y-%m-%d')})")
