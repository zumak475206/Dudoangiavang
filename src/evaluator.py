"""
Module evaluator.py:
Hệ thống Đánh giá Thống nhất (Unified Evaluation System) & Dự báo Đa bước 7 ngày
phục vụ đồ án Machine Learning dự đoán giá vàng.

Chức năng cốt lõi:
1. Bộ độ đo thống nhất (Unified Metrics):
   - RMSE (Root Mean Squared Error): Sai số toàn phương trung bình ($ USD).
   - MAE (Mean Absolute Error): Sai số tuyệt đối trung bình ($ USD).
   - MAPE (Mean Absolute Percentage Error): Sai số phần trăm tuyệt đối trung bình (%).
   - Directional Accuracy (Hit Ratio / DA): Tỷ lệ dự báo đúng chiều Tăng/Giảm (%).
   - R² Score: Hệ số xác định mức độ giải thích phương sai của mô hình.

2. Bảng xếp hạng Leaderboard:
   - So sánh định lượng công bằng giữa 3 tầng mô hình:
     * Tầng 1: Baseline (Naive Persistence, Ridge Regression)
     * Tầng 2: Tree-based (XGBoost Regressor, LightGBM Regressor)
     * Tầng 3: Deep Learning (PyTorch Multi-variable LSTM)
   - Lưu trữ tự động ra 'reports/model_leaderboard.csv'.

3. Dự báo Đa bước 7 ngày (Recursive Multi-Step Forecasting):
   - Dự báo quỹ đạo giá vàng 7 phiên giao dịch tiếp theo dựa trên phiên gần nhất.
   - Bổ sung dải khoảng tin cậy 95% (95% Confidence Interval) cho từng bước dự báo:
     [Lower_Bound, Upper_Bound] = P_hat +/- 1.96 * sigma_residual * sqrt(step).
   - Lưu trữ tự động ra 'reports/forecast_7days.csv'.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Đảm bảo import được src.* khi chạy trực tiếp script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.models.baselines import BaseForecaster, NaivePersistenceModel, LinearRegressionBaseline
from src.models.tree_models import XGBoostForecaster, LightGBMForecaster
from src.models.lstm_model import LSTMForecaster
from src.preprocessor import TimeSeriesScaler
from src.features import get_feature_columns_for_training


# ==============================================================================
# 1. BỘ ĐỘ ĐO THỐNG NHẤT (UNIFIED METRICS)
# ==============================================================================

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error ($ USD)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error ($ USD)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Tránh chia cho 0 bằng cách loại bỏ giá trị <= 0."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mask = y_t > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100.0)


def calculate_directional_accuracy(y_true_dir: np.ndarray, y_pred_dir: np.ndarray) -> float:
    """
    Directional Accuracy (Hit Ratio) - Tỷ lệ dự báo đúng chiều Tăng/Giảm (%).
    Công thức: DA = (Tổng số phiên đoán đúng hướng / Tổng số phiên) * 100
    """
    yt = np.asarray(y_true_dir).ravel()
    yp = np.asarray(y_pred_dir).ravel()
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp) * 100.0)


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hệ số xác định R2 Score."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    if len(y_t) < 2:
        return 0.0
    return float(r2_score(y_t, y_p))


def compute_all_metrics(
    y_true_close: np.ndarray,
    y_pred_close: np.ndarray,
    y_true_dir: np.ndarray,
    y_pred_dir: np.ndarray,
) -> Dict[str, float]:
    """
    Tính toán trọn bộ 5 chỉ số đánh giá cốt lõi cho một bộ kết quả dự báo.
    """
    return {
        "rmse_usd": calculate_rmse(y_true_close, y_pred_close),
        "mae_usd": calculate_mae(y_true_close, y_pred_close),
        "mape_pct": calculate_mape(y_true_close, y_pred_close),
        "directional_accuracy_pct": calculate_directional_accuracy(y_true_dir, y_pred_dir),
        "r2_score": calculate_r2(y_true_close, y_pred_close),
    }


# ==============================================================================
# 2. ĐÁNH GIÁ MÔ HÌNH & XÂY DỰNG LEADERBOARD
# ==============================================================================

def evaluate_single_model(
    model: BaseForecaster,
    X_test: Union[pd.DataFrame, np.ndarray],
    current_close_test: Union[pd.Series, np.ndarray],
    y_test_close: Union[pd.Series, np.ndarray],
    y_test_dir: Union[pd.Series, np.ndarray],
    X_context: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Đánh giá một mô hình đơn lẻ trên tập Test và trả về metrics kèm chuỗi dự báo.
    """
    curr_close = np.asarray(current_close_test).ravel()
    y_true_c = np.asarray(y_test_close).ravel()
    y_true_d = np.asarray(y_test_dir).ravel()

    # Xử lý riêng cho LSTM nếu cần context warm-up
    if isinstance(model, LSTMForecaster) and X_context is not None:
        pred_close = model.predict_close(X_test, curr_close, X_context=X_context)
        pred_dir = model.predict_direction(X_test, curr_close, X_context=X_context)
    else:
        pred_close = model.predict_close(X_test, curr_close)
        pred_dir = model.predict_direction(X_test, curr_close)

    # Căn chỉnh kích thước nếu có mô hình trả về độ dài ngắn hơn
    min_len = min(len(pred_close), len(y_true_c))
    pred_close_adj = pred_close[-min_len:]
    pred_dir_adj = pred_dir[-min_len:]
    y_true_c_adj = y_true_c[-min_len:]
    y_true_d_adj = y_true_d[-min_len:]

    metrics = compute_all_metrics(y_true_c_adj, pred_close_adj, y_true_d_adj, pred_dir_adj)
    metrics["model_name"] = model.name
    metrics["predicted_close"] = pred_close_adj
    metrics["predicted_direction"] = pred_dir_adj
    metrics["residuals"] = y_true_c_adj - pred_close_adj
    metrics["residual_std"] = float(np.std(metrics["residuals"]))

    return metrics


def generate_leaderboard(
    models: Dict[str, BaseForecaster],
    X_test: Union[pd.DataFrame, np.ndarray],
    current_close_test: Union[pd.Series, np.ndarray],
    y_test_close: Union[pd.Series, np.ndarray],
    y_test_dir: Union[pd.Series, np.ndarray],
    X_context: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Tạo Bảng xếp hạng Leaderboard so sánh toàn diện tất cả các mô hình.
    """
    records = []
    for model_key, model in models.items():
        eval_res = evaluate_single_model(
            model=model,
            X_test=X_test,
            current_close_test=current_close_test,
            y_test_close=y_test_close,
            y_test_dir=y_test_dir,
            X_context=X_context,
        )
        records.append({
            "Mô Hình": eval_res["model_name"],
            "RMSE ($)": round(eval_res["rmse_usd"], 2),
            "MAE ($)": round(eval_res["mae_usd"], 2),
            "MAPE (%)": round(eval_res["mape_pct"], 3),
            "Directional Acc (%)": round(eval_res["directional_accuracy_pct"], 2),
            "R2 Score": round(eval_res["r2_score"], 4),
            "Residual Std ($)": round(eval_res["residual_std"], 2),
        })

    df_leaderboard = pd.DataFrame(records)
    # Sắp xếp ưu tiên: RMSE tăng dần
    df_leaderboard = df_leaderboard.sort_values(by="RMSE ($)", ascending=True).reset_index(drop=True)
    return df_leaderboard


# ==============================================================================
# 3. DỰ BÁO ĐA BƯỚC 7 NGÀY (RECURSIVE MULTI-STEP FORECASTING)
# ==============================================================================

def forecast_next_n_days(
    model: BaseForecaster,
    latest_feature_row: pd.Series,
    latest_close: float,
    latest_date: pd.Timestamp,
    feature_cols: List[str],
    n_days: int = 7,
    residual_std: Optional[float] = None,
) -> pd.DataFrame:
    """
    Thực hiện dự báo quỹ đạo giá vàng n ngày tiếp theo bằng kỹ thuật Recursive Forecasting.

    Thuật toán:
    1. Bước 1: Dùng đặc trưng của phiên hiện tại t dự báo return r_hat_{t+1}.
    2. Quy đổi ra giá Close dự báo: P_hat_{t+1} = P_t * (1 + r_hat_{t+1}).
    3. Cập nhật các đặc trưng tự hồi quy (lags):
       - lag1 return <- r_hat_{t+1}
       - lag1 close <- P_hat_{t+1}
    4. Tăng ngày giao dịch (bỏ qua thứ 7, chủ nhật).
    5. Tính dải khoảng tin cậy 95%: CI = 1.96 * residual_std * sqrt(step).
    6. Lặp lại cho đến bước n.

    Args:
        model: Mô hình đã được huấn luyện.
        latest_feature_row: Dòng đặc trưng phiên gần nhất (đã chuẩn hóa hoặc raw tương ứng).
        latest_close: Mức giá Close thực tế của phiên gần nhất ($).
        latest_date: Ngày của phiên gần nhất.
        feature_cols: Danh sách tên cột đặc trưng X.
        n_days: Số ngày cần dự báo (mặc định 7 ngày).
        residual_std: Độ lệch chuẩn phần dư sai số để tính khoảng tin cậy.

    Returns:
        pd.DataFrame chứa kết quả dự báo 7 ngày.
    """
    if residual_std is None or residual_std <= 0:
        residual_std = 25.0  # Mặc định khoảng 25 USD nếu chưa có thống kê phần dư

    current_p = float(latest_close)
    current_features = latest_feature_row[feature_cols].copy().values.astype(np.float32)

    # Sinh danh sách các ngày giao dịch tiếp theo (bỏ qua Thứ Bảy & Chủ Nhật)
    future_dates = []
    curr_d = pd.to_datetime(latest_date)
    while len(future_dates) < n_days:
        curr_d += pd.Timedelta(days=1)
        if curr_d.weekday() < 5:  # 0: Thứ 2, 4: Thứ 6
            future_dates.append(curr_d)

    forecast_records = []

    for step in range(1, n_days + 1):
        # 1. Dự báo return cho bước tiếp theo
        feat_input = current_features.reshape(1, -1)
        if isinstance(model, LSTMForecaster):
            # Tạo cửa sổ lặp giả lập cho LSTM
            window = np.repeat(feat_input, model.seq_len, axis=0).reshape(1, model.seq_len, -1)
            import torch
            with torch.no_grad():
                pred_ret = model.net(torch.tensor(window, dtype=torch.float32).to(model.device)).item()
        else:
            pred_ret = float(model.predict(feat_input)[0])

        # 2. Quy đổi sang giá Close
        next_close = current_p * (1.0 + pred_ret)

        # 3. Tính khoảng tin cậy 95%
        # Biên độ sai số dãn dần theo căn bậc 2 của bước thời gian: margin = 1.96 * sigma * sqrt(step)
        margin = 1.96 * residual_std * np.sqrt(step)
        lower_bound = max(0.0, next_close - margin)
        upper_bound = next_close + margin

        direction_label = "TĂNG" if pred_ret > 0 else "GIẢM"
        change_usd = next_close - current_p
        change_pct = pred_ret * 100.0

        forecast_records.append({
            "Bước": f"T+{step}",
            "Ngày Giao Dịch": future_dates[step - 1].strftime("%Y-%m-%d"),
            "Thứ": future_dates[step - 1].strftime("%A"),
            "Giá Dự Báo ($)": round(next_close, 2),
            "Thay Đổi ($)": round(change_usd, 2),
            "Tỷ Suất (%)": round(change_pct, 3),
            "Xu Hướng": direction_label,
            "Cận Dưới 95% ($)": round(lower_bound, 2),
            "Cận Trên 95% ($)": round(upper_bound, 2),
        })

        # 4. Cập nhật trạng thái cho bước lặp tiếp theo
        current_p = next_close

        # Cập nhật đặc trưng trễ (nếu các cột lag nằm trong feature_cols)
        # Giả lập suy giảm tỷ suất sinh lời theo quán tính
        if "gold_return_lag1" in feature_cols:
            idx = feature_cols.index("gold_return_lag1")
            current_features[idx] = pred_ret

    return pd.DataFrame(forecast_records)


# ==============================================================================
# 4. WORKFLOW CHẠY TOÀN BỘ ĐÁNH GIÁ (PIPELINE EXECUTION)
# ==============================================================================

def run_evaluation_pipeline(
    data_path: Optional[str] = None,
    save_reports: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, BaseForecaster]]:
    """
    Quy trình Đánh giá Thống nhất hoàn chỉnh:
    1. Tải và phân tách tập dữ liệu featured_gold_data.csv.
    2. Khởi tạo và huấn luyện 5 mô hình (Naive, Ridge, XGBoost, LightGBM, LSTM).
    3. Đánh giá trên tập Test độc lập (2024 - nay) và xuất Leaderboard.
    4. Dự báo quỹ đạo 7 ngày cho mô hình có thành tích tốt nhất.
    5. Lưu kết quả kiểm toán vào thư mục reports/.
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "featured_gold_data.csv"
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    feature_cols = get_feature_columns_for_training(df)

    # 1. Phân tách chuỗi thời gian nghiêm ngặt 70/15/15
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    train_df = df.iloc[:t1]
    val_df = df.iloc[t1:t2]
    test_df = df.iloc[t2:]

    # 2. Chuẩn hóa phân phối (Fit Train only)
    scaler = TimeSeriesScaler("standard")
    scaler.fit(train_df, feature_cols)
    train_s = scaler.transform(train_df)
    val_s = scaler.transform(val_df)
    test_s = scaler.transform(test_df)

    X_train = train_s[feature_cols]
    y_train = train_df["target_return"]
    X_val = val_s[feature_cols]
    y_val = val_df["target_return"]
    X_test = test_s[feature_cols]

    curr_close_test = test_df["gold_close"].values
    y_test_close = test_df["target_close"].values
    y_test_dir = test_df["target_direction"].values

    if verbose:
        print("\n" + "=" * 90)
        print(f"{'HỆ THỐNG ĐÁNH GIÁ THỐNG NHẤT & DỰ BÁO 7 NGÀY (TASK 9 - CHECKPOINT 3)':^90}")
        print("=" * 90)
        print(f"  [Dữ liệu] Tổng số: {n} phiên | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        print(f"  [Đặc trưng] Số chiều không gian X: {len(feature_cols)} biến kinh tế vĩ mô & chỉ báo kỹ thuật.")
        print("=" * 90)

    # 3. Khởi tạo 5 mô hình đại diện 3 tầng kiến trúc
    models: Dict[str, BaseForecaster] = {
        "Naive Persistence": NaivePersistenceModel(mode="last_price"),
        "Ridge Regression": LinearRegressionBaseline(model_type="ridge", alpha=10.0, target_type="return"),
        "XGBoost Regressor": XGBoostForecaster(target_type="return"),
        "LightGBM Regressor": LightGBMForecaster(target_type="return"),
        "PyTorch LSTM": LSTMForecaster(
            input_size=len(feature_cols),
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            fc_hidden=32,
            seq_len=30,
            batch_size=32,
            max_epochs=20,  # 20 epochs cho đánh giá nhanh và ổn định
            learning_rate=1e-3,
            patience=5,
            target_type="return",
            verbose=False,
        ),
    }

    # 4. Huấn luyện toàn bộ mô hình
    for name, model in models.items():
        if verbose:
            print(f"  >>> Huấn luyện mô hình: {name}...")
        if isinstance(model, (XGBoostForecaster, LightGBMForecaster, LSTMForecaster)):
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
        else:
            model.fit(X_train, y_train)

    # 5. Xây dựng Leaderboard
    leaderboard = generate_leaderboard(
        models=models,
        X_test=X_test,
        current_close_test=curr_close_test,
        y_test_close=y_test_close,
        y_test_dir=y_test_dir,
        X_context=X_val,
    )

    if verbose:
        print("\n" + "=" * 95)
        print(f"{'BẢNG XẾP HẠNG LEADERBOARD TOÀN DIỆN (TẬP TEST ĐỘC LẬP 2024 - NAY)':^95}")
        print("=" * 95)
        print(leaderboard.to_string(index=False))
        print("=" * 95)

    # 6. Dự báo Đa bước 7 ngày với Mô hình Xuất sắc nhất
    best_model_name = leaderboard.iloc[0]["Mô Hình"]
    best_residual_std = float(leaderboard.iloc[0]["Residual Std ($)"])
    best_model = None
    for m in models.values():
        if m.name == best_model_name:
            best_model = m
            break
    if best_model is None:
        best_model = models["LightGBM Regressor"]

    latest_row = test_s.iloc[-1]
    latest_close = float(test_df["gold_close"].iloc[-1])
    latest_date = test_df.index[-1]

    forecast_7d = forecast_next_n_days(
        model=best_model,
        latest_feature_row=latest_row,
        latest_close=latest_close,
        latest_date=latest_date,
        feature_cols=feature_cols,
        n_days=7,
        residual_std=best_residual_std,
    )

    if verbose:
        print(f"\n" + "=" * 95)
        print(f"DỰ BÁO QUỸ ĐẠO GIÁ VÀNG 7 NGÀY TIẾP THEO (Mô hình: {best_model_name})")
        print(f"Giá phiên chốt ({latest_date.strftime('%Y-%m-%d')}): ${latest_close:,.2f} USD/oz")
        print("=" * 95)
        print(forecast_7d.to_string(index=False))
        print("=" * 95)

    # 7. Xuất file báo cáo
    if save_reports:
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        lb_path = os.path.join(reports_dir, "model_leaderboard.csv")
        fc_path = os.path.join(reports_dir, "forecast_7days.csv")

        leaderboard.to_csv(lb_path, index=False, encoding="utf-8-sig")
        forecast_7d.to_csv(fc_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"\n  [Lưu Trữ] Đã lưu Bảng xếp hạng vào: {lb_path}")
            print(f"  [Lưu Trữ] Đã lưu Dự báo 7 ngày vào: {fc_path}")

    return leaderboard, forecast_7d, models


if __name__ == "__main__":
    run_evaluation_pipeline(save_reports=True, verbose=True)
