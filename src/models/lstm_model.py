"""
Module lstm_model.py:
Xây dựng mạng học sâu PyTorch LSTM (Long Short-Term Memory) đa biến
cho bài toán dự báo chuỗi thời gian giá vàng.

Kiến trúc GoldLSTM:
1. Input Layer: Chuỗi sliding window W ngày, mỗi ngày có D đặc trưng kinh tế vĩ mô.
   Input Tensor: (batch_size, seq_len, input_size) = (B, W, D)

2. LSTM Layers: 2 lớp LSTM xếp chồng (Stacked LSTM):
   - Layer 1: input_size -> hidden_size (học pattern cơ bản: trend, momentum)
   - Layer 2: hidden_size -> hidden_size (học pattern bậc cao: chu kỳ, macro regime)
   - Dropout giữa các layer (tránh co-adaptation của neurons)

3. Batch Normalization Layer: Chuẩn hóa hidden state cuối cùng để ổn định gradient.

4. Output Head (Regression):
   - Linear(hidden_size -> 64) với ReLU activation
   - Dropout(p=0.2)
   - Linear(64 -> 1): Dự báo tỷ suất sinh lời r_hat_{t+1}

Nguyên lý học:
   - Thay vì học trực tiếp trên mức giá Close (non-stationary, vi phạm giả thuyết của LSTM),
     mô hình học trên TỶ SUẤT SINH LỜI (Return - Stationary, đã kiểm định ADF).
   - Sử dụng Teacher Forcing trong giai đoạn training (tự nhiên qua DataLoader).
   - Multi-step forecasting bằng kỹ thuật Recursive Prediction (dự báo tuần tự từng bước).

Tham chiếu khoa học:
   - Hochreiter & Schmidhuber (1997): Long Short-Term Memory, Neural Computation.
   - Chen et al. (2015): A LSTM-Based Method for Stock Returns Prediction.
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.baselines import BaseForecaster


# ==============================================================================
# 1. PYTORCH DATASET: Sliding Window Time Series
# ==============================================================================

class GoldTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset triển khai kỹ thuật Sliding Window (Cửa sổ trượt) cho chuỗi thời gian.

    Nguyên lý:
    Với mỗi phiên t, mô hình nhìn lại W phiên trước (t-W đến t-1) để dự báo Return tại phiên t:
        X[i] = features[i : i + seq_len]       -> shape (seq_len, n_features)
        y[i] = target_return[i + seq_len]       -> scalar
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 30,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            X: Ma trận đặc trưng (n_samples, n_features), đã chuẩn hóa.
            y: Vector biến mục tiêu (n_samples,).
            seq_len: Kích thước cửa sổ lịch sử (W ngày).
            dtype: Kiểu dữ liệu PyTorch Tensor.
        """
        assert len(X) == len(y), "X và y phải có cùng số dòng!"
        assert len(X) > seq_len, f"Cần ít nhất {seq_len + 1} mẫu để tạo dataset!"

        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Số lượng cửa sổ hợp lệ = n_samples - seq_len
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_window = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        y_target  = self.y[idx + self.seq_len].unsqueeze(0)  # (1,) -> output của phiên tiếp theo
        return x_window, y_target


# ==============================================================================
# 2. PYTORCH MODULE: GoldLSTM Architecture
# ==============================================================================

class GoldLSTMModule(nn.Module):
    """
    Kiến trúc mạng LSTM đa biến 2 lớp với Batch Normalization và Dropout.

    Forward pass:
        Input:  x  shape (batch, seq_len, input_size)
        Output: out shape (batch, 1)  -- tỷ suất sinh lời r_hat_{t+1}
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_hidden: int = 64,
    ):
        """
        Args:
            input_size:  Số chiều đặc trưng đầu vào (D = 56).
            hidden_size: Số chiều hidden state của LSTM.
            num_layers:  Số lớp LSTM xếp chồng.
            dropout:     Tỷ lệ Dropout giữa các LSTM layer (chỉ áp dụng khi num_layers > 1).
            fc_hidden:   Chiều rộng lớp Fully Connected trung gian.
        """
        super().__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Lớp LSTM xếp chồng (Stacked LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Input shape: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0,  # Dropout chỉ giữa layers
        )

        # Batch Normalization trên hidden state cuối cùng để ổn định gradient
        self.bn = nn.BatchNorm1d(hidden_size)

        # Dropout sau BN
        self.dropout = nn.Dropout(p=dropout)

        # Output head: hidden_size -> fc_hidden -> 1
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, 1) -- tỷ suất sinh lời dự báo
        """
        # 1. LSTM forward: chỉ lấy hidden state tại bước thời gian cuối cùng
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) -- hidden state của step cuối
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 2. Lấy hidden state của lớp LSTM trên cùng tại bước thời gian cuối
        # h_n[-1]: (batch, hidden_size)
        last_hidden = h_n[-1]

        # 3. Batch Normalization -> Dropout
        normed = self.bn(last_hidden)
        dropped = self.dropout(normed)

        # 4. Fully Connected head
        out = self.fc(dropped)  # (batch, 1)
        return out


# ==============================================================================
# 3. SKLEARN-COMPATIBLE WRAPPER: LSTMForecaster
# ==============================================================================

class LSTMForecaster(BaseForecaster):
    """
    Wrapper hoàn chỉnh tích hợp GoldLSTMModule vào pipeline BaseForecaster.

    Cung cấp interface giống scikit-learn (fit/predict) cho phép so sánh
    trực tiếp với XGBoost, LightGBM, và Baseline trong module evaluator.py.
    """

    def __init__(
        self,
        input_size: int = 56,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        fc_hidden: int = 64,
        seq_len: int = 30,
        batch_size: int = 64,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        target_type: str = "return",
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            input_size:    Số đặc trưng đầu vào (phải khớp với số cột X).
            hidden_size:   Số units trong mỗi LSTM layer (128 = cân bằng capacity/speed).
            num_layers:    Số LSTM layer xếp chồng (2 cho chuỗi thời gian tài chính).
            dropout:       Tỷ lệ dropout (0.2 chống overfitting).
            fc_hidden:     Số units trong FC layer trung gian.
            seq_len:       Kích thước cửa sổ trượt (30 ngày = 1 tháng giao dịch).
            batch_size:    Kích thước mini-batch.
            max_epochs:    Số epoch tối đa.
            learning_rate: Learning rate ban đầu (Adam optimizer).
            weight_decay:  L2 regularization trong Adam.
            patience:      Số epoch chịu đựng không cải thiện val loss (Early Stopping).
            target_type:   'return' hoặc 'close'.
            device:        'cpu', 'cuda', hoặc None (tự phát hiện).
            verbose:       In log tiến trình huấn luyện.
        """
        super().__init__(name="PyTorch LSTM", target_type=target_type)

        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.dropout       = dropout
        self.fc_hidden     = fc_hidden
        self.seq_len       = seq_len
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.patience      = patience
        self.verbose       = verbose

        # Tự phát hiện device: GPU nếu có, ngược lại CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Khởi tạo module PyTorch (chưa load weights)
        self.net: Optional[GoldLSTMModule] = None

        # Log huấn luyện để kiểm tra convergence
        self.train_losses_: List[float] = []
        self.val_losses_:   List[float] = []
        self.best_epoch_:   int = 0

    def _build_net(self, input_size: int) -> GoldLSTMModule:
        """Khởi tạo và chuyển mạng lên device."""
        net = GoldLSTMModule(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            fc_hidden=self.fc_hidden,
        )
        return net.to(self.device)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        verbose: Optional[bool] = None,
    ):
        """
        Huấn luyện LSTM với Early Stopping và ReduceLROnPlateau scheduler.

        Args:
            X, y: Tập Train (numpy/pandas).
            X_val, y_val: Tập Validation cho early stopping. Bắt buộc cung cấp để tránh overfitting.
            verbose: Ghi đè tham số verbose của instance.
        """
        _verbose = verbose if verbose is not None else self.verbose

        # 1. Chuẩn bị dữ liệu
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        input_size = X_arr.shape[1]

        # 2. Khởi tạo mạng (input_size có thể thay đổi khi test với dữ liệu khác)
        self.net = self._build_net(input_size)
        self.input_size = input_size

        # 3. Tạo DataLoader cho Train
        train_dataset = GoldTimeSeriesDataset(X_arr, y_arr, seq_len=self.seq_len)
        train_loader  = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # KHÔNG shuffle -- chuỗi thời gian phải theo thứ tự!
            drop_last=False,
            num_workers=0,  # Windows compatibility
        )

        # 4. Tạo DataLoader cho Validation (nếu có)
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val, dtype=np.float32)
            y_val_arr = np.asarray(y_val, dtype=np.float32).ravel()

            if len(X_val_arr) > self.seq_len:
                val_dataset = GoldTimeSeriesDataset(X_val_arr, y_val_arr, seq_len=self.seq_len)
                val_loader  = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        # 5. Optimizer và Learning Rate Scheduler
        optimizer = Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,        # Giảm LR xuống còn 50% khi val loss plateau
            patience=5,        # Chờ 5 epoch không cải thiện mới giảm LR
            min_lr=1e-6,
        )
        criterion = nn.MSELoss()

        # 6. Vòng lặp huấn luyện với Early Stopping
        best_val_loss    = float("inf")
        best_state_dict  = None
        no_improve_count = 0

        self.train_losses_ = []
        self.val_losses_   = []

        if _verbose:
            print(f"\n  [LSTM] Huấn luyện trên {self.device} | input_size={input_size} | seq_len={self.seq_len}")
            print(f"  [LSTM] Tập Train: {len(train_dataset)} windows | Max epochs: {self.max_epochs}")
            if val_loader:
                val_win = len(val_loader.dataset)
                print(f"  [LSTM] Tập Val:   {val_win} windows | Early stopping patience: {self.patience}")
            print()

        for epoch in range(1, self.max_epochs + 1):
            # --- Training phase ---
            self.net.train()
            epoch_train_loss = 0.0
            n_train_batches  = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.net(X_batch)  # (batch, 1)
                loss  = criterion(preds, y_batch)
                loss.backward()

                # Gradient Clipping để tránh exploding gradient trong LSTM
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_train_loss += loss.item()
                n_train_batches  += 1

            avg_train_loss = epoch_train_loss / max(n_train_batches, 1)
            self.train_losses_.append(avg_train_loss)

            # --- Validation phase ---
            if val_loader is not None:
                self.net.eval()
                epoch_val_loss  = 0.0
                n_val_batches   = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        preds   = self.net(X_batch)
                        loss    = criterion(preds, y_batch)
                        epoch_val_loss += loss.item()
                        n_val_batches  += 1

                avg_val_loss = epoch_val_loss / max(n_val_batches, 1)
                self.val_losses_.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                # Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss   = avg_val_loss
                    best_state_dict = {k: v.clone() for k, v in self.net.state_dict().items()}
                    no_improve_count = 0
                    self.best_epoch_ = epoch
                else:
                    no_improve_count += 1

                if _verbose and (epoch % 5 == 0 or epoch == 1):
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(f"  Epoch {epoch:>3}/{self.max_epochs} | "
                          f"Train Loss: {avg_train_loss:.6f} | "
                          f"Val Loss: {avg_val_loss:.6f} | "
                          f"LR: {lr_now:.2e} | "
                          f"{'★ BEST' if no_improve_count == 0 else f'No improve: {no_improve_count}/{self.patience}'}")

                if no_improve_count >= self.patience:
                    if _verbose:
                        print(f"\n  [LSTM] Early stopping tại epoch {epoch}. Best epoch: {self.best_epoch_} (Val Loss: {best_val_loss:.6f})")
                    break
            else:
                # Không có val -> dùng train loss để schedule LR
                scheduler.step(avg_train_loss)
                if _verbose and (epoch % 10 == 0 or epoch == 1):
                    print(f"  Epoch {epoch:>3}/{self.max_epochs} | Train Loss: {avg_train_loss:.6f}")

        # 7. Khôi phục trọng số tốt nhất
        if best_state_dict is not None:
            self.net.load_state_dict(best_state_dict)
            if _verbose:
                print(f"  [LSTM] Đã khôi phục trọng số tốt nhất từ epoch {self.best_epoch_}.")

        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Dự báo tỷ suất sinh lời cho từng phiên trong X.

        Quan trọng: Với mỗi phiên t, dự báo dùng cửa sổ [t-seq_len, ..., t-1].
        Nên kết quả có độ dài = len(X) - seq_len. Khi đánh giá trên tập test,
        cần căn chỉnh với predict_aligned() hoặc dùng seq_len đầu tiên từ val.

        Returns:
            preds: np.ndarray shape (n_samples - seq_len,)
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("LSTM chưa được huấn luyện! Hãy gọi fit() trước.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)

        self.net.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(self.seq_len, len(X_arr)):
                window = torch.tensor(
                    X_arr[i - self.seq_len : i],
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)  # (1, seq_len, input_size)

                pred = self.net(window)  # (1, 1)
                all_preds.append(pred.item())

        return np.array(all_preds, dtype=np.float32)

    def predict_with_context(
        self,
        X_context: Union[pd.DataFrame, np.ndarray],
        X_target:  Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Dự báo trên X_target bằng cách sử dụng X_context làm "warm-up" lịch sử.

        Giải quyết vấn đề "cold start" khi ranh giới Train/Val/Test cắt đứt chuỗi lịch sử.
        Sử dụng seq_len ngày cuối của X_context + tất cả ngày của X_target để dự báo.

        Args:
            X_context: seq_len ngày cuối của tập Train/Val (lịch sử để warm-up LSTM).
            X_target:  Tập Test cần dự báo.

        Returns:
            preds: np.ndarray shape (len(X_target),)
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("LSTM chưa được huấn luyện!")

        ctx_arr = X_context.values if isinstance(X_context, pd.DataFrame) else np.asarray(X_context, dtype=np.float32)
        tgt_arr = X_target.values  if isinstance(X_target,  pd.DataFrame) else np.asarray(X_target, dtype=np.float32)

        # Lấy seq_len ngày cuối của context làm lịch sử ban đầu
        ctx_tail = ctx_arr[-self.seq_len:]  # (seq_len, n_features)
        combined  = np.vstack([ctx_tail, tgt_arr])  # (seq_len + len(tgt), n_features)

        self.net.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(self.seq_len, len(combined)):
                window = torch.tensor(
                    combined[i - self.seq_len : i],
                    dtype=torch.float32,
                ).unsqueeze(0).to(self.device)

                pred = self.net(window)
                all_preds.append(pred.item())

        return np.array(all_preds, dtype=np.float32)

    def predict_close(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        current_close: Union[pd.Series, np.ndarray],
        X_context: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Dự báo mức giá đóng cửa P_hat_{t+1} = P_t * (1 + r_hat_{t+1}).

        Args:
            X: Tập Test features.
            current_close: Giá đóng cửa hiện tại P_t tương ứng với mỗi dòng X.
            X_context: Nếu cung cấp, dùng predict_with_context (khuyến nghị).
        """
        if X_context is not None:
            preds_ret = self.predict_with_context(X_context, X)
        else:
            preds_ret = self.predict(X)

        # Căn chỉnh current_close với kết quả (có thể ngắn hơn nếu không dùng context)
        curr = np.asarray(current_close).ravel()
        if X_context is None and len(preds_ret) < len(curr):
            curr = curr[self.seq_len:]

        return curr * (1.0 + preds_ret)

    def predict_direction(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        current_close: Optional[Union[pd.Series, np.ndarray]] = None,
        X_context: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> np.ndarray:
        """Dự báo xu hướng nhị phân: 1 nếu r_hat > 0 (tăng), 0 nếu giảm/đứng giá."""
        if X_context is not None:
            preds_ret = self.predict_with_context(X_context, X)
        else:
            preds_ret = self.predict(X)
        return (preds_ret > 0).astype(int)

    def get_loss_history(self) -> pd.DataFrame:
        """Trả về lịch sử train loss và val loss theo từng epoch."""
        n = max(len(self.train_losses_), len(self.val_losses_))
        return pd.DataFrame({
            "epoch":      list(range(1, n + 1)),
            "train_loss": self.train_losses_ + [None] * (n - len(self.train_losses_)),
            "val_loss":   self.val_losses_   + [None] * (n - len(self.val_losses_)),
        })

    def save_weights(self, path: str) -> None:
        """Lưu trọng số mô hình vào file .pt."""
        if self.net is None:
            raise RuntimeError("Mô hình chưa được khởi tạo!")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "state_dict":   self.net.state_dict(),
            "input_size":   self.input_size,
            "hidden_size":  self.hidden_size,
            "num_layers":   self.num_layers,
            "dropout":      self.dropout,
            "fc_hidden":    self.fc_hidden,
            "seq_len":      self.seq_len,
            "best_epoch":   self.best_epoch_,
            "train_losses": self.train_losses_,
            "val_losses":   self.val_losses_,
        }, path)
        print(f"  [LSTM] Đã lưu trọng số vào: {path}")

    def load_weights(self, path: str) -> "LSTMForecaster":
        """Tải lại trọng số mô hình từ file .pt."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file weights: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        self.input_size  = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers  = checkpoint["num_layers"]
        self.dropout     = checkpoint["dropout"]
        self.fc_hidden   = checkpoint["fc_hidden"]
        self.seq_len     = checkpoint["seq_len"]
        self.best_epoch_ = checkpoint.get("best_epoch", 0)
        self.train_losses_ = checkpoint.get("train_losses", [])
        self.val_losses_   = checkpoint.get("val_losses", [])

        self.net = self._build_net(self.input_size)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        self.is_fitted = True
        print(f"  [LSTM] Đã tải trọng số từ: {path}")
        return self


# ==============================================================================
# 4. MAIN: Thực nghiệm huấn luyện và đánh giá nhanh
# ==============================================================================

if __name__ == "__main__":
    from src.preprocessor import time_series_split, TimeSeriesScaler
    from src.features import get_feature_columns_for_training

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "featured_gold_data.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy {data_path}. Hãy chạy features.py trước!")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
    print(f"[INFO] Dữ liệu: {df.shape[0]} phiên, {df.shape[1]} cột.")

    # 1. Phân tách chuỗi thời gian
    n  = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    train_df = df.iloc[:t1]
    val_df   = df.iloc[t1:t2]
    test_df  = df.iloc[t2:]
    feature_cols = get_feature_columns_for_training(df)

    print(f"[INFO] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} phiên")
    print(f"[INFO] Số đặc trưng X: {len(feature_cols)}")

    # 2. Chuẩn hóa (fit chỉ trên Train)
    scaler = TimeSeriesScaler("standard")
    scaler.fit(train_df, feature_cols)
    train_s = scaler.transform(train_df)
    val_s   = scaler.transform(val_df)
    test_s  = scaler.transform(test_df)

    X_train = train_s[feature_cols].values.astype(np.float32)
    y_train = train_df["target_return"].values.astype(np.float32)
    X_val   = val_s[feature_cols].values.astype(np.float32)
    y_val   = val_df["target_return"].values.astype(np.float32)
    X_test  = test_s[feature_cols].values.astype(np.float32)

    # 3. Khởi tạo và huấn luyện LSTM
    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=64,      # Nhỏ hơn để chạy nhanh trên CPU
        num_layers=2,
        dropout=0.2,
        fc_hidden=32,
        seq_len=30,
        batch_size=32,
        max_epochs=30,       # Demo nhanh
        learning_rate=1e-3,
        patience=10,
        target_type="return",
        verbose=True,
    )

    print("\n" + "="*70)
    print(f"{'HUẤN LUYỆN PYTORCH LSTM (TASK 8)':^70}")
    print("="*70)

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # 4. Đánh giá trên tập Test (dùng context để tránh cold start)
    print("\n[INFO] Đánh giá trên tập Test độc lập (2024 - nay)...")

    # Lấy seq_len ngày cuối của val làm context để warm-up LSTM state
    preds_ret = model.predict_with_context(X_val, X_test)
    curr_close_test = test_df["gold_close"].values
    y_test_close    = test_df["target_close"].values
    y_test_dir      = test_df["target_direction"].values

    pred_close = curr_close_test * (1.0 + preds_ret)
    pred_dir   = (preds_ret > 0).astype(int)

    rmse = float(np.sqrt(np.mean((pred_close - y_test_close) ** 2)))
    mae  = float(np.mean(np.abs(pred_close - y_test_close)))
    mape = float(np.mean(np.abs((pred_close - y_test_close) / y_test_close)) * 100.0)
    da   = float(np.mean(pred_dir == y_test_dir) * 100.0)

    print("\n" + "="*65)
    print(f"{'KẾT QUẢ ĐÁNH GIÁ LSTM TRÊN TẬP TEST':^65}")
    print("="*65)
    print(f"  RMSE               : ${rmse:.2f}")
    print(f"  MAE                : ${mae:.2f}")
    print(f"  MAPE               : {mape:.4f}%")
    print(f"  Directional Acc.   : {da:.2f}%")
    print(f"  Best Epoch         : {model.best_epoch_} / {model.max_epochs}")
    print(f"  Final Train Loss   : {model.train_losses_[-1]:.6f}")
    if model.val_losses_:
        print(f"  Best Val Loss      : {min(model.val_losses_):.6f}")
    print("="*65)

    # 5. Lưu trọng số
    weights_dir  = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    weights_path = os.path.join(weights_dir, "lstm_gold_weights.pt")
    os.makedirs(weights_dir, exist_ok=True)
    model.save_weights(weights_path)
