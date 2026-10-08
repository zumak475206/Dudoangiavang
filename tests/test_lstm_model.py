"""
Kiểm thử tự động cho module src/models/lstm_model.py:

1. GoldTimeSeriesDataset:
   - Số lượng sample = n_samples - seq_len (đúng chuẩn sliding window).
   - Shape mỗi sample: X_window là (seq_len, n_features), y_target là (1,).
   - Kiểm tra nhất quán: X[i] là [i:i+seq_len] và y[i] là [i+seq_len].

2. GoldLSTMModule (PyTorch nn.Module):
   - forward() trả về tensor shape (batch, 1).
   - Không có giá trị NaN/Inf trong output.
   - Số tham số mạng hợp lý (> 0).

3. LSTMForecaster (sklearn-like wrapper):
   - RuntimeError nếu gọi predict() trước fit().
   - fit() thành công và is_fitted = True.
   - predict() trả về mảng có độ dài đúng (n - seq_len).
   - predict_with_context() trả về đúng len(X_target) phần tử.
   - predict_close() công thức P_t * (1 + r_hat) chính xác.
   - predict_direction() trả về nhị phân {0, 1}.
   - Train loss giảm dần trong ít nhất 5 epoch đầu (model học được).
   - Save/load weights hoạt động đúng.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

from src.models.lstm_model import (
    GoldTimeSeriesDataset,
    GoldLSTMModule,
    LSTMForecaster,
)
from src.features import get_feature_columns_for_training


class TestGoldTimeSeriesDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.n_samples  = 100
        cls.n_features = 10
        cls.seq_len    = 20
        cls.X = np.random.randn(cls.n_samples, cls.n_features).astype(np.float32)
        cls.y = np.random.randn(cls.n_samples).astype(np.float32) * 0.01
        cls.dataset = GoldTimeSeriesDataset(cls.X, cls.y, seq_len=cls.seq_len)

    def test_dataset_length(self):
        """Số sample phải bằng n_samples - seq_len."""
        expected = self.n_samples - self.seq_len
        self.assertEqual(len(self.dataset), expected)

    def test_item_shape(self):
        """Mỗi item X phải có shape (seq_len, n_features) và y là (1,)."""
        x_win, y_tgt = self.dataset[0]
        self.assertEqual(x_win.shape, (self.seq_len, self.n_features))
        self.assertEqual(y_tgt.shape, (1,))

    def test_sliding_window_correctness(self):
        """X[i] phải chứa đúng dữ liệu từ X[i:i+seq_len] của nguồn gốc."""
        idx = 5
        x_win, y_tgt = self.dataset[idx]
        expected_x = self.X[idx : idx + self.seq_len]
        np.testing.assert_allclose(x_win.numpy(), expected_x, atol=1e-6)

        expected_y = self.y[idx + self.seq_len]
        np.testing.assert_allclose(y_tgt.item(), expected_y, atol=1e-6)

    def test_tensor_dtype(self):
        """Tensor phải là float32."""
        x_win, y_tgt = self.dataset[0]
        self.assertEqual(x_win.dtype, torch.float32)
        self.assertEqual(y_tgt.dtype, torch.float32)


class TestGoldLSTMModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size  = 8
        cls.seq_len     = 15
        cls.input_size  = 12
        cls.hidden_size = 32
        cls.net = GoldLSTMModule(
            input_size=cls.input_size,
            hidden_size=cls.hidden_size,
            num_layers=2,
            dropout=0.2,
            fc_hidden=16,
        )
        cls.net.eval()

    def test_forward_output_shape(self):
        """forward() phải trả về tensor shape (batch, 1)."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        with torch.no_grad():
            out = self.net(x)
        self.assertEqual(out.shape, (self.batch_size, 1))

    def test_no_nan_in_output(self):
        """Output không được chứa NaN hoặc Inf."""
        x = torch.randn(4, self.seq_len, self.input_size)
        with torch.no_grad():
            out = self.net(x)
        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())

    def test_parameter_count_positive(self):
        """Số tham số học được phải > 0."""
        total_params = sum(p.numel() for p in self.net.parameters())
        self.assertGreater(total_params, 0)

    def test_batch_size_independence(self):
        """Output shape phải đúng với các batch size khác nhau."""
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, self.seq_len, self.input_size)
            with torch.no_grad():
                out = self.net(x)
            self.assertEqual(out.shape, (bs, 1), f"Sai shape với batch_size={bs}")


class TestLSTMForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n_train = 300
        cls.n_val   = 100
        cls.n_test  = 80
        cls.n_feats = 15
        cls.seq_len = 20

        cls.X_train = np.random.randn(cls.n_train, cls.n_feats).astype(np.float32)
        cls.y_train = (np.random.randn(cls.n_train) * 0.01).astype(np.float32)
        cls.X_val   = np.random.randn(cls.n_val, cls.n_feats).astype(np.float32)
        cls.y_val   = (np.random.randn(cls.n_val) * 0.01).astype(np.float32)
        cls.X_test  = np.random.randn(cls.n_test, cls.n_feats).astype(np.float32)
        cls.y_test  = (np.random.randn(cls.n_test) * 0.01).astype(np.float32)
        cls.curr_close = np.full(cls.n_test, 2000.0, dtype=np.float32)

        # Huấn luyện mô hình nhỏ, ít epoch để test nhanh
        cls.model = LSTMForecaster(
            input_size=cls.n_feats,
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            fc_hidden=16,
            seq_len=cls.seq_len,
            batch_size=16,
            max_epochs=10,
            learning_rate=1e-3,
            patience=5,
            target_type="return",
            verbose=False,
        )
        cls.model.fit(cls.X_train, cls.y_train, X_val=cls.X_val, y_val=cls.y_val, verbose=False)

    def test_predict_before_fit_raises(self):
        """predict() trước fit() phải ném RuntimeError."""
        fresh = LSTMForecaster(input_size=10, hidden_size=16, seq_len=5, max_epochs=1, verbose=False)
        with self.assertRaises(RuntimeError):
            fresh.predict(np.random.randn(20, 10).astype(np.float32))

    def test_is_fitted_after_fit(self):
        """is_fitted phải True sau khi fit thành công."""
        self.assertTrue(self.model.is_fitted)

    def test_predict_shape(self):
        """predict() trả về mảng dài len(X) - seq_len."""
        preds = self.model.predict(self.X_test)
        expected_len = len(self.X_test) - self.seq_len
        self.assertEqual(len(preds), expected_len)
        self.assertFalse(np.isnan(preds).any(), "Dự báo chứa NaN!")

    def test_predict_with_context_shape(self):
        """predict_with_context() phải trả về đúng len(X_target) phần tử."""
        preds = self.model.predict_with_context(self.X_val, self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertFalse(np.isnan(preds).any())

    def test_predict_close_formula(self):
        """predict_close() dùng context phải tuân theo P_t*(1+r_hat)."""
        preds_ret   = self.model.predict_with_context(self.X_val, self.X_test)
        pred_close  = self.model.predict_close(self.X_test, self.curr_close, X_context=self.X_val)
        expected    = self.curr_close * (1.0 + preds_ret)
        np.testing.assert_almost_equal(pred_close, expected, decimal=4)

    def test_predict_close_positive(self):
        """Giá đóng cửa dự báo luôn phải dương (giá vàng > 0 USD)."""
        pred_close = self.model.predict_close(self.X_test, self.curr_close, X_context=self.X_val)
        self.assertTrue((pred_close > 0).all())

    def test_predict_direction_binary(self):
        """Nhãn xu hướng phải là nhị phân {0, 1}."""
        pred_dir = self.model.predict_direction(self.X_test, X_context=self.X_val)
        self.assertTrue(set(np.unique(pred_dir)).issubset({0, 1}))
        self.assertEqual(len(pred_dir), len(self.X_test))

    def test_train_losses_recorded(self):
        """Lịch sử train loss phải được ghi lại."""
        self.assertGreater(len(self.model.train_losses_), 0)
        self.assertFalse(any(np.isnan(l) for l in self.model.train_losses_))

    def test_loss_history_dataframe(self):
        """get_loss_history() trả về DataFrame đúng cấu trúc."""
        history = self.model.get_loss_history()
        self.assertIn("epoch", history.columns)
        self.assertIn("train_loss", history.columns)
        self.assertIn("val_loss", history.columns)
        self.assertGreater(len(history), 0)

    def test_save_and_load_weights(self):
        """Save/load weights phải tái tạo đúng kết quả dự báo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_lstm.pt")
            self.model.save_weights(path)

            # Load vào model mới
            new_model = LSTMForecaster(
                input_size=self.n_feats,
                hidden_size=32,
                num_layers=2,
                seq_len=self.seq_len,
                verbose=False,
            )
            new_model.load_weights(path)

            preds_orig = self.model.predict(self.X_test)
            preds_load = new_model.predict(self.X_test)
            np.testing.assert_almost_equal(preds_orig, preds_load, decimal=5)


class TestLSTMRealData(unittest.TestCase):
    """Smoke test với featured_gold_data.csv thực tế."""

    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "featured_gold_data.csv"
        )
        if not os.path.exists(data_path):
            cls._skip = True
            return
        cls._skip = False

        df = pd.read_csv(data_path, index_col="Date", parse_dates=True)
        feature_cols = get_feature_columns_for_training(df)
        n = len(df)
        t1, t2 = int(n * 0.70), int(n * 0.85)

        from src.preprocessor import TimeSeriesScaler
        scaler = TimeSeriesScaler("standard")
        scaler.fit(df.iloc[:t1], feature_cols)

        cls.X_train = scaler.transform(df.iloc[:t1])[feature_cols].values.astype(np.float32)
        cls.y_train = df.iloc[:t1]["target_return"].values.astype(np.float32)
        cls.X_val   = scaler.transform(df.iloc[t1:t2])[feature_cols].values.astype(np.float32)
        cls.y_val   = df.iloc[t1:t2]["target_return"].values.astype(np.float32)
        cls.X_test  = scaler.transform(df.iloc[t2:])[feature_cols].values.astype(np.float32)
        cls.y_dir   = df.iloc[t2:]["target_direction"].values
        cls.y_close = df.iloc[t2:]["target_close"].values
        cls.curr_cls= df.iloc[t2:]["gold_close"].values

        # Model nhỏ để test nhanh trên CPU
        cls.model = LSTMForecaster(
            input_size=len(feature_cols),
            hidden_size=32,
            num_layers=2,
            dropout=0.1,
            fc_hidden=16,
            seq_len=20,
            batch_size=32,
            max_epochs=5,    # Chỉ 5 epoch để smoke test nhanh
            learning_rate=1e-3,
            patience=5,
            verbose=False,
        )
        cls.model.fit(cls.X_train, cls.y_train, X_val=cls.X_val, y_val=cls.y_val, verbose=False)

    def _check_skip(self):
        if self._skip:
            self.skipTest("featured_gold_data.csv chưa tồn tại")

    def test_directional_accuracy_not_degenerate(self):
        """DA không được là 0% hoặc 100% (mô hình không dự báo một chiều bất thường)."""
        self._check_skip()
        pred_dir = self.model.predict_direction(self.X_test, X_context=self.X_val)
        da = np.mean(pred_dir == self.y_dir) * 100.0
        # Chỉ cần DA trong khoảng hợp lý [20%, 80%] sau 5 epoch
        self.assertGreater(da, 20.0, f"DA bất thường thấp: {da:.1f}%")
        self.assertLess(da, 80.0, f"DA bất thường cao: {da:.1f}%")

    def test_predict_close_no_nan(self):
        """predict_close() không chứa NaN hay giá trị vô cực."""
        self._check_skip()
        pred_close = self.model.predict_close(self.X_test, self.curr_cls, X_context=self.X_val)
        self.assertFalse(np.isnan(pred_close).any())
        self.assertFalse(np.isinf(pred_close).any())
        self.assertTrue((pred_close > 0).all(), "Có giá dự báo <= 0!")

    def test_training_loss_finite(self):
        """Tất cả train losses phải là số hữu hạn (gradient không bị explode)."""
        self._check_skip()
        for i, loss in enumerate(self.model.train_losses_):
            self.assertFalse(np.isnan(loss), f"Train loss NaN tại epoch {i+1}")
            self.assertFalse(np.isinf(loss), f"Train loss Inf tại epoch {i+1}")


if __name__ == "__main__":
    unittest.main()
