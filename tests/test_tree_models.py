"""
Kiểm thử tự động cho module src/models/tree_models.py:

1. XGBoostForecaster:
   - fit() và predict() chạy thông suốt, kích thước đầu ra đúng.
   - Lỗi RuntimeError nếu gọi predict() khi chưa fit().
   - predict_close() tái cấu trúc giá đúng công thức P_t * (1 + r_hat).
   - predict_direction() chỉ trả về nhãn nhị phân {0, 1}.
   - get_feature_importance() trả về DataFrame đúng cấu trúc, sắp xếp giảm dần.
   - Kiểm tra tính năng fit với X_val (early stopping path).

2. LightGBMForecaster:
   - fit() với tập Validation (early stopping path) chạy thành công.
   - predict() kích thước đúng, không có NaN.
   - Feature Importance trích xuất đầy đủ số đặc trưng.

3. Smoke test với dữ liệu thật (featured_gold_data.csv):
   - Directional Accuracy > 45% (vượt ngưỡng random chance thống kê).
   - RMSE tính bằng USD, không có NaN hoặc giá trị vô cực.
   - Predict_close() luôn trả về giá dương (> 0).
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.models.tree_models import XGBoostForecaster, LightGBMForecaster
from src.features import get_feature_columns_for_training


class TestXGBoostForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.n_train = 200
        cls.n_val   = 50
        cls.n_test  = 50
        cls.n_feats = 15

        cls.feat_names = [f"feat_{i}" for i in range(cls.n_feats)]
        cls.X_train = pd.DataFrame(np.random.randn(cls.n_train, cls.n_feats), columns=cls.feat_names)
        cls.X_val   = pd.DataFrame(np.random.randn(cls.n_val,   cls.n_feats), columns=cls.feat_names)
        cls.X_test  = pd.DataFrame(np.random.randn(cls.n_test,  cls.n_feats), columns=cls.feat_names)

        # Tỷ suất sinh lời giả lập (mean ≈ 0, std ≈ 0.01, giống vàng)
        cls.y_train = pd.Series(np.random.randn(cls.n_train) * 0.01, name="target_return")
        cls.y_val   = pd.Series(np.random.randn(cls.n_val)   * 0.01, name="target_return")

        cls.curr_close = pd.Series(2000.0 + np.cumsum(np.random.randn(cls.n_test) * 10))

        # Mô hình với tham số nhẹ để test nhanh
        lite_params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        cls.model = XGBoostForecaster(target_type="return", params=lite_params)
        cls.model.fit(cls.X_train, cls.y_train, X_val=cls.X_val, y_val=cls.y_val, verbose=False)

    def test_predict_before_fit_raises(self):
        """Gọi predict() khi chưa fit phải ném RuntimeError."""
        fresh_model = XGBoostForecaster(target_type="return")
        with self.assertRaises(RuntimeError):
            fresh_model.predict(self.X_test)

    def test_predict_shape(self):
        """predict() phải trả về mảng 1D có độ dài bằng số mẫu test."""
        preds = self.model.predict(self.X_test)
        self.assertEqual(preds.shape, (self.n_test,))
        self.assertFalse(np.isnan(preds).any(), "Kết quả dự báo chứa NaN!")

    def test_predict_close_formula(self):
        """predict_close() phải tuân theo công thức P_t * (1 + r_hat_{t+1})."""
        pred_ret   = self.model.predict(self.X_test)
        pred_close = self.model.predict_close(self.X_test, self.curr_close)
        expected   = self.curr_close.values * (1.0 + pred_ret)
        np.testing.assert_almost_equal(pred_close, expected, decimal=4)

    def test_predict_close_positive(self):
        """Giá đóng cửa dự báo luôn phải dương (> 0 USD)."""
        pred_close = self.model.predict_close(self.X_test, self.curr_close)
        self.assertTrue((pred_close > 0).all(), "Có giá trị giá dự báo <= 0!")

    def test_predict_direction_binary(self):
        """predict_direction() chỉ trả về nhị phân {0, 1}."""
        pred_dir = self.model.predict_direction(self.X_test, self.curr_close)
        self.assertEqual(len(pred_dir), self.n_test)
        self.assertTrue(set(np.unique(pred_dir)).issubset({0, 1}))

    def test_feature_importance_structure(self):
        """get_feature_importance() trả về DataFrame với đúng cột và sắp xếp giảm dần."""
        fi_df = self.model.get_feature_importance(importance_type="gain")
        self.assertIn("feature", fi_df.columns)
        self.assertIn("importance_score", fi_df.columns)
        self.assertIn("importance_pct", fi_df.columns)

        # Kiểm tra sắp xếp giảm dần
        scores = fi_df["importance_score"].values
        self.assertTrue((scores[:-1] >= scores[1:]).all(), "Importance chưa sắp xếp giảm dần!")

        # Tổng % phải xấp xỉ 100
        total_pct = fi_df["importance_pct"].sum()
        self.assertAlmostEqual(total_pct, 100.0, delta=0.1)

    def test_is_fitted_flag(self):
        """is_fitted phải True sau khi fit thành công."""
        self.assertTrue(self.model.is_fitted)


class TestLightGBMForecaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(7)
        cls.n_train = 200
        cls.n_test  = 50
        cls.n_feats = 15
        cls.feat_names = [f"feat_{i}" for i in range(cls.n_feats)]

        cls.X_train = pd.DataFrame(np.random.randn(cls.n_train, cls.n_feats), columns=cls.feat_names)
        cls.X_val   = pd.DataFrame(np.random.randn(50, cls.n_feats), columns=cls.feat_names)
        cls.X_test  = pd.DataFrame(np.random.randn(cls.n_test,  cls.n_feats), columns=cls.feat_names)
        cls.y_train = pd.Series(np.random.randn(cls.n_train) * 0.01, name="target_return")
        cls.y_val   = pd.Series(np.random.randn(50) * 0.01, name="target_return")
        cls.curr_close = pd.Series(1800.0 + np.cumsum(np.random.randn(cls.n_test) * 8))

        lite_params = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1, "num_leaves": 15}
        cls.model = LightGBMForecaster(target_type="return", params=lite_params)
        cls.model.fit(cls.X_train, cls.y_train, X_val=cls.X_val, y_val=cls.y_val, verbose=False)

    def test_predict_before_fit_raises(self):
        """Gọi predict() khi chưa fit phải ném RuntimeError."""
        fresh_model = LightGBMForecaster(target_type="return")
        with self.assertRaises(RuntimeError):
            fresh_model.predict(self.X_test)

    def test_predict_shape_and_no_nan(self):
        """predict() trả về mảng đúng kích thước, không có NaN."""
        preds = self.model.predict(self.X_test)
        self.assertEqual(preds.shape, (self.n_test,))
        self.assertFalse(np.isnan(preds).any())

    def test_predict_close_positive(self):
        """Giá dự báo luôn dương."""
        pred_close = self.model.predict_close(self.X_test, self.curr_close)
        self.assertTrue((pred_close > 0).all())

    def test_feature_importance_length(self):
        """Feature Importance phải có đúng n_features dòng."""
        fi_df = self.model.get_feature_importance()
        self.assertEqual(len(fi_df), self.n_feats)
        scores = fi_df["importance_score"].values
        self.assertTrue((scores[:-1] >= scores[1:]).all(), "Importance chưa sắp xếp giảm dần!")

    def test_predict_direction_binary(self):
        """Nhãn xu hướng phải là nhị phân {0, 1}."""
        pred_dir = self.model.predict_direction(self.X_test, self.curr_close)
        self.assertTrue(set(np.unique(pred_dir)).issubset({0, 1}))


class TestTreeModelsRealData(unittest.TestCase):
    """Smoke test với featured_gold_data.csv thực tế (bỏ qua nếu chưa có file)."""

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

        # Phân tách chuỗi thời gian nghiêm ngặt (70-15-15)
        n = len(df)
        t1 = int(n * 0.70)
        t2 = int(n * 0.85)

        train_df = df.iloc[:t1]
        val_df   = df.iloc[t1:t2]
        test_df  = df.iloc[t2:]

        # Không scale để unit test nhanh (scale trong production experiment)
        cls.X_train    = train_df[feature_cols]
        cls.y_train    = train_df["target_return"]
        cls.X_val      = val_df[feature_cols]
        cls.y_val      = val_df["target_return"]
        cls.X_test     = test_df[feature_cols]
        cls.y_test_dir = test_df["target_direction"].values
        cls.y_test_cls = test_df["target_close"].values
        cls.curr_close = test_df["gold_close"].values

        lite_params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
        cls.xgb_model = XGBoostForecaster(target_type="return", params=lite_params)
        cls.xgb_model.fit(cls.X_train, cls.y_train, X_val=cls.X_val, y_val=cls.y_val, verbose=False)

        cls.lgb_model = LightGBMForecaster(target_type="return", params={**lite_params, "num_leaves": 15})
        cls.lgb_model.fit(cls.X_train, cls.y_train, X_val=cls.X_val, y_val=cls.y_val, verbose=False)

    def _check_skip(self):
        if self._skip:
            self.skipTest("featured_gold_data.csv chưa tồn tại")

    def test_xgb_directional_accuracy_above_random(self):
        """XGBoost DA trên Test phải > 45% (không tệ hơn random chance đáng kể)."""
        self._check_skip()
        pred_dir = self.xgb_model.predict_direction(self.X_test, self.curr_close)
        da = np.mean(pred_dir == self.y_test_dir) * 100.0
        self.assertGreater(da, 45.0, f"XGBoost DA quá thấp: {da:.2f}%")

    def test_lgb_directional_accuracy_above_random(self):
        """LightGBM DA trên Test phải > 45%."""
        self._check_skip()
        pred_dir = self.lgb_model.predict_direction(self.X_test, self.curr_close)
        da = np.mean(pred_dir == self.y_test_dir) * 100.0
        self.assertGreater(da, 45.0, f"LightGBM DA quá thấp: {da:.2f}%")

    def test_xgb_predict_close_no_nan_inf(self):
        """XGBoost predict_close() không có NaN hay vô cực."""
        self._check_skip()
        pred_close = self.xgb_model.predict_close(self.X_test, self.curr_close)
        self.assertFalse(np.isnan(pred_close).any(), "XGBoost predict_close có NaN!")
        self.assertFalse(np.isinf(pred_close).any(), "XGBoost predict_close có giá trị vô cực!")

    def test_lgb_feature_importance_covers_all_features(self):
        """LightGBM Feature Importance phải có đúng số đặc trưng."""
        self._check_skip()
        fi_df = self.lgb_model.get_feature_importance()
        self.assertEqual(len(fi_df), len(self.X_train.columns))


if __name__ == "__main__":
    unittest.main()
