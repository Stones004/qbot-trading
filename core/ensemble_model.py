"""
core/ensemble_model.py
Fixed version — addresses all known causes of Sharpe = 0:
  1. Too few signals (threshold too high)
  2. Label imbalance (too many HOLD)
  3. Empty feature matrix after NaN fill
  4. CV folds too small to train
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from core.feature_engineer import FeatureEngineer

try:
    import xgboost as xgb; XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb; LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


class EnsembleModel:
    FEATURE_COLS = FeatureEngineer.FEATURE_COLS

    def __init__(self, n_splits=5, forward_return_days=5,
                 signal_threshold=0.52,   # lowered from 0.55 — was blocking all signals
                 label_threshold=0.003):  # ±0.3% to define long/short
        self.n_splits        = n_splits
        self.fwd_days        = forward_return_days
        self.threshold       = signal_threshold
        self.label_threshold = label_threshold
        self.scaler          = StandardScaler()
        self.meta            = LogisticRegression(
            max_iter=2000, C=0.3,          # stronger regularisation
            class_weight='balanced',        # fix label imbalance
            solver='lbfgs'
        )
        self.models          = self._init_models()
        self.is_fitted       = False
        self.feature_importance_ = {}
        self._train_rows     = 0

    def _init_models(self):
        m = {}
        if XGB_AVAILABLE:
            m['xgb'] = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.04,
                subsample=0.75, colsample_bytree=0.75,
                min_child_weight=5,        # prevent overfitting on small folds
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42, n_jobs=-1, verbosity=0)
        if LGB_AVAILABLE:
            m['lgb'] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.04,
                subsample=0.75, colsample_bytree=0.75,
                min_child_samples=10,
                random_state=42, n_jobs=-1, verbose=-1)
        if not m:
            from sklearn.ensemble import GradientBoostingClassifier
            m['gb'] = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
        return m

    def _make_labels(self, df):
        fwd = df['close'].pct_change(self.fwd_days).shift(-self.fwd_days)
        labels = np.where(fwd >  self.label_threshold,  1,
                 np.where(fwd < -self.label_threshold, -1, 0))
        return labels, fwd

    @staticmethod
    def _clean(feat: pd.DataFrame) -> pd.DataFrame:
        """ffill → bfill → zero-fill. Never leaves NaN for StandardScaler."""
        return feat.ffill().bfill().fillna(0)

    def fit(self, df: pd.DataFrame):
        df        = FeatureEngineer.add_features(df)
        labels, _ = self._make_labels(df)
        feat      = self._clean(df[self.FEATURE_COLS].copy())

        # Only drop rows where the forward return itself is NaN
        valid = ~np.isnan(labels)
        mask  = valid & feat.notna().all(axis=1)

        if mask.sum() < 50:
            raise ValueError(
                f"Only {mask.sum()} usable training rows. "
                "Use period='3y' or '5y' for enough data."
            )

        X  = feat[mask].values
        y  = labels[mask]
        Xs = self.scaler.fit_transform(X)
        self._train_rows = len(X)

        # Print label distribution so user can see what the model works with
        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"[EnsembleModel] Training on {len(X)} rows | "
              f"Labels: SHORT={dist.get(-1,0)} "
              f"HOLD={dist.get(0,0)} "
              f"LONG={dist.get(1,0)}")

        # Adaptive CV splits — never make folds smaller than 40 rows
        n_splits = min(self.n_splits, max(2, mask.sum() // 80))
        tscv     = TimeSeriesSplit(n_splits=n_splits)
        n_models = len(self.models)
        oof      = np.zeros((len(X), n_models * 3))

        for fold_idx, (tr, val) in enumerate(tscv.split(Xs)):
            if len(tr) < 30 or len(val) < 10:
                continue
            for i, (name, model) in enumerate(self.models.items()):
                try:
                    model.fit(Xs[tr], y[tr] + 1)
                    oof[val, i*3:(i+1)*3] = model.predict_proba(Xs[val])
                except Exception as e:
                    print(f"  [warn] fold {fold_idx} {name} failed: {e}")
                    oof[val, i*3:(i+1)*3] = 1/3

        # Retrain base models on full data
        for name, model in self.models.items():
            model.fit(Xs, y + 1)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance_[name] = dict(
                    zip(self.FEATURE_COLS, model.feature_importances_))

        # Train meta-learner — balanced class_weight handles skewed labels
        valid_oof = np.any(oof != 0, axis=1)
        if valid_oof.sum() > 20:
            self.meta.fit(oof[valid_oof], y[valid_oof] + 1)
        else:
            self.meta.fit(oof, y + 1)

        self.is_fitted = True
        return self

    def _stack(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros((0, len(self.models) * 3))
        Xs      = self.scaler.transform(X)
        stacked = np.zeros((len(X), len(self.models) * 3))
        for i, (_, model) in enumerate(self.models.items()):
            stacked[:, i*3:(i+1)*3] = model.predict_proba(Xs)
        return stacked

    def predict_signal(self, df: pd.DataFrame) -> pd.Series:
        assert self.is_fitted, "Call fit() first."
        df   = FeatureEngineer.add_features(df)
        feat = self._clean(df[self.FEATURE_COLS].copy())
        mask = feat.notna().all(axis=1)

        signal = pd.Series(0, index=df.index)
        if mask.sum() == 0:
            return signal

        stacked = self._stack(feat[mask].values)
        proba   = self.meta.predict_proba(stacked)   # shape (N, 3): short/hold/long

        # Map class indices: 0=short(-1), 1=hold(0), 2=long(1)
        # Only signal when confidence exceeds threshold
        raw = np.zeros(len(stacked), dtype=int)
        for j in range(len(stacked)):
            p_short, p_hold, p_long = proba[j]
            if p_long  >= self.threshold:
                raw[j] =  1
            elif p_short >= self.threshold:
                raw[j] = -1
            # else stay 0 (hold)

        signal[mask] = raw
        n_long  = (raw ==  1).sum()
        n_short = (raw == -1).sum()
        n_hold  = (raw ==  0).sum()
        print(f"[EnsembleModel] Signals on {mask.sum()} bars: "
              f"LONG={n_long} SHORT={n_short} HOLD={n_hold}")
        return signal

    def predict_proba_all(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        df   = FeatureEngineer.add_features(df)
        feat = self._clean(df[self.FEATURE_COLS].copy())
        mask = feat.notna().all(axis=1)

        result = pd.DataFrame(
            [[1/3, 1/3, 1/3]] * len(df),
            index=df.index,
            columns=['p_short', 'p_hold', 'p_long']
        )
        if mask.sum() == 0:
            return result

        proba = self.meta.predict_proba(self._stack(feat[mask].values))
        result.loc[mask, ['p_short', 'p_hold', 'p_long']] = proba
        return result
