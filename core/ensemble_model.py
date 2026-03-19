"""core/ensemble_model.py — XGBoost + LightGBM + meta-learner stacked ensemble"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
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

    def __init__(self, n_splits=5, forward_return_days=5, signal_threshold=0.55):
        self.n_splits   = n_splits
        self.fwd_days   = forward_return_days
        self.threshold  = signal_threshold
        self.scaler     = StandardScaler()
        self.meta       = LogisticRegression(max_iter=1000, C=0.5)
        self.models     = self._init_models()
        self.is_fitted  = False
        self.feature_importance_ = {}

    def _init_models(self):
        m = {}
        if XGB_AVAILABLE:
            m['xgb'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1, verbosity=0)
        if LGB_AVAILABLE:
            m['lgb'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbose=-1)
        if not m:
            from sklearn.ensemble import RandomForestClassifier
            m['rf'] = RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        return m

    def _make_labels(self, df):
        fwd = df['close'].pct_change(self.fwd_days).shift(-self.fwd_days)
        return np.where(fwd > 0.002, 1, np.where(fwd < -0.002, -1, 0)), fwd

    @staticmethod
    def _clean(feat: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs from long rolling windows (252-day high/low etc).
        ffill -> bfill -> zero-fill so StandardScaler never sees NaN."""
        return feat.ffill().bfill().fillna(0)

    def fit(self, df: pd.DataFrame):
        df        = FeatureEngineer.add_features(df)
        labels, _ = self._make_labels(df)
        feat      = self._clean(df[self.FEATURE_COLS].copy())

        # exclude only the last fwd_days rows where label is NaN
        valid = ~np.isnan(labels)
        mask  = valid & feat.notna().all(axis=1)

        if mask.sum() < 50:
            raise ValueError(
                f"Only {mask.sum()} usable rows after cleaning. "
                "Use a longer data period — '2y' or more is recommended."
            )

        X  = feat[mask].values
        y  = labels[mask]
        Xs = self.scaler.fit_transform(X)

        # Reduce CV splits automatically for small datasets
        n_splits = min(self.n_splits, max(2, mask.sum() // 60))
        tscv     = TimeSeriesSplit(n_splits=n_splits)
        n_models = len(self.models)
        oof      = np.zeros((len(X), n_models * 3))

        for _, (tr, val) in enumerate(tscv.split(Xs)):
            for i, (name, model) in enumerate(self.models.items()):
                model.fit(Xs[tr], y[tr] + 1)
                oof[val, i*3:(i+1)*3] = model.predict_proba(Xs[val])

        for name, model in self.models.items():
            model.fit(Xs, y + 1)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance_[name] = dict(
                    zip(self.FEATURE_COLS, model.feature_importances_))

        self.meta.fit(oof, y + 1)
        self.is_fitted = True
        print(f"[EnsembleModel] fit on {len(X)} rows | "
              f"class dist: {dict(zip(*np.unique(y, return_counts=True)))}")
        return self

    def _stack(self, X: np.ndarray) -> np.ndarray:
        """Stack base model probabilities. Handles empty X gracefully."""
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

        raw = self.meta.predict(self._stack(feat[mask].values)) - 1
        signal[mask] = raw
        return signal

    def predict_proba_all(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        df   = FeatureEngineer.add_features(df)
        feat = self._clean(df[self.FEATURE_COLS].copy())
        mask = feat.notna().all(axis=1)

        # default neutral probability
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
