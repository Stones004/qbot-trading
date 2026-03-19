"""
core/ensemble_model.py
Fixed for mean-reversion markets (AAPL 2021-2026).

Key insight from diagnostics:
- Features have strong NEGATIVE correlation with 5-day forward returns
- This means: when RSI is high / price is above SMA / momentum is strong
  → price tends to FALL over next 5 days (mean reversion)
- The model must learn this counter-trend pattern

Fixes applied:
1. Contrarian label encoding: positive features → SHORT signal
   (handled by letting XGBoost learn the negative correlations naturally)
2. Removed class_weight='balanced' — was over-correcting and collapsing meta-learner
3. Raised C from 0.3 to 1.0 — less regularisation so meta-learner can actually learn
4. Added direct base-model voting fallback if meta-learner collapses
5. Reduced min_child_weight so trees can split on mean-reversion patterns
"""
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

    def __init__(self, n_splits=5, forward_return_days=5,
                 signal_threshold=0.40,
                 label_threshold=0.003):
        self.n_splits        = n_splits
        self.fwd_days        = forward_return_days
        self.threshold       = signal_threshold
        self.label_threshold = label_threshold
        self.scaler          = StandardScaler()
        # No class_weight, higher C — was collapsing to uniform predictions
        self.meta = LogisticRegression(
            max_iter=2000, C=1.0, solver='lbfgs'
        )
        self.models          = self._init_models()
        self.is_fitted       = False
        self.feature_importance_ = {}
        self._train_rows     = 0

    def _init_models(self):
        m = {}
        if XGB_AVAILABLE:
            m['xgb'] = xgb.XGBClassifier(
                n_estimators=300, max_depth=3,   # shallower = less overfit
                learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=3,              # reduced: 878 rows can handle finer splits
                gamma=0.1,                       # minimum gain to split — prevents noise splits
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42, n_jobs=-1, verbosity=0)
        if LGB_AVAILABLE:
            m['lgb'] = lgb.LGBMClassifier(
                n_estimators=300, max_depth=3,
                learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=15,
                reg_alpha=0.1,                   # L1 regularisation
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
        return feat.ffill().bfill().fillna(0)

    def fit(self, df: pd.DataFrame):
        df        = FeatureEngineer.add_features(df)
        labels, _ = self._make_labels(df)
        feat      = self._clean(df[self.FEATURE_COLS].copy())

        valid = ~np.isnan(labels)
        mask  = valid & feat.notna().all(axis=1)

        if mask.sum() < 50:
            raise ValueError(
                f"Only {mask.sum()} usable rows. Use period='3y' or '5y'."
            )

        X  = feat[mask].values
        y  = labels[mask]
        Xs = self.scaler.fit_transform(X)
        self._train_rows = len(X)

        unique, counts = np.unique(y, return_counts=True)
        dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"[EnsembleModel] Training on {len(X)} rows | "
              f"SHORT={dist.get(-1,0)} HOLD={dist.get(0,0)} LONG={dist.get(1,0)}")

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
                    print(f"  [warn] fold {fold_idx} {name}: {e}")
                    oof[val, i*3:(i+1)*3] = 1/3

        for name, model in self.models.items():
            model.fit(Xs, y + 1)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance_[name] = dict(
                    zip(self.FEATURE_COLS, model.feature_importances_))

        valid_oof = np.any(oof != 0, axis=1)
        n_valid   = valid_oof.sum()
        if n_valid > 20:
            self.meta.fit(oof[valid_oof], y[valid_oof] + 1)
        else:
            self.meta.fit(oof, y + 1)

        # Sanity check: does meta-learner output spread?
        meta_proba = self.meta.predict_proba(oof[valid_oof] if n_valid > 20 else oof)
        max_conf   = meta_proba.max(axis=1).mean()
        print(f"[EnsembleModel] Meta-learner avg max confidence: {max_conf:.3f} "
              f"({'OK' if max_conf > 0.38 else 'COLLAPSED — using direct vote'})")
        self._meta_collapsed = max_conf < 0.38
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

    def _predict_proba_from_stack(self, stacked: np.ndarray) -> np.ndarray:
        """
        If meta-learner collapsed, fall back to averaging base model probabilities.
        This bypasses the broken logistic regression layer.
        """
        if self._meta_collapsed:
            # Average base model probabilities directly
            n_models = len(self.models)
            avg = np.zeros((len(stacked), 3))
            for i in range(n_models):
                avg += stacked[:, i*3:(i+1)*3]
            return avg / n_models
        return self.meta.predict_proba(stacked)

    def predict_signal(self, df: pd.DataFrame) -> pd.Series:
        assert self.is_fitted
        df   = FeatureEngineer.add_features(df)
        feat = self._clean(df[self.FEATURE_COLS].copy())
        mask = feat.notna().all(axis=1)

        signal = pd.Series(0, index=df.index)
        if mask.sum() == 0:
            return signal

        stacked = self._stack(feat[mask].values)
        proba   = self._predict_proba_from_stack(stacked)

        raw = np.zeros(len(stacked), dtype=int)
        for j in range(len(stacked)):
            p_short, p_hold, p_long = proba[j]
            if   p_long  >= self.threshold: raw[j] =  1
            elif p_short >= self.threshold: raw[j] = -1

        signal[mask] = raw
        n_long  = int((raw ==  1).sum())
        n_short = int((raw == -1).sum())
        n_hold  = int((raw ==  0).sum())
        mode    = "direct avg" if self._meta_collapsed else "meta-learner"
        print(f"[EnsembleModel] Signals ({mode}): "
              f"LONG={n_long} SHORT={n_short} HOLD={n_hold}")
        return signal

    def predict_proba_all(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted
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

        stacked = self._stack(feat[mask].values)
        proba   = self._predict_proba_from_stack(stacked)
        result.loc[mask, ['p_short', 'p_hold', 'p_long']] = proba
        return result