"""
core/feature_engineer.py
Revised feature set — removes pos_52w (252-bar lookback, NaN-heavy on <5y data)
and replaces it with short-lookback features that work on any dataset size.

All features use max 50-bar lookback so they're valid from bar 50 onwards.
"""
import numpy as np
import pandas as pd


class FeatureEngineer:
    FEATURE_COLS = [
        # ── Returns (momentum) ──────────────────────────────────────────
        'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_21d',

        # ── Trend ────────────────────────────────────────────────────────
        'sma_cross_5_20',    # short vs medium trend
        'sma_cross_10_50',   # medium vs long trend
        'price_vs_sma20',    # price position relative to 20-day MA

        # ── Volatility ───────────────────────────────────────────────────
        'vol_5d', 'vol_21d',
        'vol_regime',        # current vol vs recent vol (expanding/contracting)
        'atr_pct',           # ATR as % of price (normalised)

        # ── Oscillators ──────────────────────────────────────────────────
        'rsi_14', 'rsi_28',
        'macd_hist',
        'macd_hist_change',  # momentum of MACD histogram

        # ── Mean reversion ───────────────────────────────────────────────
        'bb_pct',            # Bollinger Band percentile
        'bb_width',          # band width (measures squeeze/expansion)

        # ── Volume ───────────────────────────────────────────────────────
        'vol_sma_ratio',     # volume vs 20-day avg
        'obv_change',        # OBV 5-day % change
    ]

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df  = df.copy()
        c   = df['close']
        h   = df['high']
        l   = df['low']
        v   = df.get('volume', pd.Series(np.ones(len(df)), index=df.index))

        # Returns
        for p in [1, 3, 5, 10, 21]:
            df[f'ret_{p}d'] = c.pct_change(p)

        # Moving averages
        for w in [5, 10, 20, 50]:
            df[f'sma_{w}'] = c.rolling(w).mean()
            df[f'ema_{w}'] = c.ewm(span=w).mean()

        # Trend crossovers
        df['sma_cross_5_20']  = df['sma_5']  / (df['sma_20']  + 1e-9) - 1
        df['sma_cross_10_50'] = df['sma_10'] / (df['sma_50']  + 1e-9) - 1
        df['price_vs_sma20']  = c / (df['sma_20'] + 1e-9) - 1

        # Volatility (short lookback only — no 252-bar windows)
        df['ret_1d_raw']  = c.pct_change(1)
        df['vol_5d']      = df['ret_1d_raw'].rolling(5).std()  * np.sqrt(252)
        df['vol_21d']     = df['ret_1d_raw'].rolling(21).std() * np.sqrt(252)
        df['vol_regime']  = df['vol_5d'] / (df['vol_21d'] + 1e-9)  # >1 = expanding vol

        atr               = FeatureEngineer._atr(h, l, c, 14)
        df['atr_pct']     = atr / (c + 1e-9)   # ATR as % of price, scale-free

        # Oscillators
        df['rsi_14']       = FeatureEngineer._rsi(c, 14)
        df['rsi_28']       = FeatureEngineer._rsi(c, 28)
        macd, signal       = FeatureEngineer._macd(c)
        df['macd_hist']    = macd - signal
        df['macd_hist_change'] = df['macd_hist'].diff(3)   # histogram momentum

        # Bollinger Bands (20-bar, fully valid from bar 20)
        bb_mid            = c.rolling(20).mean()
        bb_std            = c.rolling(20).std()
        bb_range          = 4 * bb_std + 1e-9
        df['bb_pct']      = (c - (bb_mid - 2 * bb_std)) / bb_range
        df['bb_width']    = (4 * bb_std) / (bb_mid + 1e-9)  # relative band width

        # Volume
        vol_sma           = v.rolling(20).mean()
        df['vol_sma_ratio'] = v / (vol_sma + 1e-9)
        obv               = (np.sign(df['ret_1d_raw']) * v).cumsum()
        df['obv_change']  = obv.pct_change(5)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(s: pd.Series, period: int) -> pd.Series:
        delta  = s.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-9)))

    @staticmethod
    def _macd(s: pd.Series, fast=12, slow=26, signal=9):
        line   = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
        sig    = line.ewm(span=signal).mean()
        return line, sig

    @staticmethod
    def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()