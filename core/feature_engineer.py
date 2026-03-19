"""core/feature_engineer.py — 20 technical features"""
import numpy as np
import pandas as pd


class FeatureEngineer:
    FEATURE_COLS = [
        'ret_1d','ret_2d','ret_3d','ret_5d','ret_10d','ret_21d',
        'sma_cross_5_20','sma_cross_10_50',
        'vol_5d','vol_10d','vol_21d','atr_14',
        'rsi_14','rsi_28','macd_hist',
        'bb_pct','vol_sma_ratio','obv_change','pos_52w','macd'
    ]

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df['close']; h = df['high']; l = df['low']
        v = df.get('volume', pd.Series(np.ones(len(df)), index=df.index))
        for p in [1,2,3,5,10,21]:
            df[f'ret_{p}d'] = c.pct_change(p)
        for w in [5,10,20,50]:
            df[f'sma_{w}'] = c.rolling(w).mean()
            df[f'ema_{w}'] = c.ewm(span=w).mean()
        df['sma_cross_5_20']  = df['sma_5']  / df['sma_20']  - 1
        df['sma_cross_10_50'] = df['sma_10'] / df['sma_50']  - 1
        for w in [5,10,21]:
            df[f'vol_{w}d'] = df['ret_1d'].rolling(w).std() * np.sqrt(252)
        df['atr_14'] = FeatureEngineer._atr(h, l, c, 14)
        df['rsi_14'] = FeatureEngineer._rsi(c, 14)
        df['rsi_28'] = FeatureEngineer._rsi(c, 28)
        df['macd'], df['macd_signal'] = FeatureEngineer._macd(c)
        df['macd_hist'] = df['macd'] - df['macd_signal']
        bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std()
        df['bb_pct'] = (c - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)
        df['vol_sma_ratio'] = v / (v.rolling(20).mean() + 1e-9)
        df['obv'] = (np.sign(df['ret_1d']) * v).cumsum()
        df['obv_change'] = df['obv'].pct_change(5)
        df['high_52w'] = h.rolling(252).max(); df['low_52w'] = l.rolling(252).min()
        df['pos_52w'] = (c - df['low_52w']) / (df['high_52w'] - df['low_52w'] + 1e-9)
        df.replace([np.inf,-np.inf], np.nan, inplace=True)
        return df

    @staticmethod
    def _rsi(s, p):
        d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); ls = (-d.clip(upper=0)).rolling(p).mean()
        return 100 - (100 / (1 + g/(ls+1e-9)))

    @staticmethod
    def _macd(s, fast=12, slow=26, signal=9):
        m = s.ewm(span=fast).mean() - s.ewm(span=slow).mean()
        return m, m.ewm(span=signal).mean()

    @staticmethod
    def _atr(h, l, c, p):
        tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        return tr.rolling(p).mean()
