"""data/fetcher.py — yfinance with parquet cache + GBM fallback"""
import os, numpy as np, pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

try:
    import yfinance as yf; YF = True
except ImportError:
    YF = False


class DataFetcher:
    # Approximate bar counts for "max" period (as of 2026):
    # AAPL:    ~11,000 bars (1980-2026) — 45 years of regimes
    # MSFT:    ~10,000 bars (1986-2026)
    # BTC-USD: ~3,500  bars (2014-2026) — all crypto regimes
    # GC=F:    ~13,000 bars (1975-2026) — gold through every macro cycle

    @staticmethod
    def fetch(symbol: str, period: str = "2y", interval: str = "1d",
              force_refresh: bool = False) -> pd.DataFrame:
        cache_path = CACHE_DIR / f"{symbol.replace('/','-')}_{period}.parquet"
        if cache_path.exists() and not force_refresh:
            try:
                df = pd.read_parquet(cache_path)
                print(f"[cache] {symbol} ({len(df)} rows)")
                return df
            except Exception:
                pass
        if YF:
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                df.columns = [c.lower() for c in df.columns]
                df.index   = pd.to_datetime(df.index).tz_localize(None)
                df = df[['open','high','low','close','volume']].dropna()
                df.to_parquet(cache_path)
                print(f"[yfinance] {symbol} ({len(df)} rows)")
                return df
            except Exception as e:
                print(f"[yfinance] failed for {symbol}: {e}")
        print(f"[synthetic] generating data for {symbol}")
        return DataFetcher.synthetic(symbol=symbol)

    @staticmethod
    def fetch_multi(symbols, period="2y"):
        return {s: DataFetcher.fetch(s, period) for s in symbols}

    @staticmethod
    def synthetic(symbol="SYN", n=600, seed=42, mu=0.0003, sigma=0.015):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start="2022-01-01", periods=n)
        rets  = rng.normal(mu, sigma, n)
        rets += rng.choice([0,1],n,p=[0.97,0.03])*rng.normal(0,0.04,n)
        prices = 100*np.exp(np.cumsum(rets))
        noise  = rng.uniform(0.001,0.003,n)
        return pd.DataFrame({
            'open': np.roll(prices,1), 'high': prices*(1+noise),
            'low':  prices*(1-noise),  'close': prices,
            'volume': rng.integers(500_000,5_000_000,n).astype(float)
        }, index=dates)

UNIVERSE = {
    "stocks":  ["AAPL","MSFT","GOOGL","NVDA","AMZN","META","TSLA"],
    "crypto":  ["BTC-USD","ETH-USD","SOL-USD"],
    "forex":   ["EURUSD=X","GBPUSD=X","USDJPY=X"],
    "futures": ["ES=F","NQ=F","GC=F","CL=F"],
}
ALL_SYMBOLS = [s for grp in UNIVERSE.values() for s in grp]