"""backtest/engine.py — full backtest with costs, risk controls, metrics"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class BacktestConfig:
    initial_capital:    float = 100_000.0
    commission_pct:     float = 0.001
    slippage_pct:       float = 0.0005
    max_position_pct:   float = 0.20
    max_drawdown_limit: float = 0.15
    risk_free_rate:     float = 0.05
    target_vol:         float = 0.15
    use_vol_targeting:  bool  = True
    kelly_fraction:     float = 0.25
    stop_loss_pct:      float = 0.03
    take_profit_pct:    float = 0.09


class Metrics:
    @staticmethod
    def sharpe(r: pd.Series, rf=0.05, ppy=252):
        if len(r) < 2 or r.std() == 0: return 0.0
        return float((r.mean() - rf/ppy) / r.std() * np.sqrt(ppy))

    @staticmethod
    def sortino(r: pd.Series, rf=0.05, ppy=252):
        excess = r - rf/ppy
        ds = r[r<0].std()
        return float(excess.mean()/ds*np.sqrt(ppy)) if ds else 0.0

    @staticmethod
    def calmar(r: pd.Series, ppy=252):
        eq = (1+r).cumprod(); ann = (eq.iloc[-1])**(ppy/len(r))-1
        mdd = (eq/eq.cummax()-1).min()
        return float(abs(ann/mdd)) if mdd else 0.0

    @staticmethod
    def max_dd(r: pd.Series):
        eq = (1+r).cumprod()
        return float((eq/eq.cummax()-1).min()*100)

    @staticmethod
    def cagr(r: pd.Series, ppy=252):
        return float(((1+r).prod()**(ppy/len(r))-1)*100)

    @classmethod
    def full(cls, r: pd.Series, rf=0.05) -> dict:
        w = r[r>0]; l = r[r<0]
        return {
            "sharpe":        round(cls.sharpe(r,rf),3),
            "sortino":       round(cls.sortino(r,rf),3),
            "calmar":        round(cls.calmar(r),3),
            "cagr":          round(cls.cagr(r),2),
            "max_drawdown":  round(cls.max_dd(r),2),
            "total_return":  round(((1+r).prod()-1)*100,2),
            "ann_volatility":round(r.std()*np.sqrt(252)*100,2),
            "win_rate":      round(len(w)/max(len(w)+len(l),1)*100,2),
            "profit_factor": round(w.sum()/abs(l.sum()),3) if len(l)>0 else 0,
            "avg_win":       round(w.mean()*100,4) if len(w)>0 else 0,
            "avg_loss":      round(l.mean()*100,4) if len(l)>0 else 0,
            "num_trades":    int(len(w)+len(l)),
        }


class BacktestEngine:
    def __init__(self, config: BacktestConfig = None):
        self.cfg = config or BacktestConfig()

    def run_single(self, prices: pd.Series, signals: pd.Series, symbol="ASSET") -> dict:
        cfg = self.cfg
        cash = cfg.initial_capital; pos = 0.0; ep = None; ed = None; direction = 0
        equity_log = []; rets = []; trades = []; peak = cash
        prices = prices.reindex(signals.index).ffill()

        for i, date in enumerate(signals.index):
            price = prices.loc[date]; sig = int(signals.loc[date])
            if np.isnan(price): continue
            pv = cash + abs(pos)

            # stop-loss / take-profit
            if pos != 0 and ep:
                ret = (price - ep) / ep * direction
                if ret <= -cfg.stop_loss_pct or ret >= cfg.take_profit_pct:
                    reason = "stop_loss" if ret <= -cfg.stop_loss_pct else "take_profit"
                    pnl = abs(pos) * ret
                    cash += abs(pos) + pnl - abs(pos)*cfg.commission_pct
                    trades.append({"symbol":symbol,"entry_date":str(ed),"exit_date":str(date),
                        "entry_price":round(ep,4),"exit_price":round(price,4),
                        "direction":direction,"pnl":round(pnl,2),"exit_reason":reason})
                    pos=0; ep=None; ed=None; direction=0

            # signal change
            if sig != direction:
                if pos != 0 and ep:
                    xp = price*(1 - cfg.slippage_pct*direction)
                    pnl = abs(pos)*direction*(xp-ep)/ep
                    cash += abs(pos)+pnl - abs(pos)*cfg.commission_pct
                    trades.append({"symbol":symbol,"entry_date":str(ed),"exit_date":str(date),
                        "entry_price":round(ep,4),"exit_price":round(xp,4),
                        "direction":direction,"pnl":round(pnl,2),"exit_reason":"signal"})
                    pos=0; ep=None; ed=None; direction=0

                if sig != 0:
                    vol = 0.15
                    if cfg.use_vol_targeting:
                        raw = (cfg.target_vol/vol)*cfg.kelly_fraction
                        dollar = cash * min(raw, cfg.max_position_pct) * sig
                    else:
                        dollar = cash * cfg.max_position_pct * sig
                    cost = abs(dollar)*cfg.commission_pct
                    if abs(dollar)+cost <= cash:
                        ep = price*(1+cfg.slippage_pct*sig)
                        pos = dollar; ed = date; direction = sig
                        cash -= abs(dollar)+cost

            pv = cash + abs(pos)
            peak = max(peak, pv)
            if (pv-peak)/peak <= -cfg.max_drawdown_limit:
                if pos != 0: cash += abs(pos); pos=0
                break
            equity_log.append({"date":date,"equity":pv})
            if i > 0 and equity_log[-2]["equity"] > 0:
                rets.append((pv-equity_log[-2]["equity"])/equity_log[-2]["equity"])

        eq = pd.DataFrame(equity_log).set_index("date") if equity_log else pd.DataFrame(columns=["equity"])
        r  = pd.Series(rets)
        return {"equity_curve": eq, "returns": r,
                "trades": pd.DataFrame(trades),
                "metrics": Metrics.full(r, cfg.risk_free_rate) if len(r)>10 else {}}

    def run_portfolio(self, price_dict: Dict, signal_dict: Dict) -> dict:
        syms = list(price_dict.keys())
        all_dates = sorted(set().union(*[s.index for s in price_dict.values()]))
        prices_df  = pd.DataFrame({s:price_dict[s]  for s in syms},index=all_dates).ffill()
        signals_df = pd.DataFrame({s:signal_dict[s] for s in syms},index=all_dates).fillna(0)
        cash = self.cfg.initial_capital
        positions = {s:0.0 for s in syms}; entry_info = {s:None for s in syms}
        eq_log = []; all_trades = []

        for date in all_dates:
            pv = cash + sum(abs(v) for v in positions.values())
            for sym in syms:
                price = prices_df.loc[date,sym]; sig = int(signals_df.loc[date,sym])
                if np.isnan(price): continue
                pos=positions[sym]; info=entry_info[sym]; direction=int(np.sign(pos)) if pos!=0 else 0
                if pos!=0 and info:
                    ep,ed_ = info
                    ret=(price-ep)/ep*(direction)
                    if ret<=-self.cfg.stop_loss_pct or ret>=self.cfg.take_profit_pct:
                        pnl=abs(pos)*ret; cash+=abs(pos)+pnl-abs(pos)*self.cfg.commission_pct
                        all_trades.append({"symbol":sym,"date":str(date),"pnl":round(pnl,2),"direction":direction,"exit_reason":"sl/tp"})
                        positions[sym]=0; entry_info[sym]=None; direction=0
                if sig!=direction:
                    if positions[sym]!=0 and entry_info[sym]:
                        ep,_=entry_info[sym]; pnl=abs(positions[sym])*(price-ep)/ep*direction
                        cash+=abs(positions[sym])+pnl-abs(positions[sym])*self.cfg.commission_pct
                        all_trades.append({"symbol":sym,"date":str(date),"pnl":round(pnl,2),"direction":direction,"exit_reason":"signal"})
                        positions[sym]=0; entry_info[sym]=None
                    if sig!=0:
                        alloc=pv*self.cfg.max_position_pct/len(syms); cost=alloc*self.cfg.commission_pct
                        if cash>=alloc+cost:
                            positions[sym]=alloc*sig; entry_info[sym]=(price,date); cash-=alloc+cost
            total=cash+sum(abs(v) for v in positions.values())
            eq_log.append({"date":date,"equity":total})

        eq=pd.DataFrame(eq_log).set_index("date"); r=eq["equity"].pct_change().dropna()
        m=Metrics.full(r); m["symbols"]=syms; m["num_assets"]=len(syms)
        return {"equity_curve":eq,"returns":r,"trades":pd.DataFrame(all_trades),"metrics":m}

    def walk_forward(self, prices, signals, train_size=252, test_size=63):
        results=[]; idx=list(prices.index); n=len(idx)
        for start in range(0, n-train_size-test_size, test_size):
            ts=start+train_size; te=min(ts+test_size,n)
            res=self.run_single(prices.iloc[ts:te],signals.iloc[ts:te])
            results.append({"window":f"{idx[ts].date()} → {idx[te-1].date()}",
                "sharpe":res["metrics"].get("sharpe",0),"cagr":res["metrics"].get("cagr",0),
                "num_trades":res["metrics"].get("num_trades",0)})
        return pd.DataFrame(results)
