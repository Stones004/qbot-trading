"""live/paper_engine.py — paper trading with state persistence"""
import os, json, uuid
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Callable


@dataclass
class Position:
    symbol:       str
    direction:    int
    entry_price:  float
    qty:          float
    entry_time:   str
    current_price:float = 0.0
    unrealized_pnl:float = 0.0
    unrealized_pct:float = 0.0


@dataclass
class ClosedTrade:
    trade_id:    str
    symbol:      str
    direction:   int
    entry_price: float
    exit_price:  float
    qty:         float
    pnl:         float
    pnl_pct:     float
    entry_time:  str
    exit_time:   str
    exit_reason: str


class PaperEngine:
    STATE_FILE = "paper_state.json"

    def __init__(self, initial_capital=100_000.0, commission_pct=0.001,
                 slippage_pct=0.0005, stop_loss_pct=0.03, take_profit_pct=0.09,
                 broadcast_fn: Optional[Callable] = None):
        self.initial_capital = initial_capital
        self.commission      = commission_pct
        self.slippage        = slippage_pct
        self.sl_pct          = stop_loss_pct
        self.tp_pct          = take_profit_pct
        self.broadcast       = broadcast_fn or (lambda x: None)
        self.cash            = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[ClosedTrade] = []
        self.equity_log: List[dict] = []
        self._load()

    def _save(self):
        state = {
            "cash":          self.cash,
            "positions":     {k: asdict(v) for k,v in self.positions.items()},
            "closed_trades": [asdict(t) for t in self.closed_trades[-1000:]],
            "equity_log":    self.equity_log[-5000:],
        }
        with open(self.STATE_FILE,"w") as f:
            json.dump(state, f, default=str, indent=2)

    def _load(self):
        if not os.path.exists(self.STATE_FILE): return
        try:
            with open(self.STATE_FILE) as f: s = json.load(f)
            self.cash = s.get("cash", self.initial_capital)
            for sym, d in s.get("positions",{}).items():
                self.positions[sym] = Position(**d)
            self.closed_trades = [ClosedTrade(**t) for t in s.get("closed_trades",[])]
            self.equity_log    = s.get("equity_log", [])
        except Exception as e:
            print(f"[paper] could not load state: {e}")

    def on_bar(self, symbol: str, price: float, signal: int,
               portfolio_pct: float = 0.15) -> dict:
        # mark-to-market
        if symbol in self.positions:
            p = self.positions[symbol]
            p.current_price    = price
            p.unrealized_pnl   = (price - p.entry_price) * p.qty * p.direction
            p.unrealized_pct   = (price - p.entry_price) / p.entry_price * p.direction

        current_dir = self.positions[symbol].direction if symbol in self.positions else 0
        action = None

        # check sl/tp on open position
        if symbol in self.positions:
            p = self.positions[symbol]
            ret = (price - p.entry_price)/p.entry_price * p.direction
            if ret <= -self.sl_pct:
                action = self._close(symbol, price, "stop_loss")
                current_dir = 0
            elif ret >= self.tp_pct:
                action = self._close(symbol, price, "take_profit")
                current_dir = 0

        # signal-driven trade
        if signal != current_dir:
            if symbol in self.positions:
                action = self._close(symbol, price, "signal")
            if signal != 0:
                pv    = self.portfolio_value(price, symbol)
                alloc = pv * portfolio_pct
                qty   = alloc / price
                cost  = alloc * self.commission
                if self.cash >= alloc + cost:
                    slip_price = price * (1 + self.slippage * signal)
                    self.positions[symbol] = Position(
                        symbol=symbol, direction=signal,
                        entry_price=slip_price, qty=qty,
                        entry_time=datetime.now().isoformat(),
                        current_price=price
                    )
                    self.cash -= alloc + cost
                    action = f"OPENED {'LONG' if signal==1 else 'SHORT'} {symbol} {qty:.4f} @ ${slip_price:.2f}"

        pv = self.portfolio_value(price, symbol)
        tick = {"symbol":symbol,"price":price,"signal":signal,
                "action":action,"portfolio_value":round(pv,2),"cash":round(self.cash,2)}
        self.equity_log.append({"ts":datetime.now().isoformat(),"equity":pv})
        self.broadcast(tick)
        if len(self.equity_log) % 20 == 0:
            self._save()
        return tick

    def _close(self, symbol, price, reason) -> str:
        if symbol not in self.positions: return ""
        p = self.positions[symbol]
        slip = price * (1 - self.slippage * p.direction)
        pnl  = (slip - p.entry_price) * p.qty * p.direction
        self.cash += p.qty * p.entry_price + pnl - p.qty*p.entry_price*self.commission
        self.closed_trades.append(ClosedTrade(
            trade_id=str(uuid.uuid4())[:8], symbol=symbol, direction=p.direction,
            entry_price=round(p.entry_price,4), exit_price=round(slip,4),
            qty=round(p.qty,6), pnl=round(pnl,2),
            pnl_pct=round((slip-p.entry_price)/p.entry_price*p.direction*100,3),
            entry_time=p.entry_time, exit_time=datetime.now().isoformat(),
            exit_reason=reason
        ))
        del self.positions[symbol]
        return f"CLOSED {symbol} @ ${slip:.2f} P&L ${pnl:+.2f} ({reason})"

    def portfolio_value(self, price=None, symbol=None):
        total = self.cash
        for sym, pos in self.positions.items():
            p = price if sym==symbol else pos.current_price
            if p and p > 0:
                total += pos.qty*pos.entry_price + (p-pos.entry_price)*pos.qty*pos.direction
        return total

    def summary(self):
        pv = self.portfolio_value()
        wins   = [t for t in self.closed_trades if t.pnl > 0]
        losses = [t for t in self.closed_trades if t.pnl <= 0]
        return {
            "portfolio_value":  round(pv, 2),
            "cash":             round(self.cash, 2),
            "total_return_pct": round((pv/self.initial_capital-1)*100, 2),
            "total_pnl":        round(pv - self.initial_capital, 2),
            "realized_pnl":     round(sum(t.pnl for t in self.closed_trades), 2),
            "num_trades":       len(self.closed_trades),
            "win_rate":         round(len(wins)/max(len(self.closed_trades),1)*100,1),
            "open_positions": {
                sym: {"direction":p.direction,"entry":p.entry_price,
                      "unrealized_pnl":round(p.unrealized_pnl,2),
                      "unrealized_pct":round(p.unrealized_pct*100,2)}
                for sym,p in self.positions.items()
            },
            "equity_log": self.equity_log[-200:],
            "recent_trades": [asdict(t) for t in self.closed_trades[-50:]],
        }

    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_log = []
        if os.path.exists(self.STATE_FILE):
            os.remove(self.STATE_FILE)
