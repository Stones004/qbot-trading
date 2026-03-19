"""core/risk_manager.py + core/position_sizer.py"""
import numpy as np


class RiskManager:
    def __init__(self, max_drawdown=0.15, max_position_pct=0.20,
                 max_vol=0.60, stop_loss_pct=0.03, take_profit_pct=0.09):
        self.max_dd    = max_drawdown
        self.max_pos   = max_position_pct
        self.max_vol   = max_vol
        self.sl        = stop_loss_pct
        self.tp        = take_profit_pct

    def check(self, symbol, current_dd, position_pct, realized_vol) -> dict:
        reasons = []
        if current_dd  <= -self.max_dd:   reasons.append(f"drawdown {current_dd:.1%} breached limit {-self.max_dd:.1%}")
        if position_pct >= self.max_pos:  reasons.append(f"position {position_pct:.1%} at cap {self.max_pos:.1%}")
        if realized_vol >= self.max_vol:  reasons.append(f"volatility {realized_vol:.1%} too high")
        return {"approved": len(reasons) == 0, "reasons": reasons}


class PositionSizer:
    def __init__(self, target_vol=0.15, kelly_fraction=0.25, max_position_pct=0.20):
        self.target_vol  = target_vol
        self.kelly       = kelly_fraction
        self.max_pos     = max_position_pct

    def size(self, portfolio_value, realized_vol, signal_strength=1.0) -> float:
        if realized_vol <= 0:
            raw = self.kelly * self.max_pos
        else:
            raw = (self.target_vol / realized_vol) * self.kelly * signal_strength
        return portfolio_value * min(raw, self.max_pos)
