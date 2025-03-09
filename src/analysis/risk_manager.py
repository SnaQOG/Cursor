import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class PositionType(Enum):
    """Position types."""
    LONG = "long"
    SHORT = "short"

@dataclass
class PositionInfo:
    """Information about a trading position."""
    type: PositionType
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    max_drawdown: float

class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_risk_per_trade = config.get('MAX_RISK_PER_TRADE', 0.02)  # 2% per trade
        self.max_total_risk = config.get('MAX_TOTAL_RISK', 0.06)  # 6% total
        self.min_risk_reward = config.get('MIN_RISK_REWARD', 2.0)  # Minimum 2:1 ratio
        self.position_sizing_atr = config.get('POSITION_SIZING_ATR', 1.5)  # ATR multiplier
        self.trailing_stop_activation = config.get('TRAILING_STOP_ACTIVATION', 0.02)  # 2% profit
        self.trailing_stop_distance = config.get('TRAILING_STOP_DISTANCE', 0.01)  # 1% from price
        
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              volatility: float) -> float:
        """
        Calculate optimal position size based on account balance and risk parameters.
        Uses the volatility (ATR) to adjust position size.
        """
        # Calculate basic position size based on risk
        risk_amount = account_balance * self.max_risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        base_size = risk_amount / price_risk
        
        # Adjust for volatility
        volatility_factor = 1.0 / (volatility * self.position_sizing_atr)
        adjusted_size = base_size * volatility_factor
        
        # Ensure minimum and maximum sizes
        min_size = self.config.get('MIN_POSITION_SIZE', 0.001)
        max_size = self.config.get('MAX_POSITION_SIZE', 1.0)
        
        return np.clip(adjusted_size, min_size, max_size)
        
    def calculate_dynamic_stops(self,
                              entry_price: float,
                              position_type: PositionType,
                              atr: float,
                              support_resistance: Optional[Dict] = None) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit levels based on ATR and support/resistance levels.
        """
        # Base stops on ATR
        atr_multiplier = self.config.get('ATR_MULTIPLIER', 2.0)
        base_stop = atr * atr_multiplier
        
        if position_type == PositionType.LONG:
            stop_loss = entry_price - base_stop
            take_profit = entry_price + (base_stop * self.min_risk_reward)
            
            # Adjust based on support/resistance if available
            if support_resistance:
                nearest_support = max([s for s in support_resistance.get('support', [])
                                    if s < entry_price], default=None)
                nearest_resistance = min([r for r in support_resistance.get('resistance', [])
                                       if r > entry_price], default=None)
                
                if nearest_support and nearest_support > stop_loss:
                    stop_loss = nearest_support
                if nearest_resistance:
                    take_profit = min(take_profit, nearest_resistance)
                    
        else:  # SHORT position
            stop_loss = entry_price + base_stop
            take_profit = entry_price - (base_stop * self.min_risk_reward)
            
            # Adjust based on support/resistance if available
            if support_resistance:
                nearest_resistance = min([r for r in support_resistance.get('resistance', [])
                                       if r > entry_price], default=None)
                nearest_support = max([s for s in support_resistance.get('support', [])
                                    if s < entry_price], default=None)
                
                if nearest_resistance and nearest_resistance < stop_loss:
                    stop_loss = nearest_resistance
                if nearest_support:
                    take_profit = max(take_profit, nearest_support)
                    
        return stop_loss, take_profit
        
    def update_trailing_stop(self,
                           position: PositionInfo,
                           current_price: float,
                           atr: float) -> float:
        """
        Update trailing stop based on price movement and ATR.
        """
        if position.type == PositionType.LONG:
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct >= self.trailing_stop_activation:
                # Calculate new stop based on ATR
                atr_stop = current_price - (atr * self.position_sizing_atr)
                # Calculate new stop based on trailing percentage
                pct_stop = current_price * (1 - self.trailing_stop_distance)
                # Use the more conservative stop
                new_stop = max(atr_stop, pct_stop, position.stop_loss)
                return new_stop
                
        else:  # SHORT position
            profit_pct = (position.entry_price - current_price) / position.entry_price
            if profit_pct >= self.trailing_stop_activation:
                # Calculate new stop based on ATR
                atr_stop = current_price + (atr * self.position_sizing_atr)
                # Calculate new stop based on trailing percentage
                pct_stop = current_price * (1 + self.trailing_stop_distance)
                # Use the more conservative stop
                new_stop = min(atr_stop, pct_stop, position.stop_loss)
                return new_stop
                
        return position.stop_loss
        
    def calculate_risk_metrics(self,
                             positions: Dict[str, PositionInfo],
                             account_balance: float) -> Dict:
        """
        Calculate various risk metrics for the current portfolio.
        """
        total_risk = 0.0
        max_drawdown = 0.0
        risk_reward_sum = 0.0
        position_count = len(positions)
        
        for symbol, pos in positions.items():
            # Calculate risk for this position
            risk_amount = abs(pos.entry_price - pos.stop_loss) * pos.size
            risk_pct = risk_amount / account_balance
            total_risk += risk_pct
            
            # Track maximum drawdown
            max_drawdown = max(max_drawdown, risk_pct)
            
            # Sum risk-reward ratios
            risk_reward_sum += pos.risk_reward_ratio
            
        return {
            'total_risk': total_risk,
            'max_drawdown': max_drawdown,
            'avg_risk_reward': risk_reward_sum / position_count if position_count > 0 else 0,
            'position_count': position_count,
            'risk_per_position': total_risk / position_count if position_count > 0 else 0
        }
        
    def validate_new_position(self,
                            new_position: PositionInfo,
                            existing_positions: Dict[str, PositionInfo],
                            account_balance: float) -> Tuple[bool, str]:
        """
        Validate if a new position can be taken based on risk parameters.
        """
        # Calculate current risk metrics
        risk_metrics = self.calculate_risk_metrics(existing_positions, account_balance)
        
        # Calculate additional risk from new position
        new_risk = abs(new_position.entry_price - new_position.stop_loss) * new_position.size / account_balance
        total_risk = risk_metrics['total_risk'] + new_risk
        
        # Check risk-reward ratio
        if new_position.risk_reward_ratio < self.min_risk_reward:
            return False, f"Risk-reward ratio {new_position.risk_reward_ratio:.2f} below minimum {self.min_risk_reward}"
            
        # Check total risk
        if total_risk > self.max_total_risk:
            return False, f"Total risk {total_risk:.2%} would exceed maximum {self.max_total_risk:.2%}"
            
        # Check position correlation if configured
        if self.config.get('CHECK_CORRELATION', False):
            # Implementation for correlation check would go here
            pass
            
        return True, "Position validated successfully"
        
    def calculate_value_at_risk(self,
                              positions: Dict[str, PositionInfo],
                              price_history: Dict[str, np.ndarray],
                              confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) using historical simulation method.
        """
        portfolio_returns = []
        
        for symbol, position in positions.items():
            if symbol in price_history:
                # Calculate daily returns
                prices = price_history[symbol]
                returns = np.diff(prices) / prices[:-1]
                
                # Calculate position value changes
                value_changes = position.size * position.entry_price * returns
                portfolio_returns.append(value_changes)
                
        if portfolio_returns:
            # Combine returns for all positions
            total_returns = np.sum(portfolio_returns, axis=0)
            
            # Calculate VaR
            var_percentile = 1.0 - confidence_level
            var = np.percentile(total_returns, var_percentile * 100)
            
            return abs(var)
            
        return 0.0 