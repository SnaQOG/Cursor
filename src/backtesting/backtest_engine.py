import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from ..analysis.indicators import TechnicalIndicators
from ..analysis.risk_manager import RiskManager, PositionInfo, PositionType

@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    position_type: PositionType
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    max_drawdown: float
    holding_period: int

@dataclass
class BacktestResults:
    """Results of a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade: float
    num_trades: int
    trades: List[Trade]
    equity_curve: np.ndarray
    monthly_returns: Dict[str, float]
    
class BacktestEngine:
    """Backtesting engine for strategy optimization."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.indicators = TechnicalIndicators()
        self.logger = logging.getLogger(__name__)
        
        # Initialize performance tracking
        self.initial_balance = config.get('INITIAL_BALANCE', 10000)
        self.commission_rate = config.get('COMMISSION_RATE', 0.001)
        self.slippage = config.get('SLIPPAGE', 0.0005)
        
    def run_backtest(self,
                    data: Dict[str, pd.DataFrame],
                    strategy_func: Callable,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> BacktestResults:
        """
        Run backtest over historical data using the provided strategy function.
        
        Args:
            data: Dictionary of DataFrames with OHLCV data for each symbol
            strategy_func: Function that generates trading signals
            start_date: Start date for backtest
            end_date: End date for backtest
        """
        # Initialize tracking variables
        balance = self.initial_balance
        equity_curve = [balance]
        trades = []
        open_positions: Dict[str, PositionInfo] = {}
        
        # Prepare data
        aligned_data = self._align_data(data, start_date, end_date)
        dates = list(aligned_data[list(aligned_data.keys())[0]].index)
        
        # Run simulation
        for current_date in dates:
            # Update open positions
            balance = self._update_positions(
                open_positions,
                aligned_data,
                current_date,
                balance,
                trades
            )
            
            # Get new signals
            current_data = {
                symbol: df.loc[:current_date]
                for symbol, df in aligned_data.items()
            }
            signals = strategy_func(current_data)
            
            # Process signals
            for symbol, signal in signals.items():
                if signal['action'] in ['buy', 'sell']:
                    # Check if we can open new position
                    if symbol not in open_positions:
                        success = self._open_position(
                            symbol,
                            signal,
                            current_date,
                            aligned_data[symbol],
                            balance,
                            open_positions
                        )
                        if success:
                            balance -= self._calculate_commission(signal['size'] * signal['price'])
                            
                elif signal['action'] == 'close' and symbol in open_positions:
                    # Close position
                    balance = self._close_position(
                        symbol,
                        current_date,
                        aligned_data[symbol],
                        open_positions,
                        trades,
                        balance
                    )
                    
            # Update equity curve
            total_value = self._calculate_total_value(
                balance,
                open_positions,
                aligned_data,
                current_date
            )
            equity_curve.append(total_value)
            
        # Close any remaining positions
        final_date = dates[-1]
        for symbol in list(open_positions.keys()):
            balance = self._close_position(
                symbol,
                final_date,
                aligned_data[symbol],
                open_positions,
                trades,
                balance
            )
            
        # Calculate performance metrics
        results = self._calculate_performance_metrics(
            equity_curve,
            trades,
            dates
        )
        
        return results
        
    def _align_data(self,
                    data: Dict[str, pd.DataFrame],
                    start_date: Optional[datetime],
                    end_date: Optional[datetime]) -> Dict[str, pd.DataFrame]:
        """Align data from different symbols to same timeframe."""
        aligned_data = {}
        
        for symbol, df in data.items():
            # Apply date range if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
                
            aligned_data[symbol] = df
            
        return aligned_data
        
    def _update_positions(self,
                         positions: Dict[str, PositionInfo],
                         data: Dict[str, pd.DataFrame],
                         current_date: datetime,
                         balance: float,
                         trades: List[Trade]) -> float:
        """Update open positions with latest prices and check stops."""
        for symbol in list(positions.keys()):
            position = positions[symbol]
            current_price = data[symbol].loc[current_date, 'close']
            
            # Calculate unrealized PnL
            if position.type == PositionType.LONG:
                unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.size
                
            # Check stops
            hit_stop = False
            if position.type == PositionType.LONG:
                if current_price <= position.stop_loss:
                    hit_stop = True
                elif current_price >= position.take_profit:
                    hit_stop = True
            else:
                if current_price >= position.stop_loss:
                    hit_stop = True
                elif current_price <= position.take_profit:
                    hit_stop = True
                    
            if hit_stop:
                balance = self._close_position(
                    symbol,
                    current_date,
                    data[symbol],
                    positions,
                    trades,
                    balance
                )
                
            # Update trailing stops
            else:
                atr = self._calculate_atr(data[symbol])
                new_stop = self.risk_manager.update_trailing_stop(
                    position,
                    current_price,
                    atr
                )
                positions[symbol] = PositionInfo(
                    type=position.type,
                    entry_price=position.entry_price,
                    size=position.size,
                    stop_loss=new_stop,
                    take_profit=position.take_profit,
                    risk_reward_ratio=position.risk_reward_ratio,
                    max_drawdown=position.max_drawdown
                )
                
        return balance
        
    def _open_position(self,
                      symbol: str,
                      signal: Dict,
                      current_date: datetime,
                      data: pd.DataFrame,
                      balance: float,
                      positions: Dict[str, PositionInfo]) -> bool:
        """Open new position based on signal."""
        try:
            # Calculate position parameters
            entry_price = signal['price']
            position_type = PositionType.LONG if signal['action'] == 'buy' else PositionType.SHORT
            
            # Calculate stops
            atr = self._calculate_atr(data)
            stop_loss, take_profit = self.risk_manager.calculate_dynamic_stops(
                entry_price,
                position_type,
                atr
            )
            
            # Calculate position size
            size = self.risk_manager.calculate_position_size(
                balance,
                entry_price,
                stop_loss,
                atr
            )
            
            # Create position info
            risk_reward_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
            position = PositionInfo(
                type=position_type,
                entry_price=entry_price,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                max_drawdown=0.0
            )
            
            # Validate position
            valid, message = self.risk_manager.validate_new_position(
                position,
                positions,
                balance
            )
            
            if valid:
                positions[symbol] = position
                return True
            else:
                self.logger.info(f"Position rejected: {message}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return False
            
    def _close_position(self,
                       symbol: str,
                       current_date: datetime,
                       data: pd.DataFrame,
                       positions: Dict[str, PositionInfo],
                       trades: List[Trade],
                       balance: float) -> float:
        """Close position and record trade."""
        try:
            position = positions[symbol]
            exit_price = data.loc[current_date, 'close']
            
            # Calculate PnL
            if position.type == PositionType.LONG:
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
                
            pnl_pct = pnl / (position.entry_price * position.size)
            
            # Record trade
            entry_time = data.index[data['close'] == position.entry_price][0]
            holding_period = (current_date - entry_time).days
            
            trade = Trade(
                symbol=symbol,
                entry_time=entry_time,
                exit_time=current_date,
                position_type=position.type,
                entry_price=position.entry_price,
                exit_price=exit_price,
                size=position.size,
                pnl=pnl,
                pnl_pct=pnl_pct,
                max_drawdown=position.max_drawdown,
                holding_period=holding_period
            )
            trades.append(trade)
            
            # Update balance
            commission = self._calculate_commission(exit_price * position.size)
            balance += pnl - commission
            
            # Remove position
            del positions[symbol]
            
            return balance
            
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return balance
            
    def _calculate_commission(self, value: float) -> float:
        """Calculate trading commission."""
        return value * self.commission_rate
        
    def _calculate_total_value(self,
                             balance: float,
                             positions: Dict[str, PositionInfo],
                             data: Dict[str, pd.DataFrame],
                             current_date: datetime) -> float:
        """Calculate total portfolio value including open positions."""
        total_value = balance
        
        for symbol, position in positions.items():
            current_price = data[symbol].loc[current_date, 'close']
            if position.type == PositionType.LONG:
                unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                unrealized_pnl = (position.entry_price - current_price) * position.size
            total_value += unrealized_pnl
            
        return total_value
        
    def _calculate_performance_metrics(self,
                                    equity_curve: List[float],
                                    trades: List[Trade],
                                    dates: List[datetime]) -> BacktestResults:
        """Calculate performance metrics from backtest results."""
        equity_curve = np.array(equity_curve)
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        # Calculate Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)
        
        # Calculate trade metrics
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(trades)
            
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_trade = np.mean([t.pnl for t in trades])
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0
            
        # Calculate monthly returns
        equity_df = pd.Series(equity_curve, index=dates)
        monthly_returns = equity_df.resample('M').last().pct_change().to_dict()
        
        return BacktestResults(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade=avg_trade,
            num_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns
        )
        
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR value."""
        try:
            atr = self.indicators.calculate_atr(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                period
            )
            return atr[-1]
        except:
            return 0.0 