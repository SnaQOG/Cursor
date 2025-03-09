from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from src.utils.logger import TradingLogger
from src.utils.config_manager import ConfigManager
from src.utils.monitor import PerformanceMetrics

class BacktestResult:
    """Klasse zur Speicherung und Analyse von Backtesting-Ergebnissen"""
    
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []
        self.performance = PerformanceMetrics()
        self.equity_curve: List[float] = []
        self.drawdowns: List[float] = []
        self.start_balance: float = 0.0
        self.end_balance: float = 0.0
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Fügt einen Trade zu den Backtesting-Ergebnissen hinzu
        
        Args:
            trade: Dictionary mit Trade-Details
        """
        self.trades.append(trade)
        self.performance.update(trade)
        self.equity_curve.append(self.performance.total_profit_loss)
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Berechnet erweiterte Performance-Metriken
        
        Returns:
            Dictionary mit allen Metriken
        """
        metrics = self.performance.get_metrics()
        
        # Berechne zusätzliche Metriken
        if self.trades:
            # Zeitraum
            trade_dates = [datetime.fromisoformat(t['timestamp']) for t in self.trades]
            self.start_date = min(trade_dates)
            self.end_date = max(trade_dates)
            trading_days = (self.end_date - self.start_date).days
            
            # Returns
            returns = [t['profit_loss'] for t in self.trades]
            daily_returns = pd.Series(returns).resample('D').sum()
            
            metrics.update({
                'total_days': trading_days,
                'trades_per_day': len(self.trades) / max(trading_days, 1),
                'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
                'sortino_ratio': self._calculate_sortino_ratio(daily_returns),
                'max_drawdown_duration': self._calculate_max_drawdown_duration(),
                'profit_factor': self._calculate_profit_factor(),
                'recovery_factor': self._calculate_recovery_factor(),
                'risk_of_ruin': self._calculate_risk_of_ruin()
            })
            
        return metrics
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Berechnet das Sharpe Ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)  # Tägliche Risk-Free Rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() != 0 else 0.0
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Berechnet das Sortino Ratio"""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 0.0
        return np.sqrt(252) * (excess_returns.mean() / downside_std) if downside_std != 0 else 0.0
        
    def _calculate_max_drawdown_duration(self) -> int:
        """Berechnet die längste Drawdown-Dauer in Tagen"""
        if not self.equity_curve:
            return 0
            
        max_dd_duration = 0
        current_dd_duration = 0
        peak = self.equity_curve[0]
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
                
        return max_dd_duration
        
    def _calculate_profit_factor(self) -> float:
        """Berechnet den Profit Factor"""
        gross_profit = sum(t['profit_loss'] for t in self.trades if t['profit_loss'] > 0)
        gross_loss = abs(sum(t['profit_loss'] for t in self.trades if t['profit_loss'] < 0))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
    def _calculate_recovery_factor(self) -> float:
        """Berechnet den Recovery Factor"""
        if not self.performance.max_drawdown or self.performance.max_drawdown == 0:
            return float('inf')
        return self.performance.total_profit_loss / self.performance.max_drawdown
        
    def _calculate_risk_of_ruin(self) -> float:
        """Berechnet das Risiko des Ruins"""
        if not self.trades:
            return 0.0
            
        win_rate = self.performance.win_rate / 100
        win_loss_ratio = abs(self.performance.largest_win / self.performance.largest_loss) if self.performance.largest_loss != 0 else float('inf')
        
        if win_loss_ratio >= 1:
            return 0.0
        
        return ((1 - win_rate) / win_rate) ** (self.start_balance / abs(self.performance.largest_loss))
        
    def save_results(self, filename: str) -> None:
        """
        Speichert die Backtesting-Ergebnisse
        
        Args:
            filename: Name der Zieldatei
        """
        results = {
            'trades': self.trades,
            'metrics': self.calculate_metrics(),
            'equity_curve': self.equity_curve,
            'start_balance': self.start_balance,
            'end_balance': self.end_balance,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None
        }
        
        results_dir = Path('data/backtest_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / filename, 'w') as f:
            json.dump(results, f, indent=4)

class Backtester:
    """Hauptklasse für das Backtesting von Trading-Strategien"""
    
    def __init__(self, config: ConfigManager, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.results = BacktestResult()
        
    def load_historical_data(self,
                           trading_pair: str,
                           start_date: datetime,
                           end_date: datetime,
                           timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Lädt historische Daten für das Backtesting
        
        Args:
            trading_pair: Trading-Pair für das Backtesting
            start_date: Startdatum
            end_date: Enddatum
            timeframes: Liste der benötigten Timeframes
            
        Returns:
            Dictionary mit DataFrames für jeden Timeframe
        """
        data = {}
        for timeframe in timeframes:
            # Lade Daten aus CSV oder Datenbank
            filename = f"data/historical/{trading_pair}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                data[timeframe] = df
            except Exception as e:
                self.logger.error(f"Fehler beim Laden der historischen Daten: {str(e)}")
                raise
                
        return data
        
    def run_backtest(self,
                    strategy: Any,
                    trading_pair: str,
                    start_date: datetime,
                    end_date: datetime,
                    initial_balance: float = 10000.0,
                    commission: float = 0.001) -> BacktestResult:
        """
        Führt einen Backtest durch
        
        Args:
            strategy: Trading-Strategie-Instanz
            trading_pair: Zu testendes Trading-Pair
            start_date: Startdatum
            end_date: Enddatum
            initial_balance: Anfangskapital
            commission: Kommission pro Trade
            
        Returns:
            BacktestResult-Instanz mit den Ergebnissen
        """
        self.results = BacktestResult()
        self.results.start_balance = initial_balance
        balance = initial_balance
        
        # Lade historische Daten
        data = self.load_historical_data(
            trading_pair,
            start_date,
            end_date,
            self.config.get('TRADING', 'ANALYSIS_TIMEFRAMES', ['1h', '4h', '1d'])
        )
        
        # Initialisiere Strategie
        strategy.initialize(data)
        
        # Iteriere über jeden Zeitpunkt
        timestamps = sorted(data['1h'].index)  # Nutze kleinsten Timeframe als Basis
        position = None
        
        for i, timestamp in enumerate(timestamps):
            current_data = {
                tf: df[df.index <= timestamp] for tf, df in data.items()
            }
            
            # Aktualisiere Strategie
            strategy.update(current_data)
            
            # Prüfe auf Signale
            if position is None:
                signal = strategy.generate_signal(current_data)
                if signal:
                    # Öffne Position
                    entry_price = data['1h'].loc[timestamp, 'close']
                    position_size = self._calculate_position_size(balance, entry_price)
                    cost = position_size * entry_price * (1 + commission)
                    
                    if cost <= balance:
                        position = {
                            'type': signal['type'],
                            'entry_price': entry_price,
                            'size': position_size,
                            'entry_time': timestamp,
                            'stop_loss': signal.get('stop_loss'),
                            'take_profit': signal.get('take_profit')
                        }
                        balance -= cost
                        
            else:
                # Prüfe auf Exit-Signale
                current_price = data['1h'].loc[timestamp, 'close']
                exit_signal = self._check_exit_conditions(position, current_price, strategy, current_data)
                
                if exit_signal:
                    # Schließe Position
                    exit_price = current_price
                    revenue = position['size'] * exit_price * (1 - commission)
                    profit_loss = revenue - (position['size'] * position['entry_price'])
                    balance += revenue
                    
                    # Speichere Trade
                    trade = {
                        'entry_time': position['entry_time'].isoformat(),
                        'exit_time': timestamp.isoformat(),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'profit_loss': profit_loss,
                        'balance': balance,
                        'exit_reason': exit_signal['reason']
                    }
                    self.results.add_trade(trade)
                    position = None
                    
        # Schließe offene Position am Ende des Backtests
        if position:
            exit_price = data['1h'].iloc[-1]['close']
            revenue = position['size'] * exit_price * (1 - commission)
            profit_loss = revenue - (position['size'] * position['entry_price'])
            balance += revenue
            
            trade = {
                'entry_time': position['entry_time'].isoformat(),
                'exit_time': timestamps[-1].isoformat(),
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'size': position['size'],
                'profit_loss': profit_loss,
                'balance': balance,
                'exit_reason': 'backtest_end'
            }
            self.results.add_trade(trade)
            
        self.results.end_balance = balance
        return self.results
        
    def _calculate_position_size(self, balance: float, price: float) -> float:
        """Berechnet die Positionsgröße basierend auf Risikomanagement"""
        risk_percent = self.config.get('RISK', 'POSITION_SIZING', {}).get('MAX_POSITION_SIZE_PERCENT', 0.02)
        return (balance * risk_percent) / price
        
    def _check_exit_conditions(self,
                             position: Dict[str, Any],
                             current_price: float,
                             strategy: Any,
                             current_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, str]]:
        """
        Prüft Exit-Bedingungen für eine Position
        
        Returns:
            Dictionary mit Exit-Signal und Grund oder None
        """
        # Prüfe Stop Loss
        if position['stop_loss']:
            if (position['type'] == 'long' and current_price <= position['stop_loss']) or \
               (position['type'] == 'short' and current_price >= position['stop_loss']):
                return {'reason': 'stop_loss'}
                
        # Prüfe Take Profit
        if position['take_profit']:
            if (position['type'] == 'long' and current_price >= position['take_profit']) or \
               (position['type'] == 'short' and current_price <= position['take_profit']):
                return {'reason': 'take_profit'}
                
        # Prüfe Strategie-Exit
        if strategy.should_exit(position, current_data):
            return {'reason': 'strategy'}
            
        return None
        
    def optimize_strategy(self,
                        strategy: Any,
                        trading_pair: str,
                        start_date: datetime,
                        end_date: datetime,
                        parameters: Dict[str, List[Any]]) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimiert Strategie-Parameter durch Grid Search
        
        Args:
            strategy: Trading-Strategie-Instanz
            trading_pair: Zu testendes Trading-Pair
            start_date: Startdatum
            end_date: Enddatum
            parameters: Dictionary mit Parameter-Namen und möglichen Werten
            
        Returns:
            Tuple aus besten Parametern und zugehörigem BacktestResult
        """
        best_params = None
        best_result = None
        best_sharpe = float('-inf')
        
        # Generiere alle Parameterkombinationen
        param_names = list(parameters.keys())
        param_values = list(parameters.values())
        
        from itertools import product
        for params in product(*param_values):
            # Setze Parameter
            param_dict = dict(zip(param_names, params))
            strategy.set_parameters(param_dict)
            
            # Führe Backtest durch
            result = self.run_backtest(strategy, trading_pair, start_date, end_date)
            metrics = result.calculate_metrics()
            
            # Bewerte Ergebnis (hier: Sharpe Ratio)
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_params = param_dict
                best_result = result
                
        return best_params, best_result 