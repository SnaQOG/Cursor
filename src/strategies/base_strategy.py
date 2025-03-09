from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from src.utils.logger import TradingLogger
from src.utils.config_manager import ConfigManager

class BaseStrategy(ABC):
    """
    Abstrakte Basisklasse für Trading-Strategien.
    Definiert die grundlegende Struktur und gemeinsame Funktionalität.
    """
    
    def __init__(self, config: ConfigManager, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.data: Dict[str, pd.DataFrame] = {}
        self.indicators: Dict[str, Dict[str, np.ndarray]] = {}
        self.parameters: Dict[str, Any] = {}
        self.position = None
        
    def initialize(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Initialisiert die Strategie mit historischen Daten
        
        Args:
            data: Dictionary mit DataFrames für jeden Timeframe
        """
        self.data = data
        self._calculate_indicators()
        
    def update(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Aktualisiert die Strategie mit neuen Daten
        
        Args:
            data: Dictionary mit aktualisierten DataFrames
        """
        self.data = data
        self._calculate_indicators()
        
    @abstractmethod
    def _calculate_indicators(self) -> None:
        """
        Berechnet technische Indikatoren.
        Muss von konkreten Strategien implementiert werden.
        """
        pass
        
    @abstractmethod
    def generate_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """
        Generiert Trading-Signale basierend auf den Daten
        
        Args:
            data: Dictionary mit aktuellen Marktdaten
            
        Returns:
            Optional[Dict]: Trading-Signal oder None
        """
        pass
        
    @abstractmethod
    def should_exit(self, position: Dict[str, Any], data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prüft, ob eine Position geschlossen werden sollte
        
        Args:
            position: Aktuelle Position
            data: Aktuelle Marktdaten
            
        Returns:
            bool: True wenn Position geschlossen werden soll
        """
        pass
        
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Setzt die Parameter der Strategie
        
        Args:
            parameters: Dictionary mit Parametern
        """
        self.parameters = parameters
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """
        Validiert die Strategie-Parameter.
        Kann von konkreten Strategien überschrieben werden.
        """
        pass
        
    def _calculate_stop_loss(self,
                           entry_price: float,
                           position_type: str,
                           atr: Optional[float] = None) -> float:
        """
        Berechnet den Stop Loss für eine Position
        
        Args:
            entry_price: Eintrittspreis
            position_type: 'long' oder 'short'
            atr: Optional - ATR-Wert für dynamischen Stop Loss
            
        Returns:
            float: Stop Loss Preis
        """
        risk_config = self.config.get_section('RISK')
        if not risk_config:
            return 0.0
            
        stop_config = risk_config['STOP_LOSS']
        default_stop = stop_config['DEFAULT_STOP_LOSS_PERCENT'] / 100
        
        if atr and stop_config.get('USE_ATR', False):
            # Dynamischer Stop Loss basierend auf ATR
            multiplier = stop_config.get('ATR_MULTIPLIER', 2.0)
            if position_type == 'long':
                return entry_price - (atr * multiplier)
            else:
                return entry_price + (atr * multiplier)
        else:
            # Fixer Stop Loss basierend auf Prozentsatz
            if position_type == 'long':
                return entry_price * (1 - default_stop)
            else:
                return entry_price * (1 + default_stop)
                
    def _calculate_take_profit(self,
                             entry_price: float,
                             position_type: str,
                             stop_loss: float) -> float:
        """
        Berechnet das Take Profit Level für eine Position
        
        Args:
            entry_price: Eintrittspreis
            position_type: 'long' oder 'short'
            stop_loss: Stop Loss Preis
            
        Returns:
            float: Take Profit Preis
        """
        risk_config = self.config.get_section('RISK')
        if not risk_config:
            return 0.0
            
        tp_config = risk_config['TAKE_PROFIT']
        risk = abs(entry_price - stop_loss)
        reward_ratio = tp_config.get('RISK_REWARD_RATIO', 2.0)
        
        if position_type == 'long':
            return entry_price + (risk * reward_ratio)
        else:
            return entry_price - (risk * reward_ratio)
            
    def _check_trend(self,
                    data: pd.DataFrame,
                    ma_fast: int = 20,
                    ma_slow: int = 50) -> str:
        """
        Bestimmt den Trend basierend auf Moving Averages
        
        Args:
            data: DataFrame mit OHLCV-Daten
            ma_fast: Periode für schnellen MA
            ma_slow: Periode für langsamen MA
            
        Returns:
            str: 'uptrend', 'downtrend' oder 'sideways'
        """
        closes = data['close'].values
        ma_fast = pd.Series(closes).rolling(window=ma_fast).mean()
        ma_slow = pd.Series(closes).rolling(window=ma_slow).mean()
        
        if ma_fast.iloc[-1] > ma_slow.iloc[-1] and ma_fast.iloc[-2] > ma_slow.iloc[-2]:
            return 'uptrend'
        elif ma_fast.iloc[-1] < ma_slow.iloc[-1] and ma_fast.iloc[-2] < ma_slow.iloc[-2]:
            return 'downtrend'
        else:
            return 'sideways'
            
    def _check_volume_confirmation(self,
                                 data: pd.DataFrame,
                                 lookback: int = 20) -> bool:
        """
        Prüft ob das Volumen den Trend bestätigt
        
        Args:
            data: DataFrame mit OHLCV-Daten
            lookback: Anzahl der Perioden für Volumen-MA
            
        Returns:
            bool: True wenn Volumen den Trend bestätigt
        """
        volume = data['volume'].values
        volume_ma = pd.Series(volume).rolling(window=lookback).mean()
        return volume[-1] > volume_ma.iloc[-1]
        
    def _calculate_volatility(self,
                            data: pd.DataFrame,
                            window: int = 20) -> float:
        """
        Berechnet die aktuelle Volatilität
        
        Args:
            data: DataFrame mit OHLCV-Daten
            window: Fenster für Volatilitätsberechnung
            
        Returns:
            float: Volatilität
        """
        returns = np.log(data['close'] / data['close'].shift(1))
        return returns.rolling(window=window).std().iloc[-1]
        
    def _is_market_condition_suitable(self) -> bool:
        """
        Prüft ob die Marktbedingungen für Trading geeignet sind
        
        Returns:
            bool: True wenn Marktbedingungen geeignet sind
        """
        # Prüfe Volumen
        min_volume = self.config.get('MARKET_CONDITIONS', 'MIN_24H_VOLUME', 1000000)
        current_volume = self.data['1h']['volume'].sum() * 24  # Geschätztes 24h Volumen
        if current_volume < min_volume:
            return False
            
        # Prüfe Spread
        max_spread = self.config.get('MARKET_CONDITIONS', 'MAX_SPREAD_PERCENT', 0.5) / 100
        current_spread = (self.data['1h']['high'].iloc[-1] - self.data['1h']['low'].iloc[-1]) / self.data['1h']['close'].iloc[-1]
        if current_spread > max_spread:
            return False
            
        # Prüfe Volatilität
        volatility = self._calculate_volatility(self.data['1h'])
        max_volatility = self.config.get('MARKET_CONDITIONS', 'MAX_VOLATILITY', 0.1)
        if volatility > max_volatility:
            return False
            
        return True
        
    def _log_signal(self,
                    signal_type: str,
                    price: float,
                    reason: str,
                    indicators: Dict[str, Any]) -> None:
        """
        Loggt ein generiertes Trading-Signal
        
        Args:
            signal_type: Typ des Signals ('BUY' oder 'SELL')
            price: Aktueller Preis
            reason: Grund für das Signal
            indicators: Indikator-Werte die zum Signal geführt haben
        """
        self.logger.info(
            f"Signal generiert: {signal_type}",
            extra={
                'signal_type': signal_type,
                'price': price,
                'reason': reason,
                'indicators': indicators
            }
        ) 