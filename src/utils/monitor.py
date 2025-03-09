from typing import Dict, List, Optional, Any
import time
from datetime import datetime, timedelta
import json
import asyncio
from pathlib import Path
import numpy as np
from src.utils.logger import TradingLogger
from src.utils.config_manager import ConfigManager

class PerformanceMetrics:
    """Klasse zur Berechnung und Verwaltung von Performance-Metriken"""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_loss = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.avg_profit_per_trade = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.equity_curve: List[float] = []
        
    def update(self, trade_result: Dict[str, Any]) -> None:
        """
        Aktualisiert die Performance-Metriken mit einem neuen Trade-Ergebnis
        
        Args:
            trade_result: Dictionary mit Trade-Ergebnis-Daten
        """
        profit_loss = trade_result['profit_loss']
        self.total_trades += 1
        self.total_profit_loss += profit_loss
        
        if profit_loss > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.largest_win = max(self.largest_win, profit_loss)
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.largest_loss = min(self.largest_loss, profit_loss)
            
        self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Aktualisiere Win Rate und Profit Factor
        self.win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        self.avg_profit_per_trade = self.total_profit_loss / self.total_trades if self.total_trades > 0 else 0
        
        # Aktualisiere Equity Curve und Drawdown
        self.equity_curve.append(self.total_profit_loss)
        self._calculate_drawdown()
        
    def _calculate_drawdown(self) -> None:
        """Berechnet den maximalen Drawdown aus der Equity Curve"""
        if not self.equity_curve:
            return
            
        peak = self.equity_curve[0]
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Gibt alle Performance-Metriken zurück
        
        Returns:
            Dictionary mit allen Performance-Metriken
        """
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_profit_loss': self.total_profit_loss,
            'avg_profit_per_trade': self.avg_profit_per_trade,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_drawdown': self.max_drawdown,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }

class Monitor:
    """
    Hauptklasse für Monitoring und Alerting
    Überwacht Performance, Risiko und technische Aspekte des Trading Bots
    """
    
    def __init__(self, config: ConfigManager, logger: TradingLogger):
        self.config = config
        self.logger = logger
        self.performance = PerformanceMetrics()
        self.alerts: List[Dict[str, Any]] = []
        self.last_alert_check = time.time()
        self.alert_cooldown = 300  # 5 Minuten zwischen gleichen Alerts
        self.alert_history: Dict[str, float] = {}
        
    async def monitor_performance(self, trading_pair: str) -> None:
        """
        Überwacht die Trading-Performance und generiert Alerts
        
        Args:
            trading_pair: Das zu überwachende Trading-Pair
        """
        metrics = self.performance.get_metrics()
        
        # Überprüfe Risiko-Limits
        risk_config = self.config.get_section('RISK')
        if risk_config:
            if metrics['max_drawdown'] > risk_config['RISK_LIMITS']['MAX_DRAWDOWN_PERCENT']:
                await self._create_alert(
                    'RISK',
                    f"Maximaler Drawdown überschritten: {metrics['max_drawdown']:.2f}%",
                    trading_pair,
                    severity='HIGH'
                )
                
            if metrics['consecutive_losses'] > risk_config['RISK_LIMITS']['MAX_CONSECUTIVE_LOSSES']:
                await self._create_alert(
                    'RISK',
                    f"Zu viele Verluste in Folge: {metrics['consecutive_losses']}",
                    trading_pair,
                    severity='HIGH'
                )
                
            if metrics['win_rate'] < risk_config['RISK_LIMITS']['MIN_WIN_RATE'] * 100:
                await self._create_alert(
                    'RISK',
                    f"Win Rate zu niedrig: {metrics['win_rate']:.2f}%",
                    trading_pair,
                    severity='MEDIUM'
                )
                
        # Log Performance-Update
        self.logger.performance(metrics, trading_pair)
        
    async def monitor_technical(self, trading_pair: str, market_data: Dict[str, Any]) -> None:
        """
        Überwacht technische Aspekte wie Volatilität und Volumen
        
        Args:
            trading_pair: Das zu überwachende Trading-Pair
            market_data: Aktuelle Marktdaten
        """
        # Überprüfe Volumen
        if market_data.get('volume_24h', 0) < self.config.get('TRADING', 'MIN_VOLUME_24H', 1000000):
            await self._create_alert(
                'TECHNICAL',
                f"24h Volumen zu niedrig: {market_data['volume_24h']}",
                trading_pair,
                severity='LOW'
            )
            
        # Überprüfe Volatilität
        if 'volatility' in market_data:
            vol_threshold = self.config.get('RISK', 'MAX_VOLATILITY', 0.1)
            if market_data['volatility'] > vol_threshold:
                await self._create_alert(
                    'TECHNICAL',
                    f"Hohe Volatilität: {market_data['volatility']:.2f}",
                    trading_pair,
                    severity='MEDIUM'
                )
                
    async def monitor_system(self) -> None:
        """Überwacht System-Ressourcen und API-Limits"""
        # Überprüfe API Rate Limits
        rate_limits = self.config.get_section('RATE_LIMIT')
        if rate_limits and hasattr(self, 'api_calls'):
            if self.api_calls > rate_limits['MAX_CALLS_PER_MINUTE']:
                await self._create_alert(
                    'SYSTEM',
                    f"API Rate Limit fast erreicht: {self.api_calls} calls/min",
                    severity='HIGH'
                )
                
    async def _create_alert(self,
                          alert_type: str,
                          message: str,
                          trading_pair: Optional[str] = None,
                          severity: str = 'MEDIUM') -> None:
        """
        Erstellt einen neuen Alert
        
        Args:
            alert_type: Typ des Alerts (z.B. 'RISK', 'TECHNICAL', 'SYSTEM')
            message: Alert-Nachricht
            trading_pair: Optional - Betroffenes Trading-Pair
            severity: Schweregrad des Alerts ('LOW', 'MEDIUM', 'HIGH')
        """
        # Überprüfe Alert Cooldown
        alert_key = f"{alert_type}:{message}:{trading_pair}"
        current_time = time.time()
        
        if alert_key in self.alert_history:
            if current_time - self.alert_history[alert_key] < self.alert_cooldown:
                return
                
        self.alert_history[alert_key] = current_time
        
        alert = {
            'type': alert_type,
            'message': message,
            'trading_pair': trading_pair,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        
        # Log Alert
        if severity == 'HIGH':
            self.logger.critical(message, trading_pair=trading_pair)
        elif severity == 'MEDIUM':
            self.logger.warning(message, trading_pair=trading_pair)
        else:
            self.logger.info(message, trading_pair=trading_pair)
            
        # Sende Telegram-Benachrichtigung für wichtige Alerts
        if severity in ['HIGH', 'MEDIUM']:
            telegram_msg = (
                f"🚨 {severity} Alert\n"
                f"Typ: {alert_type}\n"
                f"{'Pair: ' + trading_pair + '\n' if trading_pair else ''}"
                f"Nachricht: {message}"
            )
            await self._send_telegram_alert(telegram_msg)
            
    async def _send_telegram_alert(self, message: str) -> None:
        """
        Sendet einen Alert über Telegram
        
        Args:
            message: Zu sendende Nachricht
        """
        telegram_config = self.config.get_section('TELEGRAM')
        if not telegram_config:
            return
            
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Sende an persönlichen Chat
                url = f"https://api.telegram.org/bot{telegram_config['BOT_TOKEN']}/sendMessage"
                async with session.post(url, json={
                    'chat_id': telegram_config['CHAT_ID'],
                    'text': message
                }) as response:
                    if response.status != 200:
                        self.logger.error("Fehler beim Senden des Telegram Alerts (Persönlich)")
                        
                # Sende an Gruppe
                if 'GROUP_ID' in telegram_config:
                    async with session.post(url, json={
                        'chat_id': telegram_config['GROUP_ID'],
                        'text': message
                    }) as response:
                        if response.status != 200:
                            self.logger.error("Fehler beim Senden des Telegram Alerts (Gruppe)")
                            
        except Exception as e:
            self.logger.error(f"Fehler beim Senden des Telegram Alerts: {str(e)}")
            
    def save_metrics(self) -> None:
        """Speichert aktuelle Performance-Metriken"""
        try:
            metrics = self.performance.get_metrics()
            metrics['timestamp'] = datetime.now().isoformat()
            
            metrics_dir = Path('data/metrics')
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            filename = metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Lade existierende Metriken wenn vorhanden
            if filename.exists():
                with open(filename) as f:
                    daily_metrics = json.load(f)
            else:
                daily_metrics = []
                
            daily_metrics.append(metrics)
            
            # Speichere aktualisierte Metriken
            with open(filename, 'w') as f:
                json.dump(daily_metrics, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Speichern der Metriken: {str(e)}")
            
    def load_metrics(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Lädt historische Performance-Metriken
        
        Args:
            days: Anzahl der Tage für die Metriken geladen werden sollen
            
        Returns:
            Liste von Metrik-Dictionaries
        """
        metrics = []
        metrics_dir = Path('data/metrics')
        
        if not metrics_dir.exists():
            return metrics
            
        # Lade Metriken für die angegebene Anzahl Tage
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            filename = metrics_dir / f"metrics_{date.strftime('%Y%m%d')}.json"
            
            if filename.exists():
                try:
                    with open(filename) as f:
                        daily_metrics = json.load(f)
                        metrics.extend(daily_metrics)
                except Exception as e:
                    self.logger.error(f"Fehler beim Laden der Metriken von {filename}: {str(e)}")
                    
        return metrics 