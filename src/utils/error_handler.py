from typing import Any, Callable, Dict, Optional, Type, Union
import time
from functools import wraps
import asyncio
from src.utils.logger import TradingLogger

class TradingError(Exception):
    """Basis-Exception für Trading-bezogene Fehler"""
    pass

class APIError(TradingError):
    """Fehler bei API-Aufrufen"""
    pass

class MarketDataError(TradingError):
    """Fehler beim Abrufen von Marktdaten"""
    pass

class ConfigError(TradingError):
    """Fehler in der Konfiguration"""
    pass

class DatabaseError(TradingError):
    """Fehler bei Datenbankoperationen"""
    pass

class ValidationError(TradingError):
    """Fehler bei der Validierung von Daten oder Parametern"""
    pass

class OrderError(TradingError):
    """Fehler bei der Orderausführung"""
    pass

class ErrorHandler:
    """
    Zentrale Fehlerbehandlung für den Trading Bot
    Implementiert Retry-Logik und Exponential Backoff
    """
    
    def __init__(self, logger: TradingLogger):
        self.logger = logger
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
        
    def handle_error(self,
                    error: Exception,
                    context: Dict[str, Any],
                    retry_count: int = 0) -> None:
        """
        Zentrale Fehlerbehandlung
        
        Args:
            error: Aufgetretener Fehler
            context: Kontext des Fehlers (z.B. Trading-Pair, Aktion)
            retry_count: Anzahl bisheriger Wiederholungsversuche
        """
        error_type = type(error).__name__
        error_key = f"{error_type}:{context.get('trading_pair', 'global')}"
        
        # Aktualisiere Fehlerstatistik
        current_time = time.time()
        if error_key in self.error_counts:
            if current_time - self.last_error_time[error_key] > 3600:  # Reset nach 1 Stunde
                self.error_counts[error_key] = 1
            else:
                self.error_counts[error_key] += 1
        else:
            self.error_counts[error_key] = 1
        
        self.last_error_time[error_key] = current_time
        
        # Log den Fehler
        self.logger.error(
            f"Fehler aufgetreten: {str(error)}",
            trading_pair=context.get('trading_pair'),
            trade_id=context.get('trade_id'),
            error_details={
                'error_type': error_type,
                'retry_count': retry_count,
                'error_count': self.error_counts[error_key],
                'context': context
            },
            exc_info=error
        )
        
        # Spezielle Behandlung je nach Fehlertyp
        if isinstance(error, APIError):
            self._handle_api_error(error, context, retry_count)
        elif isinstance(error, MarketDataError):
            self._handle_market_data_error(error, context, retry_count)
        elif isinstance(error, OrderError):
            self._handle_order_error(error, context, retry_count)
        elif isinstance(error, DatabaseError):
            self._handle_database_error(error, context, retry_count)
        else:
            self._handle_unknown_error(error, context, retry_count)
            
    def _handle_api_error(self,
                         error: APIError,
                         context: Dict[str, Any],
                         retry_count: int) -> None:
        """Behandelt API-spezifische Fehler"""
        if 'rate limit' in str(error).lower():
            # Implementiere Rate Limit Handling
            backoff_time = min(300, 2 ** retry_count)  # Max 5 Minuten
            self.logger.warning(
                f"Rate Limit erreicht. Warte {backoff_time} Sekunden.",
                trading_pair=context.get('trading_pair')
            )
            time.sleep(backoff_time)
        elif 'timeout' in str(error).lower():
            # Timeout-Handling
            backoff_time = min(60, 2 ** retry_count)  # Max 1 Minute
            self.logger.warning(
                f"Timeout aufgetreten. Warte {backoff_time} Sekunden.",
                trading_pair=context.get('trading_pair')
            )
            time.sleep(backoff_time)
            
    def _handle_market_data_error(self,
                                error: MarketDataError,
                                context: Dict[str, Any],
                                retry_count: int) -> None:
        """Behandelt Fehler beim Abrufen von Marktdaten"""
        backoff_time = min(30, 2 ** retry_count)  # Max 30 Sekunden
        self.logger.warning(
            f"Fehler beim Abrufen von Marktdaten. Warte {backoff_time} Sekunden.",
            trading_pair=context.get('trading_pair')
        )
        time.sleep(backoff_time)
            
    def _handle_order_error(self,
                          error: OrderError,
                          context: Dict[str, Any],
                          retry_count: int) -> None:
        """Behandelt Fehler bei der Orderausführung"""
        if 'insufficient balance' in str(error).lower():
            self.logger.critical(
                "Unzureichendes Guthaben für Order",
                trading_pair=context.get('trading_pair'),
                trade_id=context.get('trade_id'),
                error_details={'balance_required': context.get('amount')}
            )
        else:
            backoff_time = min(15, 2 ** retry_count)  # Max 15 Sekunden
            self.logger.warning(
                f"Order-Fehler. Warte {backoff_time} Sekunden.",
                trading_pair=context.get('trading_pair')
            )
            time.sleep(backoff_time)
            
    def _handle_database_error(self,
                             error: DatabaseError,
                             context: Dict[str, Any],
                             retry_count: int) -> None:
        """Behandelt Datenbankfehler"""
        self.logger.critical(
            "Kritischer Datenbankfehler aufgetreten",
            error_details={'sql_state': getattr(error, 'sqlstate', None)}
        )
            
    def _handle_unknown_error(self,
                            error: Exception,
                            context: Dict[str, Any],
                            retry_count: int) -> None:
        """Behandelt unbekannte Fehler"""
        self.logger.critical(
            f"Unbekannter Fehler aufgetreten: {str(error)}",
            trading_pair=context.get('trading_pair'),
            error_details={'error_type': type(error).__name__}
        )

def retry_on_error(max_retries: int = 3,
                  retry_exceptions: tuple = (APIError, MarketDataError),
                  backoff_factor: float = 2.0):
    """
    Decorator für automatische Wiederholungsversuche bei Fehlern
    
    Args:
        max_retries: Maximale Anzahl Wiederholungsversuche
        retry_exceptions: Tuple von Exception-Typen, die wiederholt werden sollen
        backoff_factor: Faktor für exponentielles Backoff
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            for retry in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    last_error = e
                    if retry < max_retries:
                        wait_time = backoff_factor ** retry
                        await asyncio.sleep(wait_time)
                    else:
                        raise last_error
            return None
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    last_error = e
                    if retry < max_retries:
                        wait_time = backoff_factor ** retry
                        time.sleep(wait_time)
                    else:
                        raise last_error
            return None
            
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator 