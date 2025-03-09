import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

class JSONFormatter(logging.Formatter):
    """
    Formatter für JSON-formatierte Log-Einträge
    """
    def format(self, record: logging.LogRecord) -> str:
        """
        Formatiert den Log-Eintrag als JSON
        
        Args:
            record: Der zu formatierende Log-Eintrag
            
        Returns:
            JSON-formatierter Log-Eintrag als String
        """
        log_object = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Füge zusätzliche Attribute hinzu, wenn vorhanden
        if hasattr(record, 'trading_pair'):
            log_object['trading_pair'] = record.trading_pair
        if hasattr(record, 'trade_id'):
            log_object['trade_id'] = record.trade_id
        if hasattr(record, 'error_details'):
            log_object['error_details'] = record.error_details
            
        # Füge Exception-Details hinzu, wenn vorhanden
        if record.exc_info:
            log_object['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
            
        return json.dumps(log_object)

class TradingLogger:
    """
    Erweiterter Logger für den Trading Bot mit JSON-Formatierung und Rotation
    """
    def __init__(self, 
                 name: str = 'trading_bot',
                 log_dir: str = 'logs',
                 max_bytes: int = 10485760,  # 10MB
                 backup_count: int = 5,
                 level: int = logging.INFO) -> None:
        """
        Initialisiert den Logger
        
        Args:
            name: Name des Loggers
            log_dir: Verzeichnis für Log-Dateien
            max_bytes: Maximale Größe einer Log-Datei
            backup_count: Anzahl der Backup-Dateien
            level: Log-Level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Erstelle Log-Verzeichnis
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Konfiguriere File Handler mit Rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / f'{name}.log',
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(JSONFormatter())
        
        # Konfiguriere Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # Füge Handler hinzu
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _log_with_context(self, 
                         level: int,
                         msg: str,
                         trading_pair: Optional[str] = None,
                         trade_id: Optional[str] = None,
                         error_details: Optional[Dict[str, Any]] = None,
                         exc_info: Optional[Exception] = None) -> None:
        """
        Erstellt einen Log-Eintrag mit Kontext
        
        Args:
            level: Log-Level
            msg: Log-Nachricht
            trading_pair: Optional - Betroffenes Trading-Pair
            trade_id: Optional - ID des betroffenen Trades
            error_details: Optional - Zusätzliche Fehlerdetails
            exc_info: Optional - Exception-Information
        """
        extra = {}
        if trading_pair:
            extra['trading_pair'] = trading_pair
        if trade_id:
            extra['trade_id'] = trade_id
        if error_details:
            extra['error_details'] = error_details
            
        self.logger.log(level, msg, extra=extra, exc_info=exc_info)
        
    def info(self, 
             msg: str,
             trading_pair: Optional[str] = None,
             trade_id: Optional[str] = None) -> None:
        """Log eine Info-Nachricht"""
        self._log_with_context(logging.INFO, msg, trading_pair, trade_id)
        
    def warning(self,
                msg: str,
                trading_pair: Optional[str] = None,
                trade_id: Optional[str] = None) -> None:
        """Log eine Warnung"""
        self._log_with_context(logging.WARNING, msg, trading_pair, trade_id)
        
    def error(self,
              msg: str,
              trading_pair: Optional[str] = None,
              trade_id: Optional[str] = None,
              error_details: Optional[Dict[str, Any]] = None,
              exc_info: Optional[Exception] = None) -> None:
        """Log einen Fehler"""
        self._log_with_context(logging.ERROR, msg, trading_pair, trade_id, error_details, exc_info)
        
    def critical(self,
                 msg: str,
                 trading_pair: Optional[str] = None,
                 trade_id: Optional[str] = None,
                 error_details: Optional[Dict[str, Any]] = None,
                 exc_info: Optional[Exception] = None) -> None:
        """Log einen kritischen Fehler"""
        self._log_with_context(logging.CRITICAL, msg, trading_pair, trade_id, error_details, exc_info)
        
    def debug(self,
              msg: str,
              trading_pair: Optional[str] = None,
              trade_id: Optional[str] = None) -> None:
        """Log eine Debug-Nachricht"""
        self._log_with_context(logging.DEBUG, msg, trading_pair, trade_id)
        
    def trade(self,
              action: str,
              trading_pair: str,
              trade_id: str,
              details: Dict[str, Any]) -> None:
        """
        Spezieller Logger für Trades
        
        Args:
            action: Art der Trading-Aktion (z.B. 'OPEN', 'CLOSE', 'UPDATE')
            trading_pair: Betroffenes Trading-Pair
            trade_id: ID des Trades
            details: Details zum Trade
        """
        msg = f"Trade {action}: {trading_pair} (ID: {trade_id})"
        self._log_with_context(
            logging.INFO,
            msg,
            trading_pair,
            trade_id,
            error_details={'trade_details': details}
        )
        
    def performance(self,
                   metrics: Dict[str, Any],
                   trading_pair: Optional[str] = None) -> None:
        """
        Logger für Performance-Metriken
        
        Args:
            metrics: Performance-Metriken
            trading_pair: Optional - Spezifisches Trading-Pair
        """
        msg = "Performance Update"
        if trading_pair:
            msg += f" für {trading_pair}"
        self._log_with_context(
            logging.INFO,
            msg,
            trading_pair,
            error_details={'metrics': metrics}
        )
        
    def market_data(self,
                    data_type: str,
                    trading_pair: str,
                    data: Dict[str, Any]) -> None:
        """
        Logger für Marktdaten
        
        Args:
            data_type: Art der Marktdaten (z.B. 'PRICE', 'VOLUME', 'INDICATORS')
            trading_pair: Betroffenes Trading-Pair
            data: Marktdaten
        """
        msg = f"Market Data {data_type}: {trading_pair}"
        self._log_with_context(
            logging.DEBUG,
            msg,
            trading_pair,
            error_details={'market_data': data}
        ) 