# Structured logging configuration for Alfons AI Voice Agent
# Provides consistent logging across all components

import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging with healthcare-specific fields
    """
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields if present
        if hasattr(record, 'call_sid'):
            log_entry["call_sid"] = record.call_sid
        if hasattr(record, 'patient_id'):
            log_entry["patient_id"] = record.patient_id
        if hasattr(record, 'duration_ms'):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, 'provider'):
            log_entry["provider"] = record.provider
        if hasattr(record, 'model'):
            log_entry["model"] = record.model
        
        return json.dumps(log_entry)

def setup_logging(
    level: str = "DEBUG",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    structured: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
        enable_console: Whether to enable console logging
        structured: Whether to use structured JSON logging
    
    Returns:
        Configured root logger
    """
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Choose formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class CallLogger:
    """
    Context manager for call-specific logging with automatic call_sid injection
    """
    
    def __init__(self, call_sid: str, logger: Optional[logging.Logger] = None):
        self.call_sid = call_sid
        self.logger = logger or get_logger("alfons.call")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            self.info(f"Call completed", duration_ms=duration)
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method that adds call_sid"""
        extra = {"call_sid": self.call_sid, **kwargs}
        getattr(self.logger, level)(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log("critical", message, **kwargs)

# Performance logging utilities
def log_latency(logger: logging.Logger, operation: str, duration_ms: float, **kwargs):
    """Log latency metrics for performance monitoring"""
    logger.info(
        f"{operation} completed",
        extra={"duration_ms": duration_ms, "operation": operation, **kwargs}
    )

def log_provider_usage(logger: logging.Logger, provider: str, model: str, tokens: int = None, **kwargs):
    """Log AI provider usage for cost and performance tracking"""
    extra = {"provider": provider, "model": model, **kwargs}
    if tokens:
        extra["tokens"] = tokens
    
    logger.info(f"Provider {provider} called", extra=extra)