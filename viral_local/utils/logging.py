"""
Logging configuration and utilities for the Viral-Local system.

This module provides centralized logging setup with support for both console
and file logging, configurable log levels, and structured log formatting.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors for console output."""
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_file_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_file_logging: Whether to enable file logging
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    logger = logging.getLogger("viral_local")
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    console_format = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler with rotation (if enabled)
    if enable_file_logging:
        if log_file is None:
            log_file = f"viral_local_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file if enable_file_logging else 'Disabled'}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"viral_local.{name}")


class LoggerMixin:
    """Mixin class that provides logging functionality to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_performance(operation_name: str):
    """Decorator to log performance metrics for operations."""
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            start_time = time.time()
            start_memory = _get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = _get_memory_usage()
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory if start_memory and end_memory else 0
                
                logger.info(
                    f"Performance [{operation_name}]: "
                    f"Time={execution_time:.3f}s, Memory={memory_delta:.2f}MB"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Performance [{operation_name}] FAILED after {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def _get_memory_usage() -> Optional[float]:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        return None