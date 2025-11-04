#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions and logger configuration.
Provides centralized logging, formatting, and helper functions.
"""

import sys
from pathlib import Path
from loguru import logger


# Configure loguru logger
def setup_logger(log_file="work/pipeline.log", level="INFO", rotation="10 MB"):
    """
    Configure centralized logger for the entire pipeline.
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log file
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Add file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # File always gets DEBUG level
        rotation=rotation,
        compression="zip",
        retention="30 days"
    )
    
    return logger


# Timing decorator
def log_execution_time(func):
    """Decorator to log function execution time."""
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"✓ {func.__name__} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"✗ {func.__name__} failed after {elapsed:.2f}s: {e}")
            raise
    
    return wrapper


# Format utilities
def format_value(value, spec=None, na="N/A"):
    """
    Format a value for display, handling None, NaN, and inf.
    
    Args:
        value: Value to format
        spec: Format specification string (e.g., '.2f')
        na: String to return if value is unavailable
        
    Returns:
        str: Formatted value
    """
    if value is None:
        return na
    
    try:
        import numpy as np
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return na
    except ImportError:
        pass
    
    try:
        if spec:
            return f"{value:{spec}}"
        return str(value)
    except Exception:
        return str(value)


def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning default if division fails."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default


def ensure_dir(path):
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(path):
    """Get file size in MB."""
    try:
        return Path(path).stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


# Initialize default logger
setup_logger()

