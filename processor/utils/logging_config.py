"""
Logging configuration for the document processor.
Sets up structured logging with file and console output.
"""
import os
import logging
import sys
from pathlib import Path
from datetime import datetime
import logging.handlers

from config import settings

def setup_logging():
    """
    Set up structured logging for the application.
    
    Returns:
        logging.Logger: Configured root logger
    """
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_DIR)
    
    # Try to create the directory - but handle permission errors gracefully
    try:
        log_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        print(f"Warning: Cannot create log directory at {log_dir}. Using current directory.")
        log_dir = Path('.')
    except Exception as e:
        print(f"Warning: Error creating log directory: {str(e)}. Using current directory.")
        log_dir = Path('.')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Try to set up file logging - but fall back to console-only if we have problems
    try:
        # File handler with rotation
        log_file = log_dir / f"processor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            settings.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, just use console
        print(f"Warning: Could not set up file logging: {str(e)}. Using console logging only.")
        formatter = logging.Formatter(
            settings.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Suppress excessive logging from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.INFO)
    
    # Log initial configuration
    logger = logging.getLogger("processor")
    logger.debug("Log directory: %s", log_dir)
    
    # Return logger for immediate use
    return logger

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger, typically module name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)