import os
import logging
import sys
from pathlib import Path
from datetime import datetime
import logging.handlers
import uuid
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

def setup_logging():
    """Set up structured logging for the application."""
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
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Try to set up file logging - but fall back to console-only if we have problems
    try:
        # File handler with rotation
        log_file = log_dir / f"api_{datetime.now().strftime('%Y%m%d')}.log"
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
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Suppress excessive logging from libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.INFO)
    
    # Return logger for immediate use
    return root_logger

# JSON structured logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging of all requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger = logging.getLogger("api.request")
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
            }
        )
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        # Calculate duration
        duration_ms = round((time.time() - start_time) * 1000)
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )
        
        return response

# Helper function for structured logging
def log_event(logger, message, **extra):
    """Log an event with structured data."""
    logger.info(message, extra=extra)