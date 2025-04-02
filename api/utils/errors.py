from logging import getLogger
from typing import Dict, Any, Optional, Type
from fastapi import HTTPException

logger = getLogger(__name__)

# Base exception class
class APIError(Exception):
    """Base class for all API errors."""
    status_code: int = 500
    error_type: str = "api_error"
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None, 
        error_type: Optional[str] = None, 
        is_transient: bool = False
    ):
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type
        self.is_transient = is_transient
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a dictionary for JSON responses."""
        return {
            "error": self.error_type,
            "detail": self.message,
            "is_transient": self.is_transient
        }

# Specific error classes
class WeaviateError(APIError):
    """Errors related to Weaviate operations."""
    status_code = 503
    error_type = "weaviate_error"

class MistralError(APIError):
    """Errors related to Mistral API operations."""
    status_code = 503
    error_type = "mistral_api_error"

class AuthenticationError(APIError):
    """Authentication-related errors."""
    status_code = 401
    error_type = "authentication_error"

class RateLimitError(APIError):
    """Rate limit exceeded errors."""
    status_code = 429
    error_type = "rate_limit_exceeded"

class ValidationError(APIError):
    """Data validation errors."""
    status_code = 422
    error_type = "validation_error"

# Helper to format error responses for API
def format_error_response(e: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Format error responses consistently."""
    log_prefix = f"[{request_id}] " if request_id else ""
    
    if isinstance(e, APIError):
        logger.error(f"{log_prefix}{e.error_type}: {str(e)}")
        return {
            "answer": f"I encountered an issue: {str(e)}",
            "sources": [],
            "error": e.error_type
        }
    else:
        logger.error(f"{log_prefix}Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "answer": "I encountered an unexpected error. Please try again later.",
            "sources": [],
            "error": "unexpected_error"
        }

# Convert exceptions to HTTPExceptions for FastAPI
def http_exception_handler(e: Exception) -> HTTPException:
    """Convert any exception to an appropriate HTTPException."""
    if isinstance(e, APIError):
        return HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    else:
        return HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )