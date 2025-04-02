from logging import getLogger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import os
import time
import logging
from collections import defaultdict

from config import settings

logger = getLogger(__name__)

# Storage for rate limiting
ip_request_counters = defaultdict(list)
registration_ip_counters = defaultdict(list)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Only add CSP headers for non-documentation endpoints
        if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
            
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip check for non-protected endpoints
        if request.url.path in ["/", "/status", "/docs", "/openapi.json", "/privacy", "/statistics", "/documents/count"] or request.url.path.startswith("/docs/"):
            return await call_next(request)

        # Only check API key for protected endpoints
        try:
            # Get the API key from environment
            api_key_file = settings.INTERNAL_API_KEY_FILE
            if not api_key_file or not os.path.exists(api_key_file):
                # If API key file isn't set or doesn't exist, log a warning and continue
                logger.warning(f"API key file not found: {api_key_file}")
                return await call_next(request)
                
            with open(api_key_file, "r") as f:
                expected_key = f.read().strip()
            
            # Check if API key is valid
            api_key = request.headers.get("X-API-Key")
            if not api_key or api_key != expected_key:
                logger.warning(f"Invalid API key used in request to {request.url.path}")
                return Response(
                    content='{"detail":"Invalid API key"}',
                    status_code=403,
                    media_type="application/json"
                )
                
            # If we made it here, the key is valid
            return await call_next(request)
        except Exception as e:
            # Log unexpected errors but don't block the request
            logger.error(f"Error in API key validation: {str(e)}")
            return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        path = request.url.path
        
        # Apply dedicated limits for registration endpoint
        if path == "/register":
            # Clean old timestamps (older than 15 minutes)
            registration_ip_counters[client_ip] = [
                timestamp for timestamp in registration_ip_counters[client_ip] 
                if now - timestamp < 900  # 15 minutes in seconds
            ]
            
            # Only apply limits if there have been many attempts
            if len(registration_ip_counters[client_ip]) >= 5:
                logger.warning(f"Registration rate limit exceeded for IP: {client_ip}")
                return Response(
                    content='{"detail":"Too many registration attempts. Please try again in a few minutes."}',
                    status_code=429,
                    media_type="application/json"
                )
            
            # Add timestamp for this attempt
            registration_ip_counters[client_ip].append(now)
        else:
            # Regular rate limiting for other endpoints
            # Clean old timestamps (older than 1 minute)
            ip_request_counters[client_ip] = [
                timestamp for timestamp in ip_request_counters[client_ip] 
                if now - timestamp < 60
            ]
            
            # Check if we've hit the limit
            if len(ip_request_counters[client_ip]) >= settings.MAX_REQUESTS_PER_MINUTE:
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json"
                )
                
            # Add current timestamp
            ip_request_counters[client_ip].append(now)
        
        # Process request
        return await call_next(request)

# Export the middleware classes for use in main.py
security_headers_middleware = SecurityHeadersMiddleware
api_key_middleware = APIKeyMiddleware
rate_limit_middleware = RateLimitMiddleware