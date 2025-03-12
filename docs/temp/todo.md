
Is security now up to right standard?

3. Content Security Policy (CSP)
Add Content Security Policy headers to protect against XSS:
pythonCopy#
 Add to your FastAPI main.py
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

    API Rate Limiting Enhancements
Your current rate limiting is good, but you could make it more robust:
pythonCopy# Enhance your rate limiting with IP tracking
from fastapi import Request, HTTPException
import time
from collections import defaultdict

# IP-based rate limiting
ip_request_counters = defaultdict(list)

@app.middleware("http")
async def rate_limit_by_ip(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    
    # Clean old timestamps
    now = time.time()
    ip_request_counters[client_ip] = [timestamp for timestamp in ip_request_counters[client_ip] 
                                     if now - timestamp < 60]
    
    # Check limits
    if len(ip_request_counters[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Add current timestamp
    ip_request_counters[client_ip].append(now)
    
    # Process request
    return await call_next(request)




Input Validation Enhancement
Add validation for metadata files:
pythonCopy# In processor.py, add metadata validation
def validate_metadata(metadata):
    """Validate metadata structure and content"""
    if not isinstance(metadata, dict):
        return False, "Metadata must be a JSON object"
    
    # Check for required fields
    if "itemType" not in metadata:
        return False, "Metadata must include 'itemType'"
    
    # Sanitize fields to prevent script injection
    for field, value in metadata.items():
        if isinstance(value, str):
            # Check for suspicious patterns
            if "<script>" in value.lower() or "javascript:" in value.lower():
                return False, f"Suspicious content in field '{field}'"
    
    return True, "Valid metadata"

    Secrets Management Improvement
Implement secret rotation logic:
pythonCopy# In main.py, add secret age checking
def check_secret_age(secret_path, max_age_days=90):
    """Check if a secret file is older than max_age_days"""
    if not os.path.exists(secret_path):
        return False
    
    file_timestamp = os.path.getmtime(secret_path)
    file_age_days = (time.time() - file_timestamp) / (60 * 60 * 24)
    
    if file_age_days > max_age_days:
        logger.warning(f"Secret at {secret_path} is {file_age_days:.1f} days old and should be rotated")
        return False
        
    return True

# Check at startup
if __name__ == "__main__":
    check_secret_age("/run/secrets/mistral_api_key")



Update documentation


