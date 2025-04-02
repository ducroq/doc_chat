from logging import getLogger
import random
import time
import hashlib
import json
import bcrypt
from fastapi import HTTPException, APIRouter, Depends, Request, Form

from auth.auth_service import get_api_key
from auth.user_manager import get_users_db
from chat_logging.chat_logger import chat_logger

router = APIRouter()
logger = getLogger(__name__)

def generate_math_captcha():
    """Generate a simple math problem as CAPTCHA"""
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        answer = a + b
        question = f"{a} + {b}"
    elif operation == '-':
        # Ensure positive result
        if b > a:
            a, b = b, a
        answer = a - b
        question = f"{a} - {b}"
    else:  # multiplication
        answer = a * b
        question = f"{a} × {b}"
    
    # Create a hash of the answer with a time-based salt
    timestamp = int(time.time())
    answer_hash = hashlib.sha256(f"{answer}:{timestamp}".encode()).hexdigest()
    
    return {
        "question": question,
        "hash": answer_hash,
        "timestamp": timestamp
    }

def verify_math_captcha(user_answer, answer_hash, timestamp):
    """Verify the math CAPTCHA answer"""
    # Ensure timestamp is an integer
    try:
        timestamp = int(timestamp)
        current_time = int(time.time())
        
        # Check if CAPTCHA has expired (10 minutes)
        if current_time - timestamp > 600:  # This is where the error was happening
            return False
        
        # Convert user_answer to int and verify
        user_answer = int(user_answer)
        check_hash = hashlib.sha256(f"{user_answer}:{timestamp}".encode()).hexdigest()
        return check_hash == answer_hash
    except (ValueError, TypeError) as e:
        # Log the error for debugging
        print(f"CAPTCHA verification error: {e}")
        return False

@router.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "EU-Compliant RAG API is running"}

@router.get("/status")
async def status(request: Request):
    """Check the status of the API and its connections."""
    weaviate_client = request.app.state.weaviate_client
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    mistral_client = request.app.state.mistral_client
    if not mistral_client:
        raise HTTPException(status_code=503, detail="Mistral API client not configured")   
        
    weaviate_status = "connected" if weaviate_client and weaviate_client.is_ready() else "disconnected"
    
    return {
        "api": "running",
        "weaviate": weaviate_status,
        "mistral_api": "configured" if mistral_client else "not configured"
    }

@router.post("/admin/flush-logs")
async def flush_logs(api_key: str = Depends(get_api_key)):
    """
    Manually flush all log buffers to disk.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        dict: Status message
    """
    try:
        if chat_logger and chat_logger.enabled:
            # Flush regular chat logs
            if hasattr(chat_logger, "_flush_buffer"):
                chat_logger._flush_buffer()
                
            # Flush feedback logs if that method exists
            if hasattr(chat_logger, "_flush_feedback_buffer"):
                chat_logger._flush_feedback_buffer()
                
            return {"status": "success", "message": "All log buffers flushed to disk"}
        else:
            return {"status": "warning", "message": "Chat logging is not enabled"}
    except Exception as e:
        logger.error(f"Error flushing logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error flushing logs: {str(e)}"
        )
    
@router.get("/captcha")
async def get_captcha():
    """Generate a math CAPTCHA"""
    captcha = generate_math_captcha()
    return {
        "question": captcha["question"],
        "hash": captcha["hash"],
        "timestamp": captcha["timestamp"]
    }

@router.post("/register")
async def register_user(
    request: Request,
    captcha_answer: str = Form(...),
    captcha_hash: str = Form(...),
    captcha_timestamp: str = Form(...)
):
    """Register a new user with CAPTCHA validation"""

    # You could use request.client.host for logging or additional validation
    client_ip = request.client.host

    # Parse the form data to get user information
    form_data = await request.form()
    
    # Extract user data fields
    username = form_data.get("username")
    password = form_data.get("password")
    full_name = form_data.get("full_name")
    email = form_data.get("email")
    
    # Validate required fields
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    # Verify CAPTCHA
    try:
        is_valid = verify_math_captcha(
            user_answer=captcha_answer,
            answer_hash=captcha_hash,
            timestamp=captcha_timestamp
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="CAPTCHA verification failed. Please try again."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CAPTCHA verification error: {str(e)}"
        )
    
    # Continue with registration logic...
    users_db = get_users_db()
    if username in users_db:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Validate password strength
    from utils.utils import validate_password
    is_valid, message = validate_password(password)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=message
        )
    
    # Create the user with non-admin role
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # Create new user entry
    new_user = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False,
        "is_admin": False  # Self-registered users are not admins
    }
    
    # Add to users DB
    users_db[username] = new_user
    
    # Save the updated user DB
    with open('users.json', 'w') as f:
        json.dump(users_db, f, indent=2)
    
    return {"message": "User registered successfully"}

