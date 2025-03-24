import os
import re
import json
import time
import uuid
import logging
import hashlib
import pathlib
import asyncio
from datetime import datetime, timedelta
from collections import deque
from functools import lru_cache
from typing import Optional, Dict, List, Any, Union, Tuple, Set, Annotated
from pydantic import BaseModel, Field, field_validator, ValidationError
from collections import defaultdict
import bcrypt
import secrets

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, DataType
from weaviate.classes.query import Filter
from dotenv import load_dotenv
from mistralai import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential
from chat_logger import ChatLogger

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables and settings
class Settings:
    """Application settings loaded from environment variables with defaults."""
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_API_KEY_FILE = "/run/secrets/mistral_api_key"
    MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-tiny")
    DAILY_TOKEN_BUDGET = int(os.getenv("MISTRAL_DAILY_TOKEN_BUDGET", "10000"))
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MISTRAL_MAX_REQUESTS_PER_MINUTE", "10"))
    CHAT_LOG_DIR = os.getenv("CHAT_LOG_DIR", "chat_data")
    INTERNAL_API_KEY_FILE = os.getenv("INTERNAL_API_KEY_FILE", "/run/secrets/internal_api_key")
    MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "100"))
    CACHE_EXPIRY_SECONDS = int(os.getenv("CACHE_EXPIRY_SECONDS", "3600"))  # 1 hour
    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 3
    JWT_SECRET_KEY_FILE = os.getenv("JWT_SECRET_KEY_FILE", "/run/secrets/jwt_secret_key")
    JWT_SECRET_KEY = None
    JWT_ALGORITHM = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30    

settings = Settings()

# Load API key from file if available
if not settings.MISTRAL_API_KEY and os.path.exists(settings.MISTRAL_API_KEY_FILE):
    try:
        with open(settings.MISTRAL_API_KEY_FILE, "r") as f:
            settings.MISTRAL_API_KEY = f.read().strip()
    except Exception as e:
        logger.error(f"Error reading Mistral API key from file: {str(e)}")

# Global state tracking
token_usage = {
    "count": 0,
    "reset_date": datetime.now().strftime("%Y-%m-%d")
}
request_timestamps = deque(maxlen=settings.MAX_REQUESTS_PER_MINUTE)
response_cache: Dict[str, Dict[str, Any]] = {}  # Dictionary to store cached responses

# Global state for rate limiting
ip_request_counters = defaultdict(list)

# Initialize chat logger
chat_logger = None

# OAuth2 password bearer for token-based authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load the secret from file
if os.path.exists(settings.JWT_SECRET_KEY_FILE):
    try:
        with open(settings.JWT_SECRET_KEY_FILE, "r") as f:
            settings.JWT_SECRET_KEY = f.read().strip()
    except Exception as e:
        logger.error(f"Error reading JWT secret key from file: {str(e)}")

if not settings.JWT_SECRET_KEY:
    # Fall back to a randomly generated key (not ideal but better than a fixed default)
    logger.warning("JWT secret key not found, generating a random one (will change on restart)")
    settings.JWT_SECRET_KEY = secrets.token_hex(32)

# Helper functions
def validate_user_input_content(v: str) -> str:
    """
    Validate that user input does not contain malicious patterns.
    
    Args:
        v: The  string to validate
        
    Returns:
        str: The validated string
        
    Raises:
        ValueError: If the string contains dangerous patterns
    """
    # 1. Check for script injection patterns
    dangerous_patterns = [
        '<script>', 'javascript:', 'onload=', 'onerror=', 'onclick=',
        'ondblclick=', 'onmouseover=', 'onmouseout=', 'onfocus=', 'onblur=',
        'oninput=', 'onchange=', 'onsubmit=', 'onreset=', 'onselect=',
        'onkeydown=', 'onkeypress=', 'onkeyup=', 'ondragenter=', 'ondragleave=',
        'data:text/html', 'vbscript:', 'expression(', 'document.cookie',
        'document.write', 'window.location', 'eval(', 'exec('
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in v.lower():
            raise ValueError(f'Potentially unsafe input detected: {pattern}')
    
    # 2. Check for SQL injection patterns - Fixed regex
    sql_patterns = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION',
        'FROM', 'WHERE', '1=1', 'OR 1=1', 'OR TRUE', '--'
    ]
    
    # Count SQL keywords manually to avoid regex issues
    sql_count = 0
    for pattern in sql_patterns:
        # Check for whole words only
        if re.search(r'\b' + re.escape(pattern) + r'\b', v.upper()):
            sql_count += 1
    
    # Allow a few keywords as they might be in natural language
    if sql_count >= 3:
        raise ValueError('Potential SQL injection pattern detected')
    
    # 3. Check for command injection patterns
    cmd_patterns = [
        ';', '&&', '||', '`', '$(',  # Command chaining in bash/shell
        '| ', '>>', '>', '<', 'ping ', 'wget ', 'curl ', 
        'chmod ', 'rm -', 'sudo ', '/etc/', '/bin/'
    ]
    
    for pattern in cmd_patterns:
        if pattern in v:
            raise ValueError(f'Potential command injection pattern detected: {pattern}')
    
    # 4. Check for excessive special characters (might indicate an attack)
    special_char_count = sum(1 for char in v if char in '!@#$%^&*()+={}[]|\\:;"\'<>?/~`')
    if special_char_count > len(v) * 0.3:  # If more than 30% are special characters
        raise ValueError('Too many special characters in input')
        
    # 5. Check for extremely repetitive patterns (DoS attempts)
    if re.search(r'(.)\1{20,}', v):  # Same character repeated 20+ times
        raise ValueError('Input contains excessive repetition')
        
    return v


# Models
class Query(BaseModel):
    """
    Model for user queries with validation.
    
    Attributes:
        question: The user's question to be processed
    """
    question: str = Field(..., min_length=settings.MIN_QUERY_LENGTH, max_length=settings.MAX_QUERY_LENGTH)
    
    @field_validator('question')
    @classmethod
    def validate_question_content(cls, v: str) -> str:
        """
        Validate that a question does not contain malicious patterns.
        
        Args:
            v: The question string to validate
            
        Returns:
            str: The validated question
            
        Raises:
            ValueError: If the question contains dangerous patterns
        """
        validate_user_input_content(v)
        return v
    
    @field_validator('question')
    @classmethod
    def normalize_question(cls, v: str) -> str:
        """
        Normalize question format by trimming whitespace and adding question mark if needed.
        
        Args:
            v: The question string to normalize
            
        Returns:
            str: The normalized question
        """
        # Trim excessive whitespace
        v = re.sub(r'\s+', ' ', v).strip()
        
        # Ensure the question ends with a question mark if it looks like a question
        question_starters = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'do', 'does']
        if any(v.lower().startswith(starter) for starter in question_starters) and not v.endswith('?'):
            v += '?'
            
        return v
    
class QueryWithHistory(Query):
    """
    Extended query model that includes conversation history.
    
    Attributes:
        conversation_history: Optional list of previous interactions
    """
    conversation_history: Optional[List[Dict[str, Any]]] = None 

class APIResponse(BaseModel):
    """
    Model for API responses with source citations.
    
    Attributes:
        answer: The generated response text
        sources: List of source documents used to generate the answer
        error: Optional error information
    """
    answer: str
    sources: List[Dict[str, Any]] = []
    error: Optional[str] = None

class DocumentMetadata(BaseModel):
    """
    Model for document metadata.
    
    Attributes:
        page: Optional page number within document
        heading: Optional section heading
        headingLevel: Optional heading level (1-6)
        creators: Optional list of content creators
        title: Optional document title
        date: Optional document publication date
        itemType: Optional document type (e.g., "journalArticle")
    """
    page: Optional[int] = None
    heading: Optional[str] = None
    headingLevel: Optional[int] = None
    creators: Optional[List[Dict[str, str]]] = None
    title: Optional[str] = None
    date: Optional[str] = None
    itemType: Optional[str] = None
    
class Source(BaseModel):
    """
    Model for document source information.
    
    Attributes:
        filename: The filename of the source document
        chunkId: The chunk ID within the document
        page: Optional page number
        heading: Optional section heading
        metadata: Optional document metadata
    """
    filename: str
    chunkId: int
    page: Optional[int] = None
    heading: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None

class FeedbackModel(BaseModel):
    """
    Model for user feedback on responses.
    
    Attributes:
        request_id: ID of the original request
        message_id: ID of the specific message receiving feedback
        rating: Feedback rating (positive/negative)
        feedback_text: Optional detailed feedback
        categories: Optional categories of issues
        timestamp: When the feedback was submitted
    """
    request_id: str
    message_id: str
    rating: str = Field(..., description="positive or negative")
    feedback_text: Optional[str] = None
    categories: Optional[List[str]] = []  # Change None to [] as default
    timestamp: str
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: str) -> str:
        """Validate rating is positive or negative."""
        if v.lower() not in ["positive", "negative"]:
            raise ValueError('Rating must be "positive" or "negative"')
        return v.lower()  # Normalize to lowercase
        
    @field_validator('message_id')
    @classmethod
    def validate_message_id(cls, v: str) -> str:
        """Validate message_id is not empty."""
        if not v or not v.strip():
            raise ValueError("message_id cannot be empty")
        validate_user_input_content(v)
        return v
    
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Error classes
class MistralAPIError(Exception):
    """
    Custom exception for Mistral API errors.
    
    Attributes:
        message: Error message
        error_type: Type of error
        is_transient: Whether the error is temporary and can be retried
    """
    def __init__(self, message: str, error_type: Optional[str] = None, is_transient: bool = False):
        self.message = message
        self.error_type = error_type
        self.is_transient = is_transient
        super().__init__(self.message)

class WeaviateError(Exception):
    """
    Custom exception for Weaviate errors.
    
    Attributes:
        message: Error message
        error_type: Type of error
        is_transient: Whether the error is temporary and can be retried
    """
    def __init__(self, message: str, error_type: Optional[str] = None, is_transient: bool = False):
        self.message = message
        self.error_type = error_type
        self.is_transient = is_transient
        super().__init__(self.message)

# Authentication helper functions
def load_users_from_json():
    users_file_path = 'users.json'
    try:
        if os.path.exists(users_file_path):
            with open(users_file_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Users file not found at {users_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading users from JSON: {str(e)}")
        return {}

def get_users_db():
    return load_users_from_json()

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_user(db, username: str):
    # Load fresh user data each time to catch updates
    users_db = get_users_db()
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(None, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    
    # Import JWT library only when needed
    import jwt
    
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    import jwt
    from jwt.exceptions import PyJWTError
    
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(None, username=token_data.username)  # Remove the first parameter
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user        

# Utility functions
def check_token_budget(estimated_tokens: int) -> bool:
    """
    Check if we have enough budget for this request.
    
    Args:
        estimated_tokens: Estimated token count for this request
        
    Returns:
        bool: True if there's enough budget, False otherwise
    """
    # Reset counter if it's a new day
    today = datetime.now().strftime("%Y-%m-%d")
    if token_usage["reset_date"] != today:
        token_usage["count"] = 0
        token_usage["reset_date"] = today
        logger.info(f"Token budget reset for new day: {today}")
    
    # Check if this request would exceed our budget
    if token_usage["count"] + estimated_tokens > settings.DAILY_TOKEN_BUDGET:
        return False
    return True

def update_token_usage(tokens_used: int) -> None:
    """
    Update the token usage tracker.
    
    Args:
        tokens_used: Number of tokens used in this request
    """
    token_usage["count"] += tokens_used
    logger.info(f"Token usage: {token_usage['count']}/{settings.DAILY_TOKEN_BUDGET} for {token_usage['reset_date']}")

def check_rate_limit() -> bool:
    """
    Check if we're within rate limits.
    
    Returns:
        bool: True if the request is within rate limits, False otherwise
    """
    now = time.time()
    
    # Clean old timestamps (older than 1 minute)
    while request_timestamps and now - request_timestamps[0] > 60:
        request_timestamps.popleft()
    
    # Check if we've hit the limit
    if len(request_timestamps) >= settings.MAX_REQUESTS_PER_MINUTE:
        return False
    
    # Add current timestamp and allow request
    request_timestamps.append(now)
    return True

def check_secret_age(secret_path: str, max_age_days: int = 90) -> bool:
    """
    Check if a secret file is older than max_age_days.
    
    Args:
        secret_path: Path to the secret file
        max_age_days: Maximum age in days
        
    Returns:
        bool: True if the secret is valid, False if it's too old or missing
    """
    if not os.path.exists(secret_path):
        return False
    
    file_timestamp = os.path.getmtime(secret_path)
    file_age_days = (time.time() - file_timestamp) / (60 * 60 * 24)
    
    if file_age_days > max_age_days:
        logger.warning(f"Secret at {secret_path} is {file_age_days:.1f} days old and should be rotated")
        return False
        
    return True

def handle_api_error(e: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle API errors with appropriate response and logging.
    
    Args:
        e: The exception
        request_id: Request ID for logging
    
    Returns:
        Dict[str, Any]: Appropriate error response
    """
    log_prefix = f"[{request_id}] " if request_id else ""
    
    if isinstance(e, MistralAPIError):
        error_type = e.error_type or "mistral_api_error"
        logger.error(f"{log_prefix}Mistral API error: {str(e)}")
        return {
            "answer": f"I encountered an issue while generating a response: {str(e)}",
            "sources": [],
            "error": error_type
        }
    elif isinstance(e, WeaviateError):
        error_type = e.error_type or "weaviate_error" 
        logger.error(f"{log_prefix}Weaviate error: {str(e)}")
        return {
            "answer": "I encountered an issue while searching the knowledge base.",
            "sources": [],
            "error": error_type
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

def get_cached_response(query_hash: str, model: str) -> Optional[Dict[str, Any]]:
    """
    Get a response from cache if it exists.
    
    Args:
        query_hash: Hash of the query and context
        model: Model name used for generation
    
    Returns:
        Optional[Dict[str, Any]]: Cached response or None if not found
    """
    cache_key = f"{query_hash}_{model}"
    cached_item = response_cache.get(cache_key)
    
    # If we have a cached item and it's not expired
    if cached_item:
        # Check if the cache is still valid (cached for less than configured time)
        cache_time = cached_item.get("timestamp", 0)
        if time.time() - cache_time < settings.CACHE_EXPIRY_SECONDS:
            logger.info(f"Cache hit for key: {cache_key[:10]}...")
            return cached_item.get("response")
        else:
            # Cache expired
            logger.info(f"Cache expired for key: {cache_key[:10]}...")
            del response_cache[cache_key]
    
    return None

def set_cached_response(query_hash: str, model: str, response: Dict[str, Any]) -> None:
    """
    Store a response in the cache.
    
    Args:
        query_hash: Hash of the query and context
        model: Model name used for generation
        response: The response to cache
    """
    cache_key = f"{query_hash}_{model}"
    
    # Store the response with a timestamp
    response_cache[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }
    
    # Limit cache size to configured number of entries
    if len(response_cache) > settings.MAX_CACHE_ENTRIES:
        # Remove oldest entry
        oldest_key = None
        oldest_time = float('inf')
        
        for key, data in response_cache.items():
            if data["timestamp"] < oldest_time:
                oldest_time = data["timestamp"]
                oldest_key = key
        
        if oldest_key:
            del response_cache[oldest_key]
            logger.info(f"Removed oldest cache entry: {oldest_key[:10]}...")

def expand_question_references(question: str, history: List[Dict[str, Any]]) -> str:
    """Enhanced reference resolution for questions"""
    # Simple cases - return as is
    if len(question.split()) > 7 or "?" not in question:
        return question
        
    # Reference terms to look for
    reference_terms = {
        "pronouns": ["it", "this", "that", "they", "them", "their", "these", "those"],
        "implicit": ["the", "mentioned", "discussed", "previous", "above", "earlier"]
    }
    
    # Check if question likely contains references
    has_reference = any(term in question.lower().split() for term in 
                       reference_terms["pronouns"] + reference_terms["implicit"])
    
    if not has_reference:
        return question
    
    # Get key topics from recent conversation
    topics = []
    
    # Extract last 2 exchanges at most - FIX: Define recent_turns first
    recent_turns = min(2, len(history) // 2)
    history_subset = history[-recent_turns*2:] if recent_turns > 0 else history
    
    # Simple keyword extraction (could be enhanced with NLP)
    for msg in history_subset:
        if msg.get("role") == "user":
            # Extract nouns from user questions as potential topics
            words = msg.get("content", "").split()
            # This is simplified - ideally use POS tagging
            for word in words:
                if len(word) > 4 and word.lower() not in ["what", "when", "where", "which", "about"]:
                    topics.append(word)
    
    # Use the most recent significant topic
    main_topic = topics[0] if topics else ""
    
    if main_topic:
        # Replace common pronouns with the main topic
        for pronoun in reference_terms["pronouns"]:
            # Only replace whole words, not parts of words
            question = re.sub(r'\b' + pronoun + r'\b', main_topic, question, flags=re.IGNORECASE)
    
    return question

def create_optimized_history(full_history, max_exchanges=3, max_tokens=800):
    """
    Create an optimized conversation history for the LLM.
    
    Args:
        full_history: Complete conversation history
        max_exchanges: Maximum number of back-and-forth exchanges to include
        max_tokens: Approximate maximum tokens to include (rough estimate)
        
    Returns:
        str: Optimized conversation history
    """
    # If history is short enough, use it all
    if len(full_history) <= max_exchanges * 2:  # Each exchange is a user + assistant message
        recent_history = full_history
    else:
        # Always include the most recent exchanges
        recent_history = full_history[-max_exchanges*2:]
    
    # Format the recent history
    history_text = ""
    char_count = 0  # Rough approximation: ~4 chars per token
    
    for msg in recent_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Create formatted message
        formatted_msg = f"{role.capitalize()}: {content}\n\n"
        
        # Check if adding this would exceed our rough token budget
        if char_count + len(formatted_msg) > max_tokens * 4:
            # If we're about to exceed, add a note and stop
            history_text += "...(earlier conversation summarized)...\n\n"
            break
            
        # Otherwise add the message
        history_text += formatted_msg
        char_count += len(formatted_msg)
    
    return history_text

async def log_feedback(
    feedback: Dict[str, Any],
    request_id: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Background task to log feedback.
    
    Args:
        feedback: The feedback data
        request_id: Request identifier for this feedback submission
        user_id: Optional user identifier
        metadata: Optional additional metadata
    """
    if chat_logger and chat_logger.enabled:
        try:
            # Use log_feedback method if it exists
            if hasattr(chat_logger, "log_feedback"):
                chat_logger.log_feedback(
                    feedback=feedback,
                    request_id=request_id,
                    user_id=user_id,
                    metadata=metadata
                )
            else:
                # Fall back to generic interaction logging
                logger.info(f"[{request_id}] Using generic logging for feedback")
                chat_logger.log_interaction(
                    query=f"FEEDBACK: {feedback.get('rating', 'unknown')}",
                    response={"feedback": feedback},
                    request_id=request_id,
                    user_id=user_id,
                    metadata=metadata
                )
        except Exception as e:
            logger.error(f"Error logging feedback: {str(e)}")            

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_mistral_with_retry(
    client: Mistral, 
    model: str, 
    messages: List[Dict[str, str]], 
    temperature: float
) -> Any:
    """
    Call Mistral API with retry logic for transient errors.
    
    Args:
        client: Mistral API client
        model: Model name to use
        messages: List of message objects
        temperature: Temperature setting for generation
        
    Returns:
        Any: Mistral API response
        
    Raises:
        MistralAPIError: If the API call fails after retries
    """
    try:
        # Convert dict messages to ChatMessage objects
        return await asyncio.to_thread(
            client.chat.complete,
            model=model,
            messages=messages,  # Use the dict format directly
            temperature=temperature,
        )

    except Exception as e:
        error_message = str(e).lower()
        is_transient = any(term in error_message for term in [
            "rate limit", "timeout", "connection", "too many requests", 
            "server error", "503", "502", "504"
        ])
        
        if is_transient:
            logger.warning(f"Temporary error calling Mistral API: {str(e)}. Retrying...")
            raise  # Will trigger retry
        else:
            error_type = "authentication" if "auth" in error_message else "model_error"
            raise MistralAPIError(f"Error calling Mistral API: {str(e)}", 
                                 error_type=error_type, 
                                 is_transient=False)

# Initialize clients
# Weaviate client initialization
client = None
try:
    # Parse the URL to get components
    use_https = settings.WEAVIATE_URL.startswith("https://")
    host_part = settings.WEAVIATE_URL.replace("http://", "").replace("https://", "")
    
    # Handle port if specified
    if ":" in host_part:
        host, port = host_part.split(":")
        port = int(port)
    else:
        host = host_part
        port = 443 if use_https else 80
    
    # Connect to Weaviate using the same method as the processor
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=use_https,
        grpc_host=host,
        grpc_port=50051,  # Default gRPC port
        grpc_secure=use_https,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=30, insert=30)
        )
    )
    logger.info(f"Connected to Weaviate at {settings.WEAVIATE_URL}")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {str(e)}")

# Mistral client initialization
mistral_client = None
if settings.MISTRAL_API_KEY:
    try:
        mistral_client = Mistral(api_key=settings.MISTRAL_API_KEY)
        logger.info("Mistral client initialized, using model: " + settings.MISTRAL_MODEL)
    except Exception as e:
        logger.error(f"Failed to initialize Mistral client: {str(e)}")

# FastAPI app with lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup validation
    logger.info("Performing startup validation...")

    # Initialize the chat logger
    global chat_logger
    log_dir = settings.CHAT_LOG_DIR
    chat_logger = ChatLogger(log_dir=log_dir)
    logger.info(f"Chat logger initialized in {log_dir}")
    
    # Validate Weaviate connection
    if not client:
        logger.error("CRITICAL: Weaviate client is not initialized")
    elif not client.is_ready():
        logger.error("CRITICAL: Weaviate is not ready")
    else:
        # Check if DocumentChunk collection exists
        try:
            if client.collections.exists("DocumentChunk"):
                collection = client.collections.get("DocumentChunk")
                logger.info(f"Weaviate: DocumentChunk collection exists")
                
                # Check if there's any data
                try:
                    # Get count using aggregate API
                    count = collection.aggregate.over_all().total_count
                    logger.info(f"Weaviate: DocumentChunk contains {count} objects")
                except Exception as e:
                    logger.warning(f"Could not get document count: {str(e)}")
            else:
                logger.warning("Weaviate: DocumentChunk collection does not exist - system may not find any documents")
        except Exception as e:
            logger.error(f"Error checking DocumentChunk collection: {str(e)}")
    
    # Validate Mistral API connection
    if not mistral_client:
        logger.error("CRITICAL: Mistral API client is not initialized")
    else:
        # Try a simple test query to validate API key and connectivity
        try:
            test_response = mistral_client.chat.complete(
                model=settings.MISTRAL_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            logger.info(f"Mistral API: Connection successful, using model {settings.MISTRAL_MODEL}")
        except Exception as e:
            logger.error(f"CRITICAL: Mistral API test failed: {str(e)}")
    
    # Log configuration
    logger.info(f"Configuration: DAILY_TOKEN_BUDGET={settings.DAILY_TOKEN_BUDGET}, MAX_REQUESTS_PER_MINUTE={settings.MAX_REQUESTS_PER_MINUTE}")
    logger.info("Startup validation complete")
    
    yield  # Here the app runs
    
    # Shutdown logic
    logger.info("Shutting down application...")

    # Flush any pending logs
    if chat_logger and chat_logger.enabled and hasattr(chat_logger, "_flush_buffer"):
        chat_logger._flush_buffer()    
    
    # Properly close the chat logger
    if chat_logger:
        await chat_logger.close()
    
    # Close Weaviate client if needed
    if client:
        client.close()

# Create FastAPI app with lifespan
app = FastAPI(
    title="EU-Compliant RAG API", 
    description="An EU-compliant RAG implementation using Weaviate and Mistral AI",
    version="1.0.0",
    lifespan=lifespan
)

# Register Vue dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(
    api_key: str = Depends(api_key_header)
) -> str:
    """
    Validate API key for protected endpoints.
    
    Args:
        api_key: API key from request header
        
    Returns:
        str: The validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    # Skip validation if no key file is configured
    if not os.path.exists(settings.INTERNAL_API_KEY_FILE):
        logger.warning(f"API key file not found: {settings.INTERNAL_API_KEY_FILE}")
        return api_key
    
    try:
        with open(settings.INTERNAL_API_KEY_FILE, "r") as f:
            expected_key = f.read().strip()
        
        if api_key != expected_key:
            raise HTTPException(
                status_code=403, 
                detail="Invalid API key"
            )
        
        return api_key
    except Exception as e:
        logger.error(f"Error validating API key: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Error validating API key"
        )

# Basic endpoints
@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "EU-Compliant RAG API is running"}

@app.get("/status")
async def status():
    """Check the status of the API and its connections."""
    weaviate_status = "connected" if client and client.is_ready() else "disconnected"
    
    return {
        "api": "running",
        "weaviate": weaviate_status,
        "mistral_api": "configured" if mistral_client else "not configured"
    }

# Document search endpoints
@app.post("/search")
async def search_documents(
    query: Query,
    api_key: str = Depends(get_api_key)
):
    """
    Search for relevant document chunks without LLM generation.
    
    Args:
        query: The search query
        api_key: API key for authentication
        
    Returns:
        dict: Search results
        
    Raises:
        HTTPException: If search fails
    """
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Search Weaviate for relevant chunks
        collection = client.collections.get("DocumentChunk")
        
        search_result = collection.query.near_text(
            query=query.question,
            limit=5,
            return_properties=["content", "filename", "chunkId", "metadataJson"]
        )
        
        # Format results
        results = []
        for obj in search_result.objects:
            results.append(obj.properties)
        
        return {
            "query": query.question,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/documents/count")
async def count_documents():
    """
    Count the number of unique documents indexed in the system.
    
    Returns:
        dict: Count of unique documents and their filenames
        
    Raises:
        HTTPException: If counting fails
    """
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Get the collection
        collection = client.collections.get("DocumentChunk")
        
        # Get all unique filenames
        query_result = collection.query.fetch_objects(
            return_properties=["filename"],
            limit=10000  # Use a reasonably high limit
        )
        
        # Count unique filenames
        unique_filenames = set()
        for obj in query_result.objects:
            unique_filenames.add(obj.properties["filename"])
        
        return {
            "count": len(unique_filenames),
            "documents": sorted(list(unique_filenames))
        }
        
    except Exception as e:
        logger.error(f"Error counting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/statistics")
async def get_document_statistics():
    """
    Get comprehensive statistics about documents in the system.
    
    Returns:
        dict: Document statistics including counts, metadata, and processing information
        
    Raises:
        HTTPException: If statistics gathering fails
    """
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Get the DocumentChunk collection
        collection = client.collections.get("DocumentChunk")
        
        # 1. Get all objects to gather statistics
        # Limited to 10,000 for practicality - adjust if needed
        query_result = collection.query.fetch_objects(
            return_properties=["filename", "chunkId", "content"],
            limit=10000
        )
        
        if not query_result.objects:
            return {
                "document_count": 0,
                "chunk_count": 0,
                "message": "No documents found in the system"
            }
        
        # 2. Calculate basic statistics
        document_chunks = {}
        total_content_length = 0
        
        for obj in query_result.objects:
            filename = obj.properties["filename"]
            chunk_id = obj.properties["chunkId"]
            content = obj.properties["content"]
            
            # Track chunks per document
            if filename not in document_chunks:
                document_chunks[filename] = []
            document_chunks[filename].append(chunk_id)
            
            # Track total content length
            total_content_length += len(content)
        
        # 3. Prepare document details
        documents = []
        for filename, chunks in document_chunks.items():
            documents.append({
                "filename": filename,
                "chunk_count": len(chunks),
                "first_chunk": min(chunks),
                "last_chunk": max(chunks)
            })
        
        # Sort documents by filename
        documents.sort(key=lambda x: x["filename"])
        
        # 4. Calculate summary statistics
        document_count = len(document_chunks)
        chunk_count = len(query_result.objects)
        avg_chunks_per_doc = chunk_count / max(document_count, 1)
        avg_chunk_length = total_content_length / max(chunk_count, 1)
        
        # 5. Compile and return the statistics
        return {
            "summary": {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "avg_chunks_per_document": round(avg_chunks_per_doc, 2),
                "avg_chunk_length": round(avg_chunk_length, 2),
                "total_content_length": total_content_length,
            },
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Error retrieving document statistics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")  

# Chat endpoint
@app.post("/chat", response_model=APIResponse)
async def chat(
    query: QueryWithHistory, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
    user_id: Optional[str] = Header(None)
):
    """
    RAG-based chat endpoint that queries documents and generates a response.
    
    Args:
        query: The user's question
        background_tasks: FastAPI background tasks
        api_key: API key for authentication
        user_id: Optional user identifier for logging
        
    Returns:
        APIResponse: Generated answer with source citations
        
    Raises:
        HTTPException: If the chat process fails
    """
    request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracing
    
    logger.info(f"[{request_id}] Chat request received: {query.question[:50] + '...' if len(query.question) > 50 else query.question}")
    logger.info(f"[{request_id}] Conversation history provided: {len(query.conversation_history) if query.conversation_history else 0} messages")

    # Check rate limit first
    if not check_rate_limit():
        return APIResponse(
            answer="The system is currently processing too many requests. Please try again in a minute.",
            sources=[],
            error="rate_limited"
        )    

    if not client:
        logger.error(f"[{request_id}] Weaviate connection not available")
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    if not mistral_client:
        logger.error(f"[{request_id}] Mistral API client not configured")
        raise HTTPException(status_code=503, detail="Mistral API client not configured")
    
    try:
        # Process current question with conversation context
        processed_question = query.question
        conversation_context = ""
        
        # Process conversation history if provided
        if query.conversation_history and len(query.conversation_history) > 0:
            # Build conversation context (last 3 interactions)
            recent_history = query.conversation_history[-3:] if len(query.conversation_history) > 3 else query.conversation_history
            conversation_context = "Previous conversation:\n"
            
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                conversation_context += f"{role.capitalize()}: {content}\n\n"

            # Attempt to expand references in the current question
            processed_question = expand_question_references(query.question, recent_history)

            # TODO: test optimized conversation history
            # # Alternatively, create optimized conversation history instead of using all history
            # conversation_context = create_optimized_history(
            #     query.conversation_history,
            # these numbers could be env parameters
            #     max_exchanges=3,  # Last 3 exchanges (6 messages)
            #     max_tokens=800    # Rough budget for conversation context
            # )
            # # Attempt to expand references in the current question
            # processed_question = expand_question_references(query.question, query.conversation_history)

            logger.info(f"[{request_id}] Processed question: {processed_question}")            

        # Get the collection
        collection = client.collections.get("DocumentChunk")
        
        # Search Weaviate for relevant chunks using v4 API
        # Create a hybrid query that includes context from recent conversation
        hybrid_query = processed_question
        if query.conversation_history and len(query.conversation_history) > 0:
            # Get the most recent user question for context
            recent_user_questions = [msg.get("content", "") 
                                    for msg in query.conversation_history[-3:] 
                                    if msg.get("role") == "user"]
            
            if recent_user_questions:
                # Combine current question with recent context
                # Weight current question higher (0.7) than context (0.3)
                hybrid_query = f"{processed_question} {' '.join(recent_user_questions)}"
                logger.info(f"[{request_id}] Using hybrid query for retrieval: {hybrid_query[:100]}...")

        # Use the hybrid query for retrieval
        search_result = collection.query.near_text(
            query=hybrid_query,
            limit=3,
            return_properties=["content", "filename", "chunkId", "metadataJson"]
        )

        # Check if we got any results
        if len(search_result.objects) == 0:
            return APIResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[]
            )        
        
        # Log search results
        logger.info(f"[{request_id}] Retrieved {len(search_result.objects)} relevant chunks")

        # Format context from chunks to highlight structure
        context_sections = []
        for obj in search_result.objects:
            metadata = json.loads(obj.properties.get("metadataJson", "{}"))
            heading = metadata.get("heading", "Untitled Section")
            page = metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            
            section_text = f"## {heading}{page_info}\n\n{obj.properties['content']}"
            context_sections.append(section_text)

        context = "\n\n".join(context_sections)
        
        logger.info(f"[{request_id}] Context size: {len(context)} characters")

        # Create a hash of the query and context to use as cache key
        query_text = query.question.strip().lower()
        context_hash = hashlib.md5(context.encode()).hexdigest()
        cache_key = f"{query_text}_{context_hash}"
        
        # Check cache first
        cached_result = get_cached_response(cache_key, settings.MISTRAL_MODEL)
        if cached_result:
            logger.info(f"[{request_id}] Cache hit! Returning cached response")
            return APIResponse(**cached_result)

        # Estimate tokens (very roughly - ~4 chars per token)
        estimated_prompt_tokens = (len(query.question) + len(context)) // 4
        estimated_response_tokens = 500  # Conservative estimate
        total_estimated_tokens = estimated_prompt_tokens + estimated_response_tokens
        
        # Check if we have budget
        if not check_token_budget(total_estimated_tokens):
            return APIResponse(
                answer="I'm sorry, the daily query limit has been reached to control costs. Please try again tomorrow.",
                sources=[],
                error="budget_exceeded"
            )
        
        # Log generation attempt
        logger.info(f"[{request_id}] Sending request to Mistral API using model: {settings.MISTRAL_MODEL}")
        
        start_time = time.time()        

        # Format sources for citation
        sources = []
        for obj in search_result.objects:
            source = {
                "filename": obj.properties["filename"], 
                "chunkId": obj.properties["chunkId"]
            }
            
            # Parse metadata JSON if it exists
            if "metadataJson" in obj.properties and obj.properties["metadataJson"]:
                try:
                    metadata = json.loads(obj.properties["metadataJson"])
                    source["metadata"] = metadata
                    
                    # Add page and heading if available
                    if "page" in metadata:
                        source["page"] = metadata["page"]
                    if "heading" in metadata:
                        source["heading"] = metadata["heading"]
                    if "headingLevel" in metadata:
                        source["headingLevel"] = metadata["headingLevel"]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse metadata JSON for {obj.properties['filename']}")            
            
            sources.append(source)
        
        # Use Mistral client to generate response, include the conversation context if required
        if conversation_context:
            # Improve the system prompt to be more context-aware
            system_prompt = """You are a helpful assistant that answers questions based on the provided document context. 
            Reference section headings when appropriate in your responses. 
            When answering follow-up questions, maintain consistency with your previous responses.
            If information is not in the provided context, say so rather than making up information.
            Start your responses directly by answering the question - do not begin with phrases like 'Based on the provided document context' or 'Based on our previous conversation'.
            Write in a natural, conversational tone."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Context from documents:
                 {context}
                 Previous conversation:
                 {conversation_context}
                 Current question: {query.question}
                 When answering, consider both the document context and our conversation history. 
                 If the current question refers to something we discussed earlier, use that information in your answer."""
                }            
            ]
        else:
            system_prompt =  """You are a helpful assistant that answers questions based on the provided document context. 
            Reference section headings when appropriate in your responses. Stick to the information in the context. 
            If you don't know the answer, say so.
            Start your responses directly by answering the question - do not begin with phrases like 'Based on the provided document context' or 'Based on our previous conversation'.
            Write in a natural, conversational tone."""            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Context:
                 {context}
                 Question: {query.question}"""
                }
            ]
        
        chat_response = await call_mistral_with_retry(
            client=mistral_client,
            model=settings.MISTRAL_MODEL,
            messages=messages,
            temperature=0.7,
        )
        
        answer = chat_response.choices[0].message.content

        generation_time = time.time() - start_time
        
        # Log success and timing
        logger.info(f"[{request_id}] Mistral response received in {generation_time:.2f}s")
        logger.info(f"[{request_id}] Answer length: {len(answer)} characters")     

        # Track actual usage (if available in Mistral response)
        tokens_used = 0
        if hasattr(chat_response, 'usage') and chat_response.usage:
            tokens_used = chat_response.usage.total_tokens
        else:
            # Fall back to estimation
            tokens_used = total_estimated_tokens
        
        update_token_usage(tokens_used)           

        # Cache the result before returning
        result = {"answer": answer, "sources": sources}
        set_cached_response(cache_key, settings.MISTRAL_MODEL, result)

        # Log the interaction in the background, using background_tasks to avoid delaying the response
        if chat_logger and chat_logger.enabled:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
            
            background_tasks.add_task(
                log_chat_interaction,
                query=query.question,
                response=result,
                request_id=request_id,
                user_id=user_id,
                metadata=metadata
            )
            
        return APIResponse(**result)
            
    except Exception as e:
        error_response = handle_api_error(e, request_id)
        return APIResponse(**error_response)

async def log_chat_interaction(
    query: str,
    response: Dict[str, Any],
    request_id: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Background task to log chat interactions.
    This runs asynchronously to avoid delaying the response.
    
    Args:
        query: The user's query
        response: The system's response
        request_id: Request identifier
        user_id: Optional user identifier
        metadata: Optional additional metadata
    """
    if chat_logger and chat_logger.enabled:
        try:
            # Don't use await here since log_interaction is not async
            chat_logger.log_interaction(
                query=query,
                response=response,
                request_id=request_id,
                user_id=user_id,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error logging chat interaction: {str(e)}")
    
@app.get("/privacy", response_class=HTMLResponse)
async def privacy_notice():
    """
    Serve the privacy notice.
    
    Returns:
        HTMLResponse: HTML content of the privacy notice
    """
    try:
        privacy_path = pathlib.Path("privacy_notice.html")
        if privacy_path.exists():
            return privacy_path.read_text(encoding="utf-8")
        else:
            logger.warning("privacy_notice.html not found, serving fallback notice")
            return """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Privacy Notice</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Chat Logging Privacy Notice</h1>
                    <p>When enabled, this system logs interactions for research purposes.</p>
                    <p>All data is processed in accordance with GDPR. Logs are automatically deleted after 30 days.</p>
                    <p>Please contact the system administrator for more information or to request deletion of your data.</p>
                </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving privacy notice: {str(e)}")
        return "<h1>Privacy Notice</h1><p>Error loading privacy notice.</p>"
        
@app.post("/feedback")
async def submit_feedback(
    feedback: FeedbackModel,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
    user_id: Optional[str] = Header(None)
):
    """
    Submit feedback on a previous response.
    
    Args:
        feedback: Feedback data
        background_tasks: FastAPI background tasks
        api_key: API key for authentication
        user_id: Optional user identifier
        
    Returns:
        dict: Acknowledgment
    """
    request_id = str(uuid.uuid4())[:8]  # Generate an ID for this feedback submission
    
    logger.info(f"[{request_id}] Received feedback for request {feedback.request_id}")
    logger.info(f"[{request_id}] Feedback details: {feedback.model_dump()}")
    
    # Process and store feedback
    if chat_logger and chat_logger.enabled:
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "original_request_id": feedback.request_id,
                "message_id": feedback.message_id
            }
            
            # Log feedback asynchronously
            background_tasks.add_task(
                log_feedback,
                feedback=feedback.model_dump(),
                request_id=request_id,
                user_id=user_id,
                metadata=metadata
            )
            
            return {
                "status": "success",
                "message": "Feedback received"
            }
        except Exception as e:
            logger.error(f"[{request_id}] Error storing feedback: {str(e)}")
            return {
                "status": "error",
                "message": "Failed to store feedback"
            }
    else:
        logger.warning(f"[{request_id}] Feedback received but logging is disabled")
        return {
            "status": "success",
            "message": "Feedback received but logging is disabled"
        }
    
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login")
async def login(login_request: LoginRequest):
    user = authenticate_user(login_request.username, login_request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "full_name": user.full_name
    }

@app.post("/admin/flush-logs")
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

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user    
    
# 1. First registered (executed last): Add security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Middleware to add security headers to responses.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        Response: The response with added security headers
    """
    response = await call_next(request)

    # Only add CSP headers for non-documentation endpoints
    if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
        
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# 2. Second registered (executed second): API key verification
@app.middleware("http")
async def verify_internal_api_key(request: Request, call_next):
    """
    Middleware to verify internal API key for protected endpoints.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        Response: The response
        
    Raises:
        HTTPException: If API key is invalid
    """
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
            # Use a proper exception here instead of returning directly
            raise HTTPException(status_code=403, detail="Invalid API key")
            
        # If we made it here, the key is valid
        return await call_next(request)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors but don't block the request
        logger.error(f"Error in API key validation: {str(e)}")
        return await call_next(request)
    
# 3. Last registered (executed first): Rate limiting
@app.middleware("http")
async def rate_limit_by_ip(request: Request, call_next):
    """
    Middleware to rate limit requests by IP address.
    
    Args:
        request: The incoming request
        call_next: The next middleware or route handler
        
    Returns:
        Response: The response
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Get client IP
    client_ip = request.client.host
    
    # Clean old timestamps
    now = time.time()
    ip_request_counters[client_ip] = [timestamp for timestamp in ip_request_counters[client_ip] 
                                     if now - timestamp < 60]
    
    # Check limits
    if len(ip_request_counters[client_ip]) >= settings.MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Add current timestamp
    ip_request_counters[client_ip].append(now)
    
    # Process request
    return await call_next(request)

# 4. Register exception handlers
@app.exception_handler(404)
async def not_found_exception(request, exc):
    """
    Exception handler for 404 Not Found errors.
    
    Args:
        request: The request that caused the exception
        exc: The exception
        
    Returns:
        JSONResponse: A JSON response with the error message
    """
    return JSONResponse(status_code=404, content={"error": "Not Found"})

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """
    Exception handler for Pydantic validation errors.
    
    Args:
        request: The request that caused the exception
        exc: The validation exception
        
    Returns:
        JSONResponse: A JSON response with validation error details
    """
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Exception handler for internal server errors.
    
    Args:
        request: The request that caused the exception
        exc: The exception
        
    Returns:
        JSONResponse: A JSON response with error information
    """
    # Log the error
    logger.error(f"Internal server error: {str(exc)}")
    import traceback
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later."
        }
    )
    
# Main entry point
if __name__ == "__main__":
    import uvicorn

    # Check secrets age
    check_secret_age(settings.MISTRAL_API_KEY_FILE)
    check_secret_age(settings.INTERNAL_API_KEY_FILE)
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)