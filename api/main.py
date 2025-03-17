import os
import re
import json
import time
import uuid
import logging
import hashlib
import pathlib
import asyncio
from datetime import datetime
from collections import deque
from functools import lru_cache
from typing import Optional, Dict, List, Any, Union, Tuple, Set, Annotated
from pydantic import BaseModel, Field, field_validator, ValidationError
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import APIKeyHeader
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
    query: Query, 
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
        # Get the collection
        collection = client.collections.get("DocumentChunk")
        
        # Search Weaviate for relevant chunks using v4 API
        search_result = collection.query.near_text(
            query=query.question,
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
        
        # Use Mistral client to generate response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context. Reference section headings when appropriate in your responses. Stick to the information in the context. If you don't know the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.question}"}
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
            await chat_logger.log_interaction(
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