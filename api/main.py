from contextlib import asynccontextmanager
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logging_config import setup_logging, RequestLoggingMiddleware
from utils.secret_utils import check_secret_age
from connections.weaviate_connection import create_weaviate_client
from connections.mistral_connection import create_mistral_client
from endpoints import (
    search_endpoints,
    chat_endpoints,
    authentication_endpoints,
    feedback_endpoints,
    system_endpoints,
)
from middleware.middleware import (
    security_headers_middleware,
    api_key_middleware, 
    rate_limit_middleware
)

logger = setup_logging()
logger.info("Starting EU-Compliant RAG API application")

v1_prefix = "/api/v1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup validation
    logger.info("Performing startup validation...")

    # Initialize Weaviate client
    weaviate_client = create_weaviate_client()
    app.state.weaviate_client = weaviate_client

    # Initialize Mistral client
    mistral_client = create_mistral_client()
    app.state.mistral_client = mistral_client    
    
    # Validate Weaviate connection
    if not weaviate_client:
        logger.error("CRITICAL: Weaviate client is not initialized")
    elif not weaviate_client.is_ready():
        logger.error("CRITICAL: Weaviate is not ready")
    else:
        # Check if DocumentChunk collection exists
        try:
            if weaviate_client.collections.exists("DocumentChunk"):
                collection = weaviate_client.collections.get("DocumentChunk")
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

    # Flush logs using the same function as the endpoint
    try:
        flush_result = await system_endpoints.flush_logs(None)  # Pass None for the API key
        logger.info(f"Log flush result: {flush_result}")
    except Exception as e:
        logger.error(f"Error flushing logs during shutdown: {str(e)}")
    
    # Shutdown logic
    if weaviate_client:
        try:
            weaviate_client.close()
            logger.info("Weaviate client closed.")
        except Exception as e:
            logger.warning(f"Error closing Weaviate client: {e}")
    if mistral_client:
        try:
            logger.info("Mistral client session ended.")
        except Exception as e:
            logger.warning(f"Error closing Mistral client: {e}")

# Create FastAPI app with lifespan
app = FastAPI(
    title="EU-Compliant RAG API", 
    description="An EU-compliant RAG implementation using Weaviate and Mistral AI",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(search_endpoints.router, prefix=v1_prefix)
app.include_router(chat_endpoints.router, prefix=v1_prefix)
app.include_router(authentication_endpoints.router, prefix=v1_prefix)
app.include_router(feedback_endpoints.router, prefix=v1_prefix)
app.include_router(system_endpoints.router, prefix=v1_prefix)

# Register middleware in the correct order (execute in reverse order)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(rate_limit_middleware)
app.add_middleware(api_key_middleware)
# Security headers added last (will be applied first to responses)
app.add_middleware(security_headers_middleware)

# Add CORS middleware for Vue dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/", include_in_schema=False)
async def root_redirect():
    return {"message": "EU-Compliant RAG API is running. See /docs for API documentation."}
 
# Main entry point
if __name__ == "__main__":
    import uvicorn

    # # Check secrets age
    # check_secret_age(settings.MISTRAL_API_KEY_FILE)
    # check_secret_age(settings.INTERNAL_API_KEY_FILE)
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)    