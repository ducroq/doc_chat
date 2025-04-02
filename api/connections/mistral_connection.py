import asyncio
from logging import getLogger
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from mistralai import Mistral

from config import settings
from utils.errors import MistralError

logger = getLogger(__name__)

def create_mistral_client():
    if settings.MISTRAL_API_KEY:
        try:
            client = Mistral(api_key=settings.MISTRAL_API_KEY)
            logger.info("Mistral client initialized, using model: " + settings.MISTRAL_MODEL)
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {str(e)}")
            return None
    else:
        logger.warning("MISTRAL_API_KEY not set, Mistral client will not be initialized.")
        return None
    
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
            raise MistralError(
                message=f"Error calling Mistral API: {str(e)}", 
                error_type=error_type, 
                is_transient=False
            )        
