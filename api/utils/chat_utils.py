from logging import getLogger
import time
import re
from typing import List, Optional, Dict, Any

from chat_logging.chat_logger import chat_logger
from config import settings
from utils.errors import MistralError, WeaviateError

logger = getLogger(__name__)

response_cache: Dict[str, Dict[str, Any]] = {}  # Dictionary to store cached responses

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
    
    if isinstance(e, MistralError):
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

