from logging import getLogger
import time
import re
import json
from typing import List, Optional, Dict, Any

from chat_logging.chat_logger import get_chat_logger
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
    chat_logger = get_chat_logger()
    if chat_logger and chat_logger.enabled:
        try:
            # Don't use await here since log_interaction is not async
            result = chat_logger.log_interaction(
                query=query,
                response=response,
                request_id=request_id,
                user_id=user_id,
                metadata=metadata
            )
            
            # Explicitly flush after each interaction
            if hasattr(chat_logger, "_flush_buffer"):
                chat_logger._flush_buffer()
                
        except Exception as e:
            logger.error(f"Error logging chat interaction: {str(e)}")

def is_related_to_history(current_question, conversation_history, similarity_threshold=0.3):
    """
    Check if the current question is related to recent conversation history.
    
    Args:
        current_question: The current user question
        conversation_history: List of previous message objects
        similarity_threshold: Threshold to consider questions related
        
    Returns:
        bool: True if the question appears related to recent history
    """
    # If no history, it's a new topic by default
    if not conversation_history or len(conversation_history) < 2:
        return False
    
    # Get the most recent user question
    recent_questions = [msg.get("content", "") 
                       for msg in conversation_history[-4:] 
                       if msg.get("role") == "user"]
    
    if not recent_questions:
        return False
    
    most_recent = recent_questions[0]
    
    # Simple keyword-based check
    current_words = set(current_question.lower().split())
    recent_words = set(most_recent.lower().split())
    
    # Calculate Jaccard similarity (intersection over union)
    if len(current_words) == 0 or len(recent_words) == 0:
        return False
        
    intersection = len(current_words.intersection(recent_words))
    union = len(current_words.union(recent_words))
    similarity = intersection / union
    
    # Log for debugging
    logger.debug(f"Question similarity: {similarity:.4f} between '{current_question}' and '{most_recent}'")
    
    return similarity > similarity_threshold            

def is_topic_related_to_history(current_question: str, conversation_history: List[Dict[str, Any]], 
                                similarity_threshold: float = 0.15) -> bool:
    """
    Determine if the current question is related to the conversation history.
    
    Args:
        current_question: The current question from the user
        conversation_history: List of previous conversation messages
        similarity_threshold: Threshold to consider questions related
        
    Returns:
        bool: True if the question is related to conversation history
    """
    # If no history, it's a new topic by default
    if not conversation_history or len(conversation_history) < 2:
        return False
    
    # Get recent user questions
    recent_questions = [msg.get("content", "") 
                       for msg in conversation_history[-4:] 
                       if msg.get("role") == "user"]
    
    if not recent_questions:
        return False
    
    # Normalize texts for comparison
    def normalize_text(text):
        # Convert to lowercase and remove punctuation
        normalized = ''.join(c for c in text.lower() if c.isalnum() or c.isspace())
        return normalized
    
    current_normalized = normalize_text(current_question)
    current_terms = set(current_normalized.split())
    
    # Significant terms are those with length > 2 (skip "is", "to", etc.)
    significant_terms = {term for term in current_terms if len(term) > 2}
    
    # Check for term overlap with recent questions
    for recent_q in recent_questions:
        recent_normalized = normalize_text(recent_q)
        recent_terms = set(recent_normalized.split())
        
        # Check overlap of significant terms
        overlap = significant_terms.intersection(recent_terms)
        if significant_terms and len(overlap) > 0:
            return True
    
    # For very short queries, also calculate Jaccard similarity with most recent
    most_recent = recent_questions[0]
    most_recent_normalized = normalize_text(most_recent)
    
    current_words = set(current_normalized.split())
    recent_words = set(most_recent_normalized.split())
    
    if current_words and recent_words:
        intersection = len(current_words.intersection(recent_words))
        union = len(current_words.union(recent_words))
        similarity = intersection / union
        
        return similarity > similarity_threshold
    
    return False

def filter_results_by_relevance(search_result, request_id, min_certainty=0.4):
    """
    Filter search results based on relevance score.
    
    Args:
        search_result: Result from Weaviate query
        request_id: Request identifier for logging
        min_certainty: Minimum certainty threshold (0-1)
        
    Returns:
        List: Filtered list of relevant search results
    """
    filtered_results = []
    
    for obj in search_result.objects:
        if obj.metadata:
            certainty = obj.metadata.certainty if hasattr(obj.metadata, 'certainty') else 0
            
            if certainty > min_certainty:
                filtered_results.append(obj)
                logger.info(f"[{request_id}] Added object with certainty: {certainty:.4f}")
    
    # Fallback: If no results meet threshold but we have results, take top 2
    if not filtered_results and search_result.objects:
        # Sort objects by certainty
        sorted_objects = sorted(
            [o for o in search_result.objects if o.metadata], 
            key=lambda x: x.metadata.certainty if hasattr(x.metadata, 'certainty') else 0, 
            reverse=True
        )
        
        # Take top result
        filtered_results = sorted_objects[:1]
        logger.info(f"[{request_id}] Using fallback with top {len(filtered_results)} result(s)")
    
    logger.info(f"[{request_id}] Retrieved {len(filtered_results)} relevant chunks")
    return filtered_results

def format_context_for_llm(filtered_results, request_id):
    """
    Format the retrieved chunks into a context string for the LLM.
    
    Args:
        filtered_results: Filtered search results
        request_id: Request identifier for logging
        
    Returns:
        str: Formatted context
    """
    context_sections = []
    
    for obj in filtered_results:
        metadata = json.loads(obj.properties.get("metadataJson", "{}"))
        heading = metadata.get("heading", "Untitled Section")
        page = metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        
        section_text = f"## {heading}{page_info}\n\n{obj.properties['content']}"
        context_sections.append(section_text)

    context = "\n\n".join(context_sections)
    
    logger.info(f"[{request_id}] Context size: {len(context)} characters")
    return context