import logging
from datetime import datetime
import time
import uuid
import json
import hashlib
import weaviate
from mistralai import Mistral
from typing import Optional
from collections import deque
from fastapi import FastAPI, APIRouter, BackgroundTasks, Request, Depends, Header, HTTPException

from config import settings
from models.models import QueryWithHistory, APIResponse
from auth.auth_service import get_api_key 
from utils.chat_utils import log_chat_interaction, handle_api_error, get_cached_response, set_cached_response, expand_question_references
from utils.secret_utils import check_secret_age
from chat_logging.chat_logger import chat_logger
from connections.mistral_connection import call_mistral_with_retry

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state tracking
token_usage = {
    "count": 0,
    "reset_date": datetime.now().strftime("%Y-%m-%d")
}
request_timestamps = deque(maxlen=settings.MAX_REQUESTS_PER_MINUTE)
registration_timestamps = deque(maxlen=100)  # Keep last 100 timestamps for memory efficiency


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


@router.post("/chat", response_model=APIResponse)
async def chat(
    query: QueryWithHistory, 
    request: Request,
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
    weaviate_client = request.app.state.weaviate_client
    if not weaviate_client:
        logger.error(f"[{request_id}] Weaviate connection not available")
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    mistral_client = request.app.state.mistral_client
    if not mistral_client:
        logger.error(f"[{request_id}] Mistral API client not configured")
        raise HTTPException(status_code=503, detail="Mistral API client not configured")   
        
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
        collection = weaviate_client.collections.get("DocumentChunk")
        
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