from logging import getLogger
from datetime import datetime
from typing import Optional
import uuid
import os
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Header, Depends

from chat_logging.chat_logger import ChatLogger
from models.models import FeedbackModel
from auth.auth_service import get_api_key
from config import settings

router = APIRouter()
logger = getLogger(__name__)

def get_chat_logger():
    """Get or create a chat logger instance"""
    # Check environment variables directly
    enable_logging = os.getenv("ENABLE_CHAT_LOGGING", "false")
    logger.info(f"ENABLE_CHAT_LOGGING environment variable: '{enable_logging}'")
    
    if isinstance(enable_logging, str) and enable_logging.lower() in ["true", "1", "yes", "t"]:
        # Create a new logger instance if logging is enabled
        log_dir = settings.CHAT_LOG_DIR
        chat_logger = ChatLogger(log_dir=log_dir)
        return chat_logger
    else:
        return None

@router.post("/feedback")
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
    
    # Get a chat logger instance
    chat_logger = get_chat_logger()
    
    # Process and store feedback
    if chat_logger and chat_logger.enabled:
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "original_request_id": feedback.request_id,
                "message_id": feedback.message_id
            }
            
            # Log feedback directly (not as a background task to simplify)
            success = await chat_logger.alog_feedback(
                feedback=feedback.model_dump(),
                request_id=request_id,
                user_id=user_id,
                metadata=metadata
            )
            
            if success:
                return {
                    "status": "success",
                    "message": "Feedback recorded successfully"
                }
            else:
                return {
                    "status": "warning",
                    "message": "Feedback received but could not be stored"
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
