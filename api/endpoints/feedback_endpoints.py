from logging import getLogger
from datetime import datetime
from typing import Optional
import uuid
import os
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Header, Depends

from chat_logging.chat_logger import get_chat_logger
from models.models import FeedbackModel
from auth.auth_service import get_api_key
from config import settings

router = APIRouter()
logger = getLogger(__name__)

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackModel,
    api_key: str = Depends(get_api_key),
    user_id: Optional[str] = Header(None)
):
    """Submit feedback on a previous response."""
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[{request_id}] Received feedback for request {feedback.request_id}")
    logger.info(f"[{request_id}] Feedback details: {feedback.model_dump()}")
    
    # Get a chat logger instance from the central factory
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
            
            # Log feedback directly
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
