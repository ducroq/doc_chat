from logging import getLogger
from datetime import datetime
from typing import Optional
import uuid
from fastapi import APIRouter, BackgroundTasks, Header, Depends

from chat_logging.chat_logger import chat_logger
from models.models import FeedbackModel
from auth.auth_service import get_api_key

router = APIRouter()
logger = getLogger(__name__)

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
    
    # Debug logs to check chat_logger status
    if chat_logger is None:
        logger.warning(f"[{request_id}] Chat logger is None")
    else:
        logger.info(f"[{request_id}] Chat logger status - exists: True, enabled: {chat_logger.enabled}")
    
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
                chat_logger.log_feedback,
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