import re
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List, Any

from config import settings
from utils.utils import validate_user_input_content


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

class RegisterUser(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    email: Optional[str] = None
    
    @field_validator('username')
    @classmethod
    def username_must_be_valid(cls, v: str) -> str:
        if not v or not re.match(r'^[a-zA-Z0-9_-]{3,16}$', v):
            raise ValueError("Username must be 3-16 characters and contain only letters, numbers, underscores, and hyphens")
        return v
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', v):
            raise ValueError("Invalid email format")
        return v
    
class LogEntry(BaseModel):
    """Data model for chat log entries with validation."""
    timestamp: str
    request_id: str
    user_id: Optional[str] = None
    query: str
    response: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        """Validate that query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @field_validator('timestamp')
    @classmethod
    def timestamp_must_be_iso_format(cls, v: str) -> str:
        """Validate timestamp is in ISO format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Timestamp must be in ISO format")

class FeedbackEntry(BaseModel):
    """Data model for feedback log entries with validation."""
    timestamp: str
    request_id: str
    original_request_id: str
    message_id: str
    user_id: Optional[str] = None
    rating: str
    feedback_text: Optional[str] = None
    categories: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_validator('rating')
    @classmethod
    def rating_must_be_valid(cls, v: str) -> str:
        """Validate that rating is positive or negative."""
        if v not in ["positive", "negative"]:
            raise ValueError('Rating must be "positive" or "negative"')
        return v

    @field_validator('timestamp')
    @classmethod
    def timestamp_must_be_iso_format(cls, v: str) -> str:
        """Validate timestamp is in ISO format."""
        try:
            datetime.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError("Timestamp must be in ISO format")
