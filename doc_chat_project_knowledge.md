# Project Documentation

# Project Structure
```
doc_chat/
├── api/
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── user_manager.py
│   ├── chat_logging/
│   │   ├── __init__.py
│   │   ├── chat_logger.py
│   ├── connections/
│   │   ├── __init__.py
│   │   ├── mistral_connection.py
│   │   ├── weaviate_connection.py
│   ├── endpoints/
│   │   ├── __init__.py
│   │   ├── authentication_endpoints.py
│   │   ├── chat_endpoints.py
│   │   ├── feedback_endpoints.py
│   │   ├── privacy_endpoints.py
│   │   ├── search_endpoints.py
│   │   ├── system_endpoints.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── middleware.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── chat_utils.py
│   │   ├── errors.py
│   │   ├── logging_config.py
│   │   ├── secret_utils.py
│   │   ├── utils.py
│   ├── Dockerfile
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── privacy_notice.html
│   ├── requirements.txt
├── processor/
│   ├── Dockerfile
│   ├── processor.py
│   ├── requirements.txt
├── tests/
│   ├── quickstart_locally_hosted/
│   │   ├── docker-compose.yml
│   │   ├── quickstart_check_readiness.py
│   │   ├── quickstart_create_collection.py
│   │   ├── quickstart_import.py
│   │   ├── quickstart_neartext_query.py
│   │   ├── quickstart_rag.py
│   ├── direct_weaviate_check.py
│   ├── document_storage_verification.py
│   ├── pdf-extraction-tests.py
├── vue-frontend/
│   ├── public/
│   │   ├── document-chat-icon.svg
│   │   ├── index.html
│   │   ├── vite.svg
│   ├── src/
│   │   ├── assets/
│   │   │   ├── main.css
│   │   │   ├── vue.svg
│   │   ├── components/
│   │   │   ├── chat/
│   │   │   │   ├── ChatInput.vue
│   │   │   │   ├── ChatMessage.vue
│   │   │   ├── layout/
│   │   │   │   ├── Sidebar.vue
│   │   │   ├── shared/
│   │   │   │   ├── Loading.vue
│   │   ├── router/
│   │   │   ├── index.js
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   ├── authService.js
│   │   │   ├── chatService.js
│   │   ├── stores/
│   │   │   ├── chat.js
│   │   ├── views/
│   │   │   ├── ChatView.vue
│   │   │   ├── LoginView.vue
│   │   │   ├── PrivacyView.vue
│   │   │   ├── RegisterView.vue
│   │   ├── App.vue
│   │   ├── main.js
│   │   ├── style.css
│   ├── Dockerfile
│   ├── README.md
│   ├── entrypoint.sh
│   ├── index.html
│   ├── nginx.conf
│   ├── package-lock.json
│   ├── package.json
│   ├── vite.config.js
├── LICENSE
├── README.md
├── docker-compose.yml
├── manage_users.py
├── start.ps1
├── start.sh
├── stop.ps1
├── stop.sh
```

# api\auth\__init__.py
```python

```


# api\auth\auth_service.py
```python
import os
from logging import getLogger
import jwt
from jwt.exceptions import PyJWTError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader

from config import settings
from models.models import User, TokenData
from auth.user_manager import get_user, verify_password 

logger = getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
api_key_header = APIKeyHeader(name="X-API-Key")

async def authenticate_user(username: str, password: str):
    user = get_user(None, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(None, username=token_data.username)  # Remove the first parameter
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_api_key(api_key: str = Depends(api_key_header)) -> str:
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
    
```


# api\auth\user_manager.py
```python
import os
import json
import bcrypt
from models.models import UserInDB
from logging import getLogger

logger = getLogger(__name__)


def load_users_from_json():
    users_file_path = 'users.json'
    try:
        if os.path.exists(users_file_path):
            with open(users_file_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Users file not found at {users_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading users from JSON: {str(e)}")
        return {}
    
def get_users_db():
    return load_users_from_json()

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_user(db, username: str):
    # Load fresh user data each time to catch updates
    users_db = get_users_db()
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None
```


# api\chat_logging\__init__.py
```python

```


# api\chat_logging\chat_logger.py
```python
import os
from logging import getLogger
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import settings
from models.models import LogEntry, FeedbackEntry

logger = getLogger(__name__)

# Initialize chat logger
chat_logger = None

class ChatLogger:
    """
    Privacy-compliant chat logger for research purposes.
    Implements GDPR requirements including opt-in logging, log rotation,
    anonymization, and deletion capabilities.
    
    Logs are saved to a local folder within the project with configurable
    retention periods and privacy controls.
    """
    
    def __init__(
        self, 
        log_dir: str = settings.CHAT_LOG_DIR,
        buffer_size: int = settings.CHAT_LOG_BUFFER_SIZE
    ):
        """
        Initialize the chat logger with privacy controls.
        """
        self.log_dir = Path(log_dir)
        
        # Check environment variables for settings
        enable_logging = os.getenv("ENABLE_CHAT_LOGGING", "false")
        logger.info(f"ENABLE_CHAT_LOGGING environment variable: '{enable_logging}'")
        
        # Convert to boolean with proper string handling
        if isinstance(enable_logging, str):
            self.enabled = enable_logging.lower() in ["true", "1", "yes", "t"]
        else:
            self.enabled = bool(enable_logging)
            
        self.anonymize = os.getenv("ANONYMIZE_CHAT_LOGS", "true").lower() == "true"
        self.retention_days = int(os.getenv("LOG_RETENTION_DAYS", str(settings.LOG_RETENTION_DAYS)))
        self.buffer_size = buffer_size
        self.log_buffer: List[str] = []
        self.feedback_buffer: List[str] = []

        if self.enabled:
            try:
                self.log_dir.mkdir(exist_ok=True, parents=True)
                self.log_file = self.log_dir / f"chat_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
                logger.info(f"Chat logging enabled. Logs will be saved to {self.log_file}")
                logger.info(f"Log anonymization: {self.anonymize}, Retention period: {self.retention_days} days")
                
                # Run initial rotation to clean up old logs
                self._rotate_logs()
            except PermissionError as e:
                logger.error(f"Permission error creating log directory: {str(e)}")
                self.enabled = False
            except OSError as e:
                logger.error(f"OS error initializing logger: {str(e)}")
                self.enabled = False
    
    def log_interaction(
        self, 
        query: str, 
        response: Dict[str, Any], 
        request_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log a chat interaction to the log file with privacy controls.
        
        Args:
            query: The user's query
            response: The system's response
            request_id: Unique ID for the request
            user_id: Optional user identifier (will be anonymized if anonymization is enabled)
            metadata: Optional additional metadata about the interaction
            
        Returns:
            bool: Whether logging was successful
            
        Raises:
            ValueError: If any of the required parameters are invalid
        """
        if not self.enabled:
            return False
        
        # Input validation
        if not query or not query.strip():
            logger.warning("Attempted to log empty query")
            return False
            
        if not isinstance(response, dict):
            logger.warning(f"Invalid response format: {type(response)}")
            return False
        
        # Apply anonymization if enabled
        anonymized_user_id = None
        if self.anonymize and user_id:
            try:
                # Create a deterministic but anonymized ID
                anon_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user_id))
                anonymized_user_id = f"{self.ANONYMIZE_PREFIX}{anon_id[-12:]}"
            except (TypeError, ValueError) as e:
                logger.warning(f"Error anonymizing user ID: {str(e)}")
        
        # Create log entry
        try:
            # Create minimized sources data for logging
            minimized_sources = []
            for source in response.get("sources", []):
                # Extract only essential source information
                minimal_source = {
                    "filename": source.get("filename"),
                    "chunkId": source.get("chunkId")
                }
                
                # Add section and page if available
                if "heading" in source:
                    minimal_source["heading"] = source.get("heading")
                if "page" in source:
                    minimal_source["page"] = source.get("page")
                
                # Add minimal metadata if needed
                if "metadata" in source and isinstance(source["metadata"], dict):
                    minimal_metadata = {}
                    # Just preserve the document type and title
                    if "itemType" in source["metadata"]:
                        minimal_metadata["itemType"] = source["metadata"]["itemType"]
                    if "title" in source["metadata"]:
                        minimal_metadata["title"] = source["metadata"]["title"]
                    if minimal_metadata:
                        minimal_source["metadata"] = minimal_metadata
                
                minimized_sources.append(minimal_source)

            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                request_id=request_id or str(uuid.uuid4()),
                user_id=anonymized_user_id if self.anonymize and user_id else user_id,
                query=query,
                response={
                    "answer": response.get("answer"),
                    "sources": minimized_sources
                },
                metadata=metadata
            )            
            return self._write_log_entry(log_entry.model_dump())  # Using model_dump() instead of dict()
        except ValueError as e:
            logger.error(f"Error creating log entry: {str(e)}")
            return False
        
    def log_feedback(
        self, 
        feedback: Dict[str, Any],
        request_id: Optional[str] = None, 
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log feedback on a chat interaction with privacy controls.
        
        Args:
            feedback: The feedback data
            request_id: Unique ID for this feedback submission
            user_id: Optional user identifier (will be anonymized if anonymization is enabled)
            metadata: Optional additional metadata about the feedback
            
        Returns:
            bool: Whether logging was successful
            
        Raises:
            ValueError: If any of the required parameters are invalid
        """
        if not self.enabled:
            return False
        
        # Input validation
        if not feedback:
            logger.warning("Attempted to log empty feedback")
            return False
            
        if not isinstance(feedback, dict):
            logger.warning(f"Invalid feedback format: {type(feedback)}")
            return False
        
        # Apply anonymization if enabled
        anonymized_user_id = None
        if self.anonymize and user_id:
            try:
                # Create a deterministic but anonymized ID
                anon_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user_id))
                anonymized_user_id = f"{self.ANONYMIZE_PREFIX}{anon_id[-12:]}"
            except (TypeError, ValueError) as e:
                logger.warning(f"Error anonymizing user ID: {str(e)}")
        
        # Create feedback entry
        try:
            # Extract data from feedback dict
            original_request_id = feedback.get("request_id", "unknown")
            message_id = feedback.get("message_id", "unknown")
            rating = feedback.get("rating", "unknown")
            feedback_text = feedback.get("feedback_text")
            categories = feedback.get("categories", [])
            
            # Create the feedback entry
            feedback_entry = FeedbackEntry(
                timestamp=datetime.now().isoformat(),
                request_id=request_id or str(uuid.uuid4()),
                original_request_id=original_request_id,
                message_id=message_id,
                user_id=anonymized_user_id if self.anonymize and user_id else user_id,
                rating=rating,
                feedback_text=feedback_text,
                categories=categories,
                metadata=metadata
            )
            
            # Create filename based on date
            current_date = datetime.now().strftime('%Y%m%d')
            feedback_file = self.log_dir / f"feedback_log_{current_date}.jsonl"
            
            # Format as JSON line
            log_line = json.dumps(feedback_entry.model_dump()) + "\n"
            
            # Add to feedback buffer
            self.feedback_buffer.append(log_line)
            
            # If buffer reaches threshold, flush it
            if len(self.feedback_buffer) >= self.buffer_size:
                self._flush_feedback_buffer()
            
            logger.info(f"Feedback queued: {rating} for request {original_request_id}")
            return True
                
        except ValueError as e:
            logger.error(f"Error creating feedback entry: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error logging feedback: {str(e)}")
            return False
        
    def _anonymize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize potentially sensitive information in document sources.
        
        Args:
            sources: List of source documents used for generating response
            
        Returns:
            List[Dict[str, Any]]: Anonymized source references
        """
        if not self.anonymize:
            return sources
            
        # Create a copy to avoid modifying the original
        anonymized_sources = []
        
        for source in sources:
            # Create a copy of the source
            anon_source = source.copy()
            
            # Anonymize metadata if present
            if 'metadata' in anon_source:
                metadata = anon_source['metadata']
                
                # Remove any potentially sensitive fields
                if 'creators' in metadata:
                    # Keep creator types but remove specific names
                    anon_creators = []
                    for creator in metadata['creators']:
                        anon_creator = {'creatorType': creator.get('creatorType', 'unknown')}
                        anon_creators.append(anon_creator)
                    metadata['creators'] = anon_creators
                
                # Remove email addresses if present
                if 'email' in metadata:
                    del metadata['email']
                
                # Remove other potentially sensitive fields
                sensitive_fields = {'phone', 'address', 'personal_notes', 'user_comments'}
                for field in sensitive_fields.intersection(set(metadata.keys())):
                    del metadata[field]
            
            anonymized_sources.append(anon_source)
            
        return anonymized_sources

    def _write_log_entry(self, log_entry: Dict[str, Any]) -> bool:
        """
        Write a log entry to the file, with buffering for efficiency.
        
        Args:
            log_entry: The log entry to write
            
        Returns:
            bool: Whether the write was successful
        """
        try:
            # Ensure log directory exists
            self.log_dir.mkdir(exist_ok=True, parents=True)
            
            # Format as JSON line
            log_line = json.dumps(log_entry) + "\n"
            
            # Add to buffer
            self.log_buffer.append(log_line)
            
            # If buffer reaches threshold or is forced, write to file
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_buffer()
            
            # Handle log rotation if needed
            self._rotate_logs()
            return True
            
        except (IOError, PermissionError) as e:
            logger.error(f"Error writing to log file: {str(e)}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error serializing log entry to JSON: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error logging chat interaction: {str(e)}")
            return False

    def _flush_buffer(self) -> None:
        """Flush the current buffer to the log file."""
        if not self.log_buffer:
            return
            
        try:
            # Get current date to make sure we're writing to the right file
            current_date = datetime.now().strftime('%Y%m%d')
            self.log_file = self.log_dir / f"chat_log_{current_date}.jsonl"
            
            # Append buffered content to file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.writelines(self.log_buffer)
            
            # Clear buffer after successful write
            self.log_buffer = []
        except Exception as e:
            logger.error(f"Error flushing log buffer: {str(e)}")

    def _flush_feedback_buffer(self) -> None:
        """Flush the current feedback buffer to the log file."""
        if not self.feedback_buffer:
            return
            
        try:
            # Get current date to make sure we're writing to the right file
            current_date = datetime.now().strftime('%Y%m%d')
            feedback_file = self.log_dir / f"feedback_log_{current_date}.jsonl"
            
            # Ensure log directory exists
            self.log_dir.mkdir(exist_ok=True, parents=True)
            
            # Append buffered content to file
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.writelines(self.feedback_buffer)
            
            # Clear buffer after successful write
            logger.info(f"Flushed {len(self.feedback_buffer)} feedback entries to disk")
            self.feedback_buffer = []
        except Exception as e:
            logger.error(f"Error flushing feedback buffer: {str(e)}")

    def _rotate_logs(self) -> None:
        """
        Implement log rotation by deleting logs older than the retention period.
        """
        if not self.enabled or self.retention_days <= 0:
            return
            
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Check all log files in the directory
            deleted_count = 0
            for log_file in self.log_dir.glob("chat_log_*.jsonl"):
                try:
                    # Extract date from filename
                    date_str = log_file.stem.replace("chat_log_", "")
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    # Delete if older than retention period
                    if file_date < cutoff_date:
                        logger.info(f"Deleting old log file: {log_file}")
                        log_file.unlink()
                        deleted_count += 1
                except (ValueError, OSError) as e:
                    logger.warning(f"Error processing log file {log_file}: {str(e)}")
                    
            if deleted_count > 0:
                logger.info(f"Log rotation complete. Deleted {deleted_count} old log files.")
                
        except Exception as e:
            logger.error(f"Error during log rotation: {str(e)}")

    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all log entries related to a specific user ID (GDPR right to erasure).
        
        Args:
            user_id: The user ID to remove from logs
            
        Returns:
            bool: Whether the deletion was successful
        """
        if not self.enabled or not user_id:
            return False
            
        # Input validation
        if not isinstance(user_id, str) or not user_id.strip():
            logger.warning("Invalid user_id provided for deletion")
            return False
            
        success = True
        
        try:
            # Flush any pending logs first
            self._flush_buffer()
            
            # If anonymization is enabled, calculate the anonymized ID
            target_id = user_id
            if self.anonymize:
                anon_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user_id))
                target_id = f"{self.ANONYMIZE_PREFIX}{anon_id[-12:]}"
            
            # Process each log file
            for log_file in self.log_dir.glob("chat_log_*.jsonl"):
                try:
                    # Create a temporary file
                    temp_file = log_file.with_suffix(".tmp")
                    
                    # Load the entire file
                    lines_kept = 0
                    lines_removed = 0
                    
                    # Read original file
                    with open(log_file, 'r', encoding='utf-8') as input_file:
                        content = input_file.read()
                        lines = content.splitlines()
                    
                    # Filter out entries for this user
                    with open(temp_file, 'w', encoding='utf-8') as output_file:
                        for line in lines:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("user_id") != target_id:
                                    output_file.write(line + '\n')
                                    lines_kept += 1
                                else:
                                    lines_removed += 1
                            except json.JSONDecodeError:
                                # Keep lines that can't be parsed
                                output_file.write(line + '\n')
                                lines_kept += 1
                    
                    # Replace original with filtered version
                    temp_file.replace(log_file)
                    
                    if lines_removed > 0:
                        logger.info(f"Removed {lines_removed} entries for user {user_id} from {log_file}")
                    
                except Exception as e:
                    logger.error(f"Error processing file {log_file} during user data deletion: {str(e)}")
                    success = False
            
            logger.info(f"Completed deletion of user data for user ID: {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error during user data deletion: {str(e)}")
            return False
            
    def get_log_files(
        self, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> List[Path]:
        """
        Get list of log files, optionally filtered by date range.
        
        Args:
            start_date: Start date in YYYYMMDD format (inclusive)
            end_date: End date in YYYYMMDD format (inclusive)
            
        Returns:
            List[Path]: Paths to the log files
        """
        if not self.enabled:
            return []
            
        # Convert strings to date objects if provided
        start_date_obj = None
        end_date_obj = None
        
        if start_date:
            try:
                start_date_obj = datetime.strptime(start_date, "%Y%m%d")
            except ValueError:
                logger.error(f"Invalid start date format: {start_date}. Use YYYYMMDD.")
                return []
                
        if end_date:
            try:
                end_date_obj = datetime.strptime(end_date, "%Y%m%d")
            except ValueError:
                logger.error(f"Invalid end date format: {end_date}. Use YYYYMMDD.")
                return []
        
        matching_files = []
        
        # Find matching log files
        for log_file in self.log_dir.glob("chat_log_*.jsonl"):
            try:
                # Extract date from filename
                date_str = log_file.stem.replace("chat_log_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                # Check if file is within date range
                if (start_date_obj and file_date < start_date_obj) or \
                   (end_date_obj and file_date > end_date_obj):
                    continue
                
                matching_files.append(log_file)
                
            except ValueError as e:
                logger.warning(f"Error processing log file {log_file}: {str(e)}")
        
        # Sort by date
        matching_files.sort()
        return matching_files
    
    def close(self) -> None:
        """
        Flush any buffered logs and clean up resources.
        Should be called when shutting down the application.
        """
        if self.enabled:
            if self.log_buffer:
                self._flush_buffer()
            if self.feedback_buffer:
                self._flush_feedback_buffer()
            logger.info("Chat logger closed, all logs flushed to disk")

    # Async compatibility methods for projects that want to use async
    async def alog_interaction(self, *args, **kwargs) -> bool:
        """Async-compatible wrapper for log_interaction."""
        result = self.log_interaction(*args, **kwargs)
        return result  # Return the bool directly, don't make it a coroutine

    async def adelete_user_data(self, *args, **kwargs) -> bool:
        """Async-compatible wrapper for delete_user_data."""
        return self.delete_user_data(*args, **kwargs)

    async def aget_log_files(self, *args, **kwargs) -> List[Path]:
        """Async-compatible wrapper for get_log_files."""
        return self.get_log_files(*args, **kwargs)

    async def aclose(self) -> None:
        """Async-compatible wrapper for close."""
        self.close()

    async def alog_feedback(self, *args, **kwargs) -> bool:
        """Async-compatible wrapper for log_feedback."""
        return self.log_feedback(*args, **kwargs)
    
```


# api\connections\__init__.py
```python

```


# api\connections\mistral_connection.py
```python
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

```


# api\connections\weaviate_connection.py
```python
from logging import getLogger
from typing import Optional
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from fastapi import FastAPI, HTTPException, Request, Form

from config import settings

logger = getLogger(__name__)

def create_weaviate_client():
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
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {str(e)}")
        return None
    
```


# api\endpoints\__init__.py
```python

```


# api\endpoints\authentication_endpoints.py
```python
from logging import getLogger
import jwt
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from config import settings
from models.models import User, LoginRequest
from auth.auth_service import authenticate_user, get_current_active_user
from models.models import Token

router = APIRouter()
logger = getLogger(__name__)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Make sure to await the authenticate_user coroutine
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login")
async def login(login_request: LoginRequest):
    # Make sure to await the authenticate_user coroutine
    user = await authenticate_user(login_request.username, login_request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "full_name": user.full_name
    }

@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
```


# api\endpoints\chat_endpoints.py
```python
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
```


# api\endpoints\feedback_endpoints.py
```python
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

```


# api\endpoints\privacy_endpoints.py
```python
from logging import getLogger
import pathlib

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
logger = getLogger(__name__)

@router.get("/privacy", response_class=HTMLResponse)
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
```


# api\endpoints\search_endpoints.py
```python
from logging import getLogger
from fastapi import APIRouter, HTTPException, Request, Depends

from models.models import Query
from auth.auth_service import get_api_key

router = APIRouter()
logger = getLogger(__name__)

@router.post("/search")
async def search_documents(
    query: Query,
    request: Request,
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
    weaviate_client = request.app.state.weaviate_client
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Search Weaviate for relevant chunks
        collection = weaviate_client.collections.get("DocumentChunk")
        
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
    
@router.get("/documents/count")
async def count_documents(request: Request):
    """
    Count the number of unique documents indexed in the system.
    
    Returns:
        dict: Count of unique documents and their filenames
        
    Raises:
        HTTPException: If counting fails
    """
    weaviate_client = request.app.state.weaviate_client
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Get the collection
        collection = weaviate_client.collections.get("DocumentChunk")
        
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
    
@router.get("/statistics")
async def get_document_statistics(request: Request):
    """
    Get comprehensive statistics about documents in the system.
    
    Returns:
        dict: Document statistics including counts, metadata, and processing information
        
    Raises:
        HTTPException: If statistics gathering fails
    """
    weaviate_client = request.app.state.weaviate_client
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Get the DocumentChunk collection
        collection = weaviate_client.collections.get("DocumentChunk")
        
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
    
```


# api\endpoints\system_endpoints.py
```python
from logging import getLogger
import random
import time
import hashlib
import json
import bcrypt
from fastapi import HTTPException, APIRouter, Depends, Request, Form

from auth.auth_service import get_api_key
from auth.user_manager import get_users_db
from chat_logging.chat_logger import ChatLogger

router = APIRouter()
logger = getLogger(__name__)

def generate_math_captcha():
    """Generate a simple math problem as CAPTCHA"""
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    operation = random.choice(['+', '-', '*'])
    
    if operation == '+':
        answer = a + b
        question = f"{a} + {b}"
    elif operation == '-':
        # Ensure positive result
        if b > a:
            a, b = b, a
        answer = a - b
        question = f"{a} - {b}"
    else:  # multiplication
        answer = a * b
        question = f"{a} × {b}"
    
    # Create a hash of the answer with a time-based salt
    timestamp = int(time.time())
    answer_hash = hashlib.sha256(f"{answer}:{timestamp}".encode()).hexdigest()
    
    return {
        "question": question,
        "hash": answer_hash,
        "timestamp": timestamp
    }

def verify_math_captcha(user_answer, answer_hash, timestamp):
    """Verify the math CAPTCHA answer"""
    # Ensure timestamp is an integer
    try:
        timestamp = int(timestamp)
        current_time = int(time.time())
        
        # Check if CAPTCHA has expired (10 minutes)
        if current_time - timestamp > 600:  # This is where the error was happening
            return False
        
        # Convert user_answer to int and verify
        user_answer = int(user_answer)
        check_hash = hashlib.sha256(f"{user_answer}:{timestamp}".encode()).hexdigest()
        return check_hash == answer_hash
    except (ValueError, TypeError) as e:
        # Log the error for debugging
        print(f"CAPTCHA verification error: {e}")
        return False

@router.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "EU-Compliant RAG API is running"}

@router.get("/status")
async def status(request: Request):
    """Check the status of the API and its connections."""
    weaviate_client = request.app.state.weaviate_client
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    mistral_client = request.app.state.mistral_client
    if not mistral_client:
        raise HTTPException(status_code=503, detail="Mistral API client not configured")   
        
    weaviate_status = "connected" if weaviate_client and weaviate_client.is_ready() else "disconnected"
    
    return {
        "api": "running",
        "weaviate": weaviate_status,
        "mistral_api": "configured" if mistral_client else "not configured"
    }

@router.post("/admin/flush-logs")
async def flush_logs(api_key: str = Depends(get_api_key)):
    """
    Manually flush all log buffers to disk.
    
    Args:
        api_key: API key for authentication
    
    Returns:
        dict: Status message
    """
    try:
        # Create a temporary chat logger instance to access the chat log files
        chat_logger = ChatLogger()
        
        if chat_logger and chat_logger.enabled:
            # Flush regular chat logs
            if hasattr(chat_logger, "_flush_buffer"):
                chat_logger._flush_buffer()
                
            # Flush feedback logs if that method exists
            if hasattr(chat_logger, "_flush_feedback_buffer"):
                chat_logger._flush_feedback_buffer()
                
            # Properly close the logger
            if hasattr(chat_logger, "close"):
                chat_logger.close()
                
            return {"status": "success", "message": "All log buffers flushed to disk"}
        else:
            return {"status": "warning", "message": "Chat logging is not enabled"}
    except Exception as e:
        logger.error(f"Error flushing logs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error flushing logs: {str(e)}"
        )
        
@router.get("/captcha")
async def get_captcha():
    """Generate a math CAPTCHA"""
    captcha = generate_math_captcha()
    return {
        "question": captcha["question"],
        "hash": captcha["hash"],
        "timestamp": captcha["timestamp"]
    }

@router.post("/register")
async def register_user(
    request: Request,
    captcha_answer: str = Form(...),
    captcha_hash: str = Form(...),
    captcha_timestamp: str = Form(...)
):
    """Register a new user with CAPTCHA validation"""

    # You could use request.client.host for logging or additional validation
    client_ip = request.client.host

    # Parse the form data to get user information
    form_data = await request.form()
    
    # Extract user data fields
    username = form_data.get("username")
    password = form_data.get("password")
    full_name = form_data.get("full_name")
    email = form_data.get("email")
    
    # Validate required fields
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    # Verify CAPTCHA
    try:
        is_valid = verify_math_captcha(
            user_answer=captcha_answer,
            answer_hash=captcha_hash,
            timestamp=captcha_timestamp
        )
        
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="CAPTCHA verification failed. Please try again."
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CAPTCHA verification error: {str(e)}"
        )
    
    # Continue with registration logic...
    users_db = get_users_db()
    if username in users_db:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Validate password strength
    from utils.utils import validate_password
    is_valid, message = validate_password(password)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=message
        )
    
    # Create the user with non-admin role
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # Create new user entry
    new_user = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False,
        "is_admin": False  # Self-registered users are not admins
    }
    
    # Add to users DB
    users_db[username] = new_user
    
    # Save the updated user DB
    with open('users.json', 'w') as f:
        json.dump(users_db, f, indent=2)
    
    return {"message": "User registered successfully"}


```


# api\middleware\__init__.py
```python

```


# api\middleware\middleware.py
```python
from logging import getLogger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import os
import time
import logging
from collections import defaultdict

from config import settings

logger = getLogger(__name__)

# Storage for rate limiting
ip_request_counters = defaultdict(list)
registration_ip_counters = defaultdict(list)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Only add CSP headers for non-documentation endpoints
        if not request.url.path.startswith("/docs") and not request.url.path.startswith("/redoc"):
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
            
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
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
                logger.warning(f"Invalid API key used in request to {request.url.path}")
                return Response(
                    content='{"detail":"Invalid API key"}',
                    status_code=403,
                    media_type="application/json"
                )
                
            # If we made it here, the key is valid
            return await call_next(request)
        except Exception as e:
            # Log unexpected errors but don't block the request
            logger.error(f"Error in API key validation: {str(e)}")
            return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        now = time.time()
        path = request.url.path
        
        # Apply dedicated limits for registration endpoint
        if path == "/register":
            # Clean old timestamps (older than 15 minutes)
            registration_ip_counters[client_ip] = [
                timestamp for timestamp in registration_ip_counters[client_ip] 
                if now - timestamp < 900  # 15 minutes in seconds
            ]
            
            # Only apply limits if there have been many attempts
            if len(registration_ip_counters[client_ip]) >= 5:
                logger.warning(f"Registration rate limit exceeded for IP: {client_ip}")
                return Response(
                    content='{"detail":"Too many registration attempts. Please try again in a few minutes."}',
                    status_code=429,
                    media_type="application/json"
                )
            
            # Add timestamp for this attempt
            registration_ip_counters[client_ip].append(now)
        else:
            # Regular rate limiting for other endpoints
            # Clean old timestamps (older than 1 minute)
            ip_request_counters[client_ip] = [
                timestamp for timestamp in ip_request_counters[client_ip] 
                if now - timestamp < 60
            ]
            
            # Check if we've hit the limit
            if len(ip_request_counters[client_ip]) >= settings.MAX_REQUESTS_PER_MINUTE:
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json"
                )
                
            # Add current timestamp
            ip_request_counters[client_ip].append(now)
        
        # Process request
        return await call_next(request)

# Export the middleware classes for use in main.py
security_headers_middleware = SecurityHeadersMiddleware
api_key_middleware = APIKeyMiddleware
rate_limit_middleware = RateLimitMiddleware
```


# api\models\__init__.py
```python

```


# api\models\models.py
```python
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

```


# api\utils\__init__.py
```python

```


# api\utils\chat_utils.py
```python
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


```


# api\utils\errors.py
```python
from logging import getLogger
from typing import Dict, Any, Optional, Type
from fastapi import HTTPException

logger = getLogger(__name__)

# Base exception class
class APIError(Exception):
    """Base class for all API errors."""
    status_code: int = 500
    error_type: str = "api_error"
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None, 
        error_type: Optional[str] = None, 
        is_transient: bool = False
    ):
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_type is not None:
            self.error_type = error_type
        self.is_transient = is_transient
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to a dictionary for JSON responses."""
        return {
            "error": self.error_type,
            "detail": self.message,
            "is_transient": self.is_transient
        }

# Specific error classes
class WeaviateError(APIError):
    """Errors related to Weaviate operations."""
    status_code = 503
    error_type = "weaviate_error"

class MistralError(APIError):
    """Errors related to Mistral API operations."""
    status_code = 503
    error_type = "mistral_api_error"

class AuthenticationError(APIError):
    """Authentication-related errors."""
    status_code = 401
    error_type = "authentication_error"

class RateLimitError(APIError):
    """Rate limit exceeded errors."""
    status_code = 429
    error_type = "rate_limit_exceeded"

class ValidationError(APIError):
    """Data validation errors."""
    status_code = 422
    error_type = "validation_error"

# Helper to format error responses for API
def format_error_response(e: Exception, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Format error responses consistently."""
    log_prefix = f"[{request_id}] " if request_id else ""
    
    if isinstance(e, APIError):
        logger.error(f"{log_prefix}{e.error_type}: {str(e)}")
        return {
            "answer": f"I encountered an issue: {str(e)}",
            "sources": [],
            "error": e.error_type
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

# Convert exceptions to HTTPExceptions for FastAPI
def http_exception_handler(e: Exception) -> HTTPException:
    """Convert any exception to an appropriate HTTPException."""
    if isinstance(e, APIError):
        return HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    else:
        return HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )
```


# api\utils\logging_config.py
```python
import os
import logging
import sys
from pathlib import Path
from datetime import datetime
import logging.handlers
import uuid
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

def setup_logging():
    """Set up structured logging for the application."""
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_DIR)
    
    # Try to create the directory - but handle permission errors gracefully
    try:
        log_dir.mkdir(exist_ok=True, parents=True)
    except PermissionError:
        print(f"Warning: Cannot create log directory at {log_dir}. Using current directory.")
        log_dir = Path('.')
    except Exception as e:
        print(f"Warning: Error creating log directory: {str(e)}. Using current directory.")
        log_dir = Path('.')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Try to set up file logging - but fall back to console-only if we have problems
    try:
        # File handler with rotation
        log_file = log_dir / f"api_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, just use console
        print(f"Warning: Could not set up file logging: {str(e)}. Using console logging only.")
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Suppress excessive logging from libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("weaviate").setLevel(logging.INFO)
    
    # Return logger for immediate use
    return root_logger

# JSON structured logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured logging of all requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger = logging.getLogger("api.request")
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", ""),
            }
        )
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        # Calculate duration
        duration_ms = round((time.time() - start_time) * 1000)
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )
        
        return response

# Helper function for structured logging
def log_event(logger, message, **extra):
    """Log an event with structured data."""
    logger.info(message, extra=extra)
```


# api\utils\secret_utils.py
```python
import os
import time
from logging import getLogger

logger = getLogger(__name__)

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

```


# api\utils\utils.py
```python
from logging import getLogger
import re

logger = getLogger(__name__)

def validate_password(password):
    """
    Validate password against requirements:
    - At least 8 characters
    - Contains at least one uppercase letter
    - Contains at least one lowercase letter
    - Contains at least one digit
    - Contains at least one of these special characters: @#$%^&+=!
    - Doesn't contain problematic characters like quotes, backticks, or backslashes
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[@#$%^&+=!]', password):
        return False, "Password must contain at least one special character (@#$%^&+=!)"
    
    # Check for problematic characters
    if re.search(r'[\'"`\\]', password):
        return False, "Password contains invalid characters (quotes, backticks, or backslashes are not allowed)"
    
    # If all checks pass
    return True, "Password is valid"

def validate_user_input_content(v: str) -> str:
    """
    Validate that user input does not contain malicious patterns.
    
    Args:
        v: The  string to validate
        
    Returns:
        str: The validated string
        
    Raises:
        ValueError: If the string contains dangerous patterns
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


```


# api\Dockerfile
```text
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire codebase
COPY . .

# Create an empty __init__.py if it doesn't exist
RUN touch __init__.py

# Create necessary directories and set permissions
RUN mkdir -p logs chat_data && \
    chmod -R 777 logs && \
    chmod -R 777 chat_data

# Run directly with Python instead of using module imports
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


# api\__init__.py
```python

```


# api\config.py
```python
import os
import secrets
from typing import Optional, List
from pydantic_settings import BaseSettings

from logging import getLogger

logger = getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API settings
    API_TITLE: str = "EU-Compliant RAG API"
    API_DESCRIPTION: str = "An EU-compliant RAG implementation using Weaviate and Mistral AI"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Directories and paths
    DOCS_DIR: str = "data"
    LOG_DIR: str = "logs"
    CHAT_LOG_DIR: str = "chat_data"
    
    # Security settings
    INTERNAL_API_KEY_FILE: str = "/run/secrets/internal_api_key"
    INTERNAL_API_KEY: Optional[str] = None
    JWT_SECRET_KEY_FILE: str = "/run/secrets/jwt_secret_key"
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    CORS_ALLOW_CREDENTIALS: bool = True
    
    # Rate limiting
    MAX_REQUESTS_PER_MINUTE: int = 10
    REGISTRATION_MAX_REQUESTS_PER_HOUR: int = 10
    REGISTRATION_MAX_REQUESTS_PER_IP: int = 30
    
    # Weaviate connection
    WEAVIATE_URL: str = "http://weaviate:8080"
    WEAVIATE_TIMEOUT_SECONDS: int = 30
    
    # Mistral settings
    MISTRAL_API_KEY_FILE: str = "/run/secrets/mistral_api_key"
    MISTRAL_API_KEY: Optional[str] = None
    MISTRAL_MODEL: str = "mistral-large-latest"
    DAILY_TOKEN_BUDGET: int = 10000
    
    # Caching
    MAX_CACHE_ENTRIES: int = 100
    CACHE_EXPIRY_SECONDS: int = 3600
    
    # Chat settings
    MAX_QUERY_LENGTH: int = 1000
    MIN_QUERY_LENGTH: int = 3
    
    # Chat logging settings
    ENABLE_CHAT_LOGGING: bool = True
    ANONYMIZE_CHAT_LOGS: bool = True
    LOG_RETENTION_DAYS: int = 30
    CHAT_LOG_BUFFER_SIZE: int = 2
    ANONYMIZE_PREFIX: str = "anon_"    
    
    # Meta settings
    SECRET_ROTATION_WARNING_DAYS: int = 80
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def load_secrets(self):
        """Load secrets from files if available."""
        # Load Mistral API key
        if os.path.exists(self.MISTRAL_API_KEY_FILE):
            try:
                with open(self.MISTRAL_API_KEY_FILE, "r") as f:
                    self.MISTRAL_API_KEY = f.read().strip()
                logger.info(f"Mistral key loaded from file: {self.MISTRAL_API_KEY_FILE}")
            except Exception as e:
                logger.error(f"Error reading Mistral API key from file: {str(e)}")

        # Load Internal API key
        if os.path.exists(self.INTERNAL_API_KEY_FILE):
            try:
                with open(self.INTERNAL_API_KEY_FILE, "r") as f:
                    self.INTERNAL_API_KEY = f.read().strip()
                logger.info(f"Internal API key loaded from file: {self.INTERNAL_API_KEY_FILE}")
            except Exception as e:
                logger.error(f"Error reading internal API key from file: {str(e)}")        

        # Load JWT secret key
        if os.path.exists(self.JWT_SECRET_KEY_FILE):
            try:
                with open(self.JWT_SECRET_KEY_FILE, "r") as f:
                    self.JWT_SECRET_KEY = f.read().strip()
                logger.info(f"JWT secret key loaded from file: {self.JWT_SECRET_KEY_FILE}")
            except Exception as e:
                logger.error(f"Error reading JWT secret key from file: {str(e)}")

        # Generate secrets if not available
        if not self.MISTRAL_API_KEY:
            logger.warning("Mistral API key not found, generating a random one (will change on restart)")
            self.MISTRAL_API_KEY = secrets.token_hex(32)

        if not self.INTERNAL_API_KEY:
            logger.warning("Internal API key not found, generating a random one (will change on restart)")
            self.INTERNAL_API_KEY = secrets.token_hex(32)

        if not self.JWT_SECRET_KEY:
            logger.warning("JWT secret key not found, generating a random one (will change on restart)")
            self.JWT_SECRET_KEY = secrets.token_hex(32)

# Create settings instance
settings = Settings()
settings.load_secrets()
```


# api\main.py
```python
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
    privacy_endpoints,
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
app.include_router(privacy_endpoints.router, prefix=v1_prefix)
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
```


# api\privacy_notice.html
```text
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat Logging Privacy Notice</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        ul {
            margin-bottom: 20px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            font-size: 0.9em;
            color: #777;
        }
        .highlight {
            background-color: #f8f9fa;
            border-left: 4px solid #4caf50;
            padding: 10px 15px;
            margin: 20px 0;
        }
        @media print {
            body {
                color: #000;
                font-size: 12pt;
            }
            h1, h2 {
                color: #000;
            }
        }
    </style>
</head>
<body>
    <h1>Chat Logging Privacy Notice</h1>
    
    <div class="highlight">
        <p><strong>Summary:</strong> When enabled, this system logs anonymized chat interactions for research purposes. Logs are automatically deleted after 30 days, and you can request earlier deletion of your data at any time.</p>
    </div>
    
    <h2>Introduction</h2>
    <p>The Document Chat System may collect and process interaction data for research and service improvement purposes. This privacy notice explains how we handle this data in compliance with GDPR and other applicable privacy regulations.</p>
    
    <h2>Data Collection</h2>
    <p>When enabled, the system logs the following information:</p>
    <ul>
        <li><strong>Date and time</strong> of interactions</li>
        <li><strong>Questions asked</strong> to the system</li>
        <li><strong>Responses provided</strong> by the system</li>
        <li><strong>Document references</strong> used to generate answers</li>
        <li><strong>Anonymized session identifiers</strong> to link related interactions</li>
        <li><strong>System performance metrics</strong></li>
    </ul>

    <h2>Feedback Collection</h2>
    <p>When enabled, the system also collects feedback on responses:</p>
    <ul>
        <li><strong>Rating information</strong> (whether a response was helpful)</li>
        <li><strong>Detailed feedback</strong> you choose to provide</li>
        <li><strong>Categories of issues</strong> you identify with responses</li>
        <li><strong>Anonymized session identifiers</strong> to link feedback to interactions</li>
    </ul>

    <p>This feedback data is processed and stored with the same privacy protections, anonymization, and retention policies as other interaction data.</p>

    
    <h2>Legal Basis for Processing</h2>
    <p>We process this data based on:</p>
    <ul>
        <li>Our legitimate interest in improving the system's accuracy and performance</li>
        <li>Your consent, which you may withdraw at any time</li>
        <li>Research purposes (statistical and scientific analysis)</li>
    </ul>
    
    <h2>How We Use This Data</h2>
    <p>The collected data is used to:</p>
    <ul>
        <li>Analyze common question patterns</li>
        <li>Improve answer accuracy and relevance</li>
        <li>Identify missing information in our document collection</li>
        <li>Fix technical issues and optimize system performance</li>
        <li>Conduct research on information retrieval and question-answering systems</li>
        <li>Improve the quality and accuracy of responses based on your feedback</li>
        <li>Address common issues identified through feedback analysis</li>
    </ul>
    
    <h2>Data Retention</h2>
    <ul>
        <li>Chat logs are retained for a maximum of 30 days by default</li>
        <li>After this period, logs are automatically deleted</li>
        <li>You may request earlier deletion of your data at any time</li>
    </ul>
    
    <h2>Data Security and Access</h2>
    <ul>
        <li>All logs are stored securely with access restricted to authorized administrators</li>
        <li>Logs are stored locally on the system's secured servers</li>
        <li>We implement appropriate technical and organizational measures to protect your data</li>
    </ul>
    
    <h2>Anonymization</h2>
    <p>By default, when logging is enabled:</p>
    <ul>
        <li>User identifiers are anonymized through cryptographic hashing</li>
        <li>IP addresses are not stored or are anonymized</li>
        <li>Personal information in queries may still be recorded; avoid including sensitive data in your questions</li>
    </ul>
    
    <div class="highlight">
        <p><strong>Important:</strong> The system may record the content of your questions. Please do not include personal data, sensitive information, or confidential details in your queries.</p>
    </div>
    
    <h2>Your Rights</h2>
    <p>Under GDPR and similar regulations, you have the right to:</p>
    <ul>
        <li><strong>Access</strong> the data we hold about you</li>
        <li>Request <strong>correction</strong> of inaccurate data</li>
        <li>Request <strong>deletion</strong> of your data</li>
        <li><strong>Object to</strong> or <strong>restrict</strong> the processing of your data</li>
        <li><strong>Withdraw consent</strong> for data collection at any time</li>
    </ul>
    
    <h2>How to Exercise Your Rights</h2>
    <p>To exercise any of these rights, please contact the system administrator. We will respond to your request within 30 days.</p>
    
    <h2>Changes to This Privacy Notice</h2>
    <p>We may update this privacy notice from time to time. Any changes will be posted with a revised effective date.</p>
    
    <div class="footer">
        <p>Effective Date: March 9, 2025</p>
        <p>Document version: 1.0</p>
    </div>
</body>
</html>
```


# api\requirements.txt
```cmake
fastapi==0.115.11
uvicorn==0.34.0
python-dotenv==1.0.1
weaviate-client==4.11.1
mistralai==1.5.0
tenacity==8.0.1
aiofiles==23.2.1
pydantic==2.10.6
pydantic-settings==2.8.1
pyjwt==2.8.0
bcrypt==4.0.1
python-multipart==0.0.9
```


# processor\Dockerfile
```text
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY processor.py .

CMD ["python", "processor.py"]
```


# processor\processor.py
```python
import os
import re
import json
import time
import logging
import uuid
import glob
import asyncio
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Set, 
    TypedDict, Callable, Iterable, Generator
)
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, DataType
from weaviate.classes.query import Filter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions
class ChunkMetadata(TypedDict, total=False):
    """Type definition for chunk metadata."""
    page: Optional[int]
    heading: Optional[str]
    level: Optional[int]
    itemType: Optional[str]
    title: Optional[str]
    date: Optional[str]
    creators: Optional[List[Dict[str, str]]]

class TextChunk(TypedDict):
    """Type definition for a text chunk."""
    content: str
    page: Optional[int]
    heading: Optional[str]
    level: Optional[int]

class ProcessingResult(TypedDict):
    """Type definition for processing result."""
    success: bool
    message: str
    chunks_processed: int
    file_path: str
    metadata: Optional[Dict[str, Any]]

class ProcessingStats(TypedDict):
    """Type definition for processing statistics."""
    total: int
    processed: int
    skipped: int
    failed: int
    start_time: float
    duration: Optional[float]

class ChunkingStrategy(str, Enum):
    """Enum for different chunking strategies."""
    SIMPLE = "simple"           # Simple character-based chunking
    PARAGRAPH = "paragraph"     # Paragraph-based chunking
    SECTION = "section"         # Section-based (using headings)
    SEMANTIC = "semantic"       # Semantic chunking using AI

# Configuration class
class ProcessorConfig:
    """
    Configuration for document processor.
    
    Attributes:
        WEAVIATE_URL (str): URL for the Weaviate instance
        DATA_FOLDER (str): Folder to watch for documents
        CHUNK_SIZE (int): Default size of text chunks
        CHUNK_OVERLAP (int): Default overlap between chunks
        CHUNKING_STRATEGY (ChunkingStrategy): Strategy for chunking text
        MAX_RETRIES (int): Maximum number of retries for database operations
        RETRY_DELAY (int): Delay between retries in seconds
        FILE_EXTENSIONS (List[str]): Supported file extensions
        MAX_WORKER_THREADS (int): Maximum number of worker threads
        BATCH_SIZE (int): Number of chunks to process in a batch
    """
    
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    DATA_FOLDER = os.getenv("DATA_FOLDER", "/data")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    CHUNKING_STRATEGY = ChunkingStrategy(os.getenv("CHUNKING_STRATEGY", "section"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "10"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
    FILE_EXTENSIONS = [".md", ".txt"]  # Supported file extensions
    MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", "5"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

config = ProcessorConfig()

# ------------ Utility Functions ------------
def chunk_text(
    text: str,
    max_chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    chunking_strategy: ChunkingStrategy = config.CHUNKING_STRATEGY
) -> List[TextChunk]:
    """
    Split text into overlapping chunks, respecting Markdown structure and sentence boundaries.
    
    This function processes Markdown documents using the following conventions:
    
    1. Page numbering:
       Use HTML comments to mark page numbers: <!-- page: 123 -->
    
    2. Heading structure:
       Standard Markdown heading syntax determines section hierarchy:
       # Heading 1
       ## Heading 2
       ### Heading 3
    
    3. Paragraphs:
       Separate paragraphs with blank lines
    
    The function preserves document structure by:
    - Keeping heading context with content
    - Respecting page boundaries
    - Maintaining heading hierarchy levels
    - Preserving paragraph and sentence boundaries when possible
    
    Args:
        text: The Markdown text to chunk
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        chunking_strategy: Strategy to use for chunking
        
    Returns:
        List[TextChunk]: List of chunks with metadata
    """
    # Input validation
    if not text:
        logger.warning("Empty text provided to chunk_text")
        return []
    
    if max_chunk_size <= 0:
        logger.warning(f"Invalid max_chunk_size: {max_chunk_size}, using default")
        max_chunk_size = config.CHUNK_SIZE
        
    if overlap < 0 or overlap >= max_chunk_size:
        logger.warning(f"Invalid overlap: {overlap}, using default")
        overlap = config.CHUNK_OVERLAP
    
    # Helper function for language-agnostic sentence splitting
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences, handling multiple languages."""
        # This pattern works for many European languages
        # It looks for periods, question marks, or exclamation points
        # followed by spaces and capital letters
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return sentences
    
    # Extract page markers
    page_pattern = re.compile(r'<!--\s*page:\s*(\d+)\s*-->')
    page_matches = list(page_pattern.finditer(text))
    
    # First, process the text into pages
    pages = []
    current_page_num = 1
    last_pos = 0
    
    # Process page markers
    for match in page_matches:
        # Add content before this page marker with current page number
        page_text = text[last_pos:match.start()]
        if page_text.strip():
            pages.append((page_text, current_page_num))
        
        # Update page number and position
        current_page_num = int(match.group(1))
        last_pos = match.end()
    
    # Add any remaining content
    if last_pos < len(text):
        pages.append((text[last_pos:], current_page_num))
    
    # Now process each page for headings and sections
    sections = []
    heading_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
    
    for page_text, page_num in pages:
        # Find all headings in this page
        heading_matches = list(heading_pattern.finditer(page_text))
        
        if not heading_matches:
            # No headings on this page, treat whole page as one section
            sections.append({
                "text": page_text,
                "page": page_num,
                "heading": "Untitled Section",
                "level": 0
            })
            continue
        
        # Process sections based on headings
        last_heading_pos = 0
        current_heading = "Untitled Section"
        current_level = 0
        
        for i, match in enumerate(heading_matches):
            # Add content before this heading (if not at start)
            if i > 0 or match.start() > 0:
                section_text = page_text[last_heading_pos:match.start()]
                if section_text.strip():
                    sections.append({
                        "text": section_text,
                        "page": page_num,
                        "heading": current_heading,
                        "level": current_level
                    })
            
            # Update current heading and position
            heading_marks = match.group(1)  # The # characters
            current_level = len(heading_marks)  # Number of # determines level
            current_heading = match.group(2).strip()  # The heading text
            last_heading_pos = match.start()
        
        # Add the final section in this page
        final_section = page_text[last_heading_pos:]
        if final_section.strip():
            sections.append({
                "text": final_section,
                "page": page_num,
                "heading": current_heading,
                "level": current_level
            })
    
    # Now chunk each section with sentence boundary detection
    chunks: List[TextChunk] = []
    
    # Use the appropriate chunking strategy
    if chunking_strategy == ChunkingStrategy.SIMPLE:
        # Simple character-based chunking without respecting structure
        for section in sections:
            section_text = section["text"]
            for i in range(0, len(section_text), max_chunk_size - overlap):
                chunk_text = section_text[i:i + max_chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "page": section["page"],
                        "heading": section["heading"],
                        "level": section["level"]
                    })
    elif chunking_strategy == ChunkingStrategy.PARAGRAPH:
        # Paragraph-aware chunking
        for section in sections:
            section_text = section["text"]
            paragraphs = section_text.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Store current chunk if not empty
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk,
                            "page": section["page"],
                            "heading": section["heading"],
                            "level": section["level"]
                        })
                    
                    # Start a new chunk with overlap if the paragraph is too large
                    if len(paragraph) > max_chunk_size:
                        # Recursively chunk large paragraphs
                        for i in range(0, len(paragraph), max_chunk_size - overlap):
                            sub_chunk = paragraph[i:i + max_chunk_size]
                            if sub_chunk.strip():
                                chunks.append({
                                    "content": sub_chunk,
                                    "page": section["page"],
                                    "heading": section["heading"],
                                    "level": section["level"]
                                })
                        current_chunk = ""
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "page": section["page"],
                    "heading": section["heading"],
                    "level": section["level"]
                })
    else:
        # Default to section-based chunking with sentence awareness (SECTION strategy)
        for section in sections:
            section_text = section["text"]
            section_heading = section["heading"]
            section_page = section["page"]
            section_level = section["level"]
            
            # Skip heading line itself when chunking
            content_start = section_text.find('\n')
            if content_start > 0:
                content = section_text[content_start:].strip()
            else:
                content = section_text.strip()
            
            if not content:
                continue  # Skip empty sections
            
            # For very small sections, keep them as a single chunk
            if len(content) <= max_chunk_size:
                chunks.append({
                    "content": content,
                    "page": section_page,
                    "heading": section_heading,
                    "level": section_level
                })
                continue
            
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            current_chunk = ""
            current_sentences = []
            
            for paragraph in paragraphs:
                # Split paragraph into sentences using our language-agnostic approach
                sentences = split_into_sentences(paragraph)
                
                # Process each sentence
                for sentence in sentences:
                    # If adding this sentence would exceed max size and we already have content
                    if len(current_chunk) + len(sentence) + 2 > max_chunk_size and current_chunk:  # +2 for the newline
                        chunks.append({
                            "content": current_chunk.strip(),
                            "page": section_page,
                            "heading": section_heading,
                            "level": section_level
                        })
                        
                        # For overlap, include sentences from the previous chunk
                        overlap_text = ""
                        overlap_size = 0
                        
                        # Work backwards through sentences to create overlap
                        for prev_sentence in reversed(current_sentences):
                            if overlap_size + len(prev_sentence) + 1 <= overlap:  # +1 for space
                                overlap_text = prev_sentence + " " + overlap_text
                                overlap_size += len(prev_sentence) + 1
                            else:
                                break
                        
                        # Start a new chunk with the overlap plus current sentence
                        current_chunk = overlap_text + sentence
                        current_sentences = [sentence]
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_sentences.append(sentence)
                
                # Add paragraph separator if this isn't the last paragraph
                if current_chunk and paragraph != paragraphs[-1]:
                    current_chunk += "\n\n"
                    current_sentences.append("\n\n")
            
            # Add the last chunk from this section
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "page": section_page,
                    "heading": section_heading,
                    "level": section_level
                })
    
    return chunks

def detect_file_encoding(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a readable text file and return the correct encoding.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple[bool, str]: (is_valid, encoding or error_message)
    """
    # Expanded list of encodings to try
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check if it's actually a file
    if not os.path.isfile(file_path):
        return False, "Path exists but is not a file"
    
    # Check for zero-length files
    try:
        if os.path.getsize(file_path) == 0:
            return False, "File is empty"
    except OSError as e:
        return False, f"Error checking file size: {str(e)}"
    
    # Try to read with each encoding
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                # Try to read a sample to verify encoding works
                sample = file.read(100)
                return True, encoding
        except UnicodeDecodeError:
            continue
        except PermissionError as e:
            return False, f"Permission error reading file: {str(e)}"
        except OSError as e:
            return False, f"Error reading file: {str(e)}"
    
    return False, "Could not decode with any supported encoding"

async def connect_with_retry(
    weaviate_url: str,
    max_retries: int = config.MAX_RETRIES,
    retry_delay: int = config.RETRY_DELAY
) -> weaviate.Client:
    """
    Connect to Weaviate with retry mechanism.
    
    Args:
        weaviate_url: URL of the Weaviate instance
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        weaviate.Client: Connected Weaviate client
        
    Raises:
        ConnectionError: If connection fails after max retries
    """
    # Parse the URL to get components
    use_https = weaviate_url.startswith("https://")
    host_part = weaviate_url.replace("http://", "").replace("https://", "")
    
    # Handle port if specified
    if ":" in host_part:
        host, port = host_part.split(":")
        port = int(port)
    else:
        host = host_part
        port = 443 if use_https else 80
    
    retries = 0
    last_exception = None
    
    while retries < max_retries:
        try:
            logger.info(f"Connecting to Weaviate (attempt {retries+1}/{max_retries})...")
            
            # Connect to Weaviate
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=use_https,
                grpc_host=host,
                grpc_port=50051, # Default gRPC port
                grpc_secure=use_https,
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=60, query=60, insert=60)
                )
            )
            
            # Verify connection
            if client.is_ready():
                logger.info("Successfully connected to Weaviate")
                return client
            else:
                logger.warning("Weaviate client not ready yet")
                raise ConnectionError("Weaviate client not ready")
        except Exception as e:
            last_exception = e
            logger.warning(f"Connection attempt {retries+1} failed: {str(e)}")
        
        # Wait before retry
        logger.info(f"Waiting {retry_delay} seconds before retry...")
        await asyncio.sleep(retry_delay)
        retries += 1
    
    # If we get here, all retries failed
    error_message = f"Failed to connect to Weaviate after {max_retries} attempts. Last error: {str(last_exception)}"
    logger.error(error_message)
    raise ConnectionError(error_message)

# ------------ Document Storage Class ------------

class DocumentStorage:
    """
    Storage class for managing document chunks in Weaviate.
    
    This class handles:
    - Setting up the Weaviate schema
    - Deleting existing chunks
    - Storing new chunks with metadata
    
    Attributes:
        client: Weaviate client connection
    """
    
    def __init__(self, weaviate_client: weaviate.Client):
        """
        Initialize storage with a Weaviate client connection.
        
        Args:
            weaviate_client: Connected Weaviate client instance
        """
        self.client = weaviate_client
        
    async def setup_schema(self) -> bool:
        """
        Set up the Weaviate schema for document chunks.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        start_time = time.time()
        logger.info("Setting up Weaviate schema for document chunks")
        
        try:
            # Check if the collection already exists
            if not self.client.collections.exists("DocumentChunk"):
                logger.info("DocumentChunk collection does not exist, creating new collection")
                creation_start = time.time()
                # Collection doesn't exist, create it            
                self.client.collections.create(
                    name="DocumentChunk",
                    vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                    properties=[
                        weaviate.classes.config.Property(
                            name="content",
                            data_type=DataType.TEXT
                        ),
                        weaviate.classes.config.Property(
                            name="filename", 
                            data_type=DataType.TEXT
                        ),
                        weaviate.classes.config.Property(
                            name="chunkId", 
                            data_type=DataType.INT
                        ),
                        weaviate.classes.config.Property(
                            name="metadataJson", 
                            data_type=DataType.TEXT
                        )
                    ]
                )
                creation_time = time.time() - creation_start
                logger.info(f"DocumentChunk collection created in Weaviate (took {creation_time:.2f}s)")
            else:
                logger.info("DocumentChunk collection already exists in Weaviate")
                
            setup_time = time.time() - start_time
            logger.info(f"Schema setup complete in {setup_time:.2f}s")
            return True
        except Exception as e:
            logger.error(f"Error setting up Weaviate schema: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def delete_chunks(self, filename: str) -> int:
        """
        Delete all chunks associated with a specific filename.
        
        Args:
            filename: The filename to delete chunks for
            
        Returns:
            int: Number of chunks deleted
        """
        start_time = time.time()
        logger.info(f"Deleting chunks for file: {filename}")
        
        try:
            # Get the collection
            collection = self.client.collections.get("DocumentChunk")
            
            # Create a proper filter for Weaviate
            where_filter = Filter.by_property("filename").equal(filename)
            
            # Delete using the filter
            deletion_start = time.time()
            result = collection.data.delete_many(
                where=where_filter
            )
            deletion_time = time.time() - deletion_start
            
            # Log the result
            deleted_count = 0
            if hasattr(result, 'successful'):
                deleted_count = result.successful
                logger.info(f"Deleted {deleted_count} existing chunks for {filename} in {deletion_time:.2f}s")
            else:
                logger.info(f"No existing chunks found for {filename} ({deletion_time:.2f}s)")
                
            total_time = time.time() - start_time
            logger.debug(f"Total chunk deletion process took {total_time:.2f}s")
            
            return deleted_count
                
        except Exception as e:
            logger.error(f"Error deleting existing chunks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    async def store_chunk(
        self, 
        content: str, 
        filename: str, 
        chunk_id: int, 
        metadata: Optional[Dict[str, Any]] = None, 
        page: Optional[int] = None, 
        heading: Optional[str] = None, 
        level: Optional[int] = None
    ) -> bool:
        """
        Store a document chunk in Weaviate with metadata as a JSON string.
        
        Args:
            content: The text content of the chunk
            filename: Source document name
            chunk_id: Sequential ID of the chunk within the document
            metadata: Document metadata from the .metadata.json file
            page: Page number where this chunk appears
            heading: Section heading text for this chunk
            level: Heading level (1 for #, 2 for ##, etc.)
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        start_time = time.time()
        chunk_size = len(content) if isinstance(content, str) else 0
        logger.debug(f"Storing chunk {chunk_id} from {filename} (size: {chunk_size} chars)")
        
        try:
            # Input validation
            if not content or not content.strip():
                logger.warning(f"Empty content for chunk {chunk_id} from {filename}")
                return False
                
            if not filename:
                logger.warning(f"Missing filename for chunk {chunk_id}")
                return False
                
            if chunk_id < 0:
                logger.warning(f"Invalid chunk_id: {chunk_id}")
                return False
            
            properties = {
                "content": content,
                "filename": filename,
                "chunkId": chunk_id
            }

            # Add page number and heading if available
            chunk_metadata = {}
            
            if page is not None:
                chunk_metadata["page"] = page
                
            if heading is not None:
                chunk_metadata["heading"] = heading

            # Add heading level if available
            if level is not None:
                chunk_metadata["headingLevel"] = level            

            # Merge with existing metadata if provided
            if metadata and isinstance(metadata, dict):
                chunk_metadata.update(metadata)

            # Add metadata as a JSON string if we have any
            if chunk_metadata:
                properties["metadataJson"] = json.dumps(chunk_metadata)    
                logger.debug(f"Added metadata to chunk {chunk_id} from {filename}")            

            # Create a UUID based on filename and chunk_id for consistency
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{chunk_id}"))

            # Get the DocumentChunk collection
            collection = self.client.collections.get("DocumentChunk")
            
            # First, try to delete the object if it exists
            try:
                collection.data.delete_by_id(obj_uuid)
                logger.debug(f"Deleted existing object with ID {obj_uuid}")
            except Exception as delete_error:
                # It's okay if the object doesn't exist yet
                logger.debug(f"Object with ID {obj_uuid} not found for deletion (expected for new chunks)")
            
            # Now insert the object
            insert_start = time.time()
            collection.data.insert(
                properties=properties,
                uuid=obj_uuid
            )
            insert_time = time.time() - insert_start
            
            total_time = time.time() - start_time
            logger.debug(f"Stored chunk {chunk_id} from {filename} (size: {chunk_size} chars) in {total_time:.3f}s (insert: {insert_time:.3f}s)")
            return True
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id} from {filename}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
                
    async def store_chunks_batch(
        self, 
        chunks: List[Dict[str, Any]], 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int]:
        """
        Store multiple chunks in a batch operation.
        
        Args:
            chunks: List of chunk data dictionaries
            filename: Source document name
            metadata: Optional document metadata
            
        Returns:
            Tuple[int, int]: (successful_count, failed_count)
        """
        if not chunks:
            logger.warning(f"No chunks provided for {filename}")
            return 0, 0
            
        success_count = 0
        fail_count = 0
        
        # Process in batches to avoid overwhelming the database
        batch_size = config.BATCH_SIZE
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for idx, chunk in enumerate(batch):
                chunk_id = i + idx
                success = await self.store_chunk(
                    content=chunk["content"],
                    filename=filename,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    page=chunk.get("page"),
                    heading=chunk.get("heading"),
                    level=chunk.get("level")
                )
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    
            # Log progress for large batches
            if len(chunks) > batch_size and i % (batch_size * 5) == 0 and i > 0:
                logger.info(f"Stored {i}/{len(chunks)} chunks from {filename}")
        
        logger.info(f"Batch storage complete for {filename}: {success_count} succeeded, {fail_count} failed")
        return success_count, fail_count

    async def get_chunks(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of chunk objects
        """
        try:
            collection = self.client.collections.get("DocumentChunk")
            results = collection.query.near_text(
                query=query,
                limit=limit,
                return_properties=["content", "filename", "chunkId", "metadataJson"]
            )
            
            return [obj.properties for obj in results.objects]
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    async def get_document_count(self) -> int:
        """
        Get the count of unique documents in the database.
        
        Returns:
            int: Number of unique documents
        """
        try:
            collection = self.client.collections.get("DocumentChunk")
            
            query_result = collection.query.fetch_objects(
                return_properties=["filename"],
                limit=10000  # Practical limit for most cases
            )
            
            unique_filenames = set()
            for obj in query_result.objects:
                unique_filenames.add(obj.properties["filename"])
                
            return len(unique_filenames)
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return -1

# ------------ Processing Tracker Class ------------
class ProcessingTracker:
    """
    Tracks files that have been processed to avoid redundant processing.
    
    Attributes:
        tracker_file_path: Path to the JSON file storing processing records
        processed_files: Dictionary of processed files with metadata
    """
    
    def __init__(self, tracker_file_path: str = ".processed_files.json"):
        """
        Initialize a tracker that keeps record of processed files.
        
        Args:
            tracker_file_path: Path to the JSON file storing processing records
        """
        logger.info(f"Initializing file processing tracker at {tracker_file_path}")
        self.tracker_file_path = tracker_file_path
        self.processed_files = self._load_tracker()
        logger.info(f"Tracker initialized with {len(self.processed_files)} previously processed files")

    def _load_tracker(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the tracker file or create it if it doesn't exist.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of processed files with metadata
        """
        if os.path.exists(self.tracker_file_path):
            try:
                logger.info(f"Loading existing tracker file from {self.tracker_file_path}")
                with open(self.tracker_file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded tracker with {len(data)} records")
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding tracker file JSON: {str(e)}")
                return {}
            except PermissionError as e:
                logger.error(f"Permission error reading tracker file: {str(e)}")
                return {}
            except Exception as e:
                logger.error(f"Error loading tracker file: {str(e)}")
                return {}
        logger.info("No existing tracker file found, starting with empty tracking")
        return {}

    def _save_tracker(self) -> bool:
        """
        Save the tracker data to file.
        
        Returns:
            bool: Whether the save was successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracker_file_path) or '.', exist_ok=True)
            
            with open(self.tracker_file_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
                
            logger.debug(f"Saved processing tracker to {self.tracker_file_path}")
            return True
        except PermissionError as e:
            logger.error(f"Permission error saving tracker file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error saving tracker file: {str(e)}")
            return False
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on modification time.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file is new or modified since last processing
        """
        try:
            file_mod_time = os.path.getmtime(file_path)
            file_key = os.path.basename(file_path)
            
            logger.debug(f"Checking if file needs processing: {file_key}")
            
            # If file not in tracker or has been modified, process it
            if file_key not in self.processed_files:
                logger.info(f"File {file_key} not in tracker, will be processed")
                return True
            
            last_mod_time = self.processed_files[file_key]['last_modified']
            time_diff = file_mod_time - last_mod_time
            
            if time_diff > 0:
                logger.info(f"File {file_key} has been modified since last processing " +
                        f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mod_time))} vs. " +
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_mod_time))})")
                return True
            else:
                logger.debug(f"File {file_key} unchanged since last processing, skipping")
                return False
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking file status: {str(e)}")
            # If in doubt, process the file
            return True

    def mark_as_processed(self, file_path: str) -> bool:
        """
        Mark a file as processed with current timestamps.
        
        Args:
            file_path: Path to the file to mark as processed
            
        Returns:
            bool: Whether marking was successful
        """
        try:
            file_mod_time = os.path.getmtime(file_path)
            file_key = os.path.basename(file_path)
            process_time = time.time()
            
            self.processed_files[file_key] = {
                'path': file_path,
                'last_modified': file_mod_time,
                'last_processed': process_time,
                'last_processed_human': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process_time))
            }
            self._save_tracker()
            logger.info(f"Marked {file_key} as processed (last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mod_time))})")
            return True
        except Exception as e:
            logger.error(f"Error marking file as processed: {str(e)}")
            return False
    
    def remove_file(self, filename: str) -> bool:
        """
        Remove a file from the tracking record.
        
        Args:
            filename: Name of the file to remove from tracking
            
        Returns:
            bool: Whether removal was successful
        """
        try:
            if filename in self.processed_files:
                logger.info(f"Removing {filename} from processing tracker")
                del self.processed_files[filename]
                self._save_tracker()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing file from tracker: {str(e)}")
            return False
    
    def get_all_tracked_files(self) -> List[str]:
        """
        Return a list of all tracked filenames.
        
        Returns:
            List[str]: List of tracked filenames
        """
        return list(self.processed_files.keys())
    
    def clear_tracking_data(self) -> bool:
        """
        Clear all tracking data.
        
        Returns:
            bool: Whether clearing was successful
        """
        try:
            self.processed_files = {}
            self._save_tracker()
            logger.info("Cleared all file tracking data")
            return True
        except Exception as e:
            logger.error(f"Error clearing tracking data: {str(e)}")
            return False
    
    async def get_tracking_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tracked files.
        
        Returns:
            Dict[str, Any]: Statistics about tracked files
        """
        stats = {
            "total_files_tracked": len(self.processed_files),
            "recently_processed": 0,
            "oldest_file": None,
            "newest_file": None,
            "average_file_age_days": 0
        }
        
        if not self.processed_files:
            return stats
            
        now = time.time()
        last_24h = now - (24 * 60 * 60)
        
        # Calculate stats
        file_ages = []
        oldest_time = now
        oldest_file = None
        newest_time = 0
        newest_file = None
        
        for filename, data in self.processed_files.items():
            # Count recently processed files
            if data.get('last_processed', 0) > last_24h:
                stats["recently_processed"] += 1
                
            # Track file ages
            mod_time = data.get('last_modified', 0)
            file_ages.append(now - mod_time)
            
            # Track oldest file
            if mod_time < oldest_time:
                oldest_time = mod_time
                oldest_file = filename
                
            # Track newest file
            if mod_time > newest_time:
                newest_time = mod_time
                newest_file = filename
        
        # Set stats
        if oldest_file:
            stats["oldest_file"] = {
                "name": oldest_file,
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oldest_time))
            }
            
        if newest_file:
            stats["newest_file"] = {
                "name": newest_file,
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest_time))
            }
            
        if file_ages:
            avg_age_seconds = sum(file_ages) / len(file_ages)
            stats["average_file_age_days"] = round(avg_age_seconds / (24 * 60 * 60), 2)
            
        return stats

# ------------ Document Processor Class ------------

class DocumentProcessor:
    """
    Processes text files into chunks and stores them in a vector database.
    
    Attributes:
        storage: DocumentStorage instance for storing document chunks
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Amount of overlap between consecutive chunks
        chunking_strategy: Strategy to use for chunking text
    """
    
    def __init__(
        self, 
        storage: DocumentStorage, 
        chunk_size: int = config.CHUNK_SIZE, 
        chunk_overlap: int = config.CHUNK_OVERLAP,
        chunking_strategy: ChunkingStrategy = config.CHUNKING_STRATEGY
    ):
        """
        Initialize a document processor that reads and chunks text files.
        
        Args:
            storage: DocumentStorage instance for storing document chunks
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Amount of overlap between consecutive chunks
            chunking_strategy: Strategy to use for chunking text
        """
        self.storage = storage
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.metrics = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "processing_time": 0
        }
    
    async def process_file(self, file_path: str) -> ProcessingResult:
        """
        Process a markdown file: read, chunk, and store in vector database.
        
        Handles Markdown files with structured headings and optional page markers.
        See chunk_text() function for details on supported Markdown conventions.
        
        Args:
            file_path: Path to the markdown file to process
                
        Returns:
            ProcessingResult: Result of the processing operation
        """
        start_time = time.time()
        file_size = 0
        chunks_processed = 0
        
        # Default result for failures
        failure_result = ProcessingResult(
            success=False,
            message="Processing failed",
            chunks_processed=0,
            file_path=file_path,
            metadata=None
        )
        
        # Log file size for context
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"Processing file: {file_path} (size: {file_size/1024:.1f} KB)")
        except Exception as e:
            logger.info(f"Processing file: {file_path} (size unknown: {str(e)})")
        
        # Validate file and get encoding
        encoding_start = time.time()
        is_valid, result = detect_file_encoding(file_path)
        encoding_time = time.time() - encoding_start
        
        if not is_valid:
            logger.error(f"Error processing file {file_path}: {result}")
            failure_result["message"] = f"File validation failed: {result}"
            return failure_result
        
        # Result is the encoding if valid
        encoding = result
        logger.info(f"File validated, using encoding: {encoding} (detection took {encoding_time:.2f}s)")
        
        try:
            read_start = time.time()
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            read_time = time.time() - read_start
            logger.info(f"File read complete in {read_time:.2f}s. Content length: {len(content)} characters")
            
            # Get the filename without the path
            filename = os.path.basename(file_path)

            # Check for associated metadata file
            metadata = None
            base_name = os.path.splitext(filename)[0]
            metadata_path = os.path.join(os.path.dirname(file_path), f"{base_name}.metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as meta_file:
                        metadata = json.load(meta_file)
                    logger.info(f"Loaded metadata from {metadata_path}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing metadata JSON from {metadata_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
            
            # Delete existing chunks for this file if any
            deletion_start = time.time()
            await self.storage.delete_chunks(filename)
            deletion_time = time.time() - deletion_start
            logger.info(f"Previous chunks deletion completed in {deletion_time:.2f}s")
            
            # Split the content into chunks
            chunk_start = time.time()
            chunks = chunk_text(
                content, 
                self.chunk_size, 
                self.chunk_overlap,
                self.chunking_strategy
            )
            chunk_time = time.time() - chunk_start
            
            # Calculate average chunk size and log
            avg_chunk_size = sum(len(chunk["content"]) for chunk in chunks) / max(len(chunks), 1)
            logger.info(f"Text chunking complete in {chunk_time:.2f}s. Created {len(chunks)} chunks with avg size of {avg_chunk_size:.1f} chars")
            
            # Store chunks in Weaviate
            storage_start = time.time()
            success_count, fail_count = await self.storage.store_chunks_batch(chunks, filename, metadata)
            storage_time = time.time() - storage_start
            
            # Update metrics
            self.metrics["chunks_created"] += len(chunks)
            self.metrics["chunks_stored"] += success_count
            chunks_processed = success_count
            
            if fail_count > 0:
                logger.warning(f"{fail_count} chunks failed to store from {filename}")
            
            total_time = time.time() - start_time
            self.metrics["processing_time"] += total_time
            self.metrics["files_processed"] += 1
            
            logger.info(f"File {filename} processed successfully in total time: {total_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                message=f"Processing successful, stored {success_count} chunks",
                chunks_processed=chunks_processed,
                file_path=file_path,
                metadata=metadata
            )
                
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error processing file {file_path}: {str(e)}")
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Unicode decode error: {str(e)}"
            return failure_result
        except PermissionError as e:
            logger.error(f"Permission error processing file {file_path}: {str(e)}")
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Permission error: {str(e)}"
            return failure_result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Processing error: {str(e)}"
            return failure_result
                    
    async def process_directory(
        self, 
        directory_path: str, 
        tracker: Optional[ProcessingTracker] = None,
        file_extensions: Optional[List[str]] = None
    ) -> ProcessingStats:
        """
        Process all markdown files in a directory.
        
        Args:
            directory_path: Path to the directory containing text files
            tracker: Optional ProcessingTracker to track processed files
            file_extensions: Optional list of file extensions to process
            
        Returns:
            ProcessingStats: Statistics about the processing
        """
        start_time = time.time()
        logger.info(f"Scanning for files in {directory_path}")
        
        # Use default extensions if none provided
        if file_extensions is None:
            file_extensions = config.FILE_EXTENSIONS
            
        # Find all files with the specified extensions
        all_files = []
        for ext in file_extensions:
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            
        logger.info(f"Found {len(all_files)} files with extensions: {', '.join(file_extensions)}")
        
        stats: ProcessingStats = {
            "total": len(all_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "start_time": start_time,
            "duration": None
        }
        
        # Filter files if tracker is provided
        files_to_process = []
        for file_path in all_files:
            if tracker and not tracker.should_process_file(file_path):
                logger.info(f"Skipping already processed file: {file_path}")
                stats["skipped"] += 1
            else:
                files_to_process.append(file_path)
        
        # Process files concurrently
        if files_to_process:
            # Create tasks for each file
            tasks = [self.process_file(file_path) for file_path in files_to_process]
            
            # Process in batches to avoid overwhelming resources
            batch_size = config.BATCH_SIZE
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                # Process the batch of files concurrently
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    file_path = files_to_process[i + j]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Exception processing {file_path}: {str(result)}")
                        stats["failed"] += 1
                    elif result["success"]:
                        stats["processed"] += 1
                        if tracker:
                            tracker.mark_as_processed(file_path)
                    else:
                        stats["failed"] += 1
                        logger.error(f"Failed to process {file_path}: {result['message']}")
                
                # Log progress for large batches
                if len(files_to_process) > batch_size and i > 0:
                    logger.info(f"Processed {i + len(batch)}/{len(files_to_process)} files")
        
        process_time = time.time() - start_time
        stats["duration"] = process_time
        logger.info(f"Directory processing complete in {process_time:.2f}s. Stats: {stats}")
        return stats

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current processing metrics.
        
        Returns:
            Dict[str, Any]: Current processing metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters to zero."""
        self.metrics = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "processing_time": 0
        }

    async def process_all_documents(self, tracker, data_folder):
        """
        Process all documents in the data folder.
        Compare current files to tracked files and handle additions, modifications, and deletions.
        
        Args:
            tracker: ProcessingTracker instance
            data_folder: Path to the folder containing documents
            
        Returns:
            dict: Statistics about processing
        """
        stats = {
            "total_files": 0,
            "new_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "unchanged_files": 0,
            "processed_success": 0,
            "processed_failed": 0
        }
        
        # Get all current files in the data folder
        current_files = set()
        for ext in config.FILE_EXTENSIONS:
            pattern = os.path.join(data_folder, f"*{ext}")
            current_files.update(os.path.abspath(f) for f in glob.glob(pattern))
        
        stats["total_files"] = len(current_files)
        logger.info(f"Found {len(current_files)} files in {data_folder}")
        
        # Get previously tracked files
        tracked_files = set()
        for filename in tracker.get_all_tracked_files():
            full_path = os.path.join(data_folder, filename)
            if os.path.isabs(full_path):
                tracked_files.add(full_path)
            else:
                tracked_files.add(os.path.abspath(full_path))
        
        logger.info(f"Found {len(tracked_files)} files in tracking data")
        
        # Process new or modified files
        for file_path in current_files:
            file_name = os.path.basename(file_path)
            
            if file_path not in tracked_files:
                logger.info(f"New file detected: {file_path}")
                stats["new_files"] += 1
            elif tracker.should_process_file(file_path):
                logger.info(f"Modified file detected: {file_path}")
                stats["modified_files"] += 1
            else:
                logger.info(f"Unchanged file: {file_path}")
                stats["unchanged_files"] += 1
                continue
            
            # Process the file
            try:
                result = await self.process_file(file_path)
                
                if result["success"]:
                    stats["processed_success"] += 1
                    tracker.mark_as_processed(file_path)
                    logger.info(f"Successfully processed: {file_path}")
                else:
                    stats["processed_failed"] += 1
                    logger.error(f"Failed to process {file_path}: {result['message']}")
            except Exception as e:
                stats["processed_failed"] += 1
                logger.error(f"Error processing {file_path}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Handle deleted files
        for file_path in tracked_files:
            if file_path not in current_files:
                file_name = os.path.basename(file_path)
                logger.info(f"Deleted file detected: {file_name}")
                stats["deleted_files"] += 1
                
                try:
                    # Delete chunks from Weaviate
                    await self.storage.delete_chunks(file_name)
                    
                    # Update tracker
                    tracker.remove_file(file_name)
                    logger.info(f"Successfully processed deletion: {file_name}")
                except Exception as e:
                    logger.error(f"Error processing deletion {file_name}: {str(e)}")
        
        return stats
    
# ------------ Main Function ------------
def main():
    """
    Main function to run the document processor.
    Instead of watching for file changes, this version compares the current state
    of the data folder with the tracking file on startup.
    """
    # Connect to Weaviate
    weaviate_url = config.WEAVIATE_URL
    data_folder = config.DATA_FOLDER
    
    try:
        # Set up a single event loop for the entire application
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Connect to Weaviate
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        client = loop.run_until_complete(connect_with_retry(weaviate_url))
        logger.info("Successfully connected to Weaviate")
        
        # Create storage, processor, and tracking instances
        storage = DocumentStorage(client)
        loop.run_until_complete(storage.setup_schema())
        
        tracker = ProcessingTracker(os.path.join(data_folder, ".processed_files.json"))
        processor = DocumentProcessor(
            storage,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            chunking_strategy=config.CHUNKING_STRATEGY
        )
        
        # Process all files in the data folder
        logger.info(f"Starting document processing for all files in {data_folder}")
        stats = loop.run_until_complete(processor.process_all_documents(tracker, data_folder))
        
        # Log processing summary
        logger.info(f"Document processing summary: {stats}")
        
        # Keep the application running for future manual invocations
        logger.info("Processor will remain running. No file watching is active.")
        logger.info("To process files again, restart the processor container.")
        
        # Get document count after processing
        doc_count = loop.run_until_complete(storage.get_document_count())
        logger.info(f"Current document count in database: {doc_count}")
        
        # Keep the container running without consuming resources
        try:
            import signal
            
            # Set up signal handling for clean shutdown
            def signal_handler(sig, frame):
                logger.info("Shutting down processor (signal received)")
                raise KeyboardInterrupt
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Sleep indefinitely until container is stopped
            while True:
                time.sleep(3600)  # Sleep for an hour
                
        except KeyboardInterrupt:
            logger.info("Process interrupted")
        finally:
            # Close the client connection
            client.close()
            # Close the event loop
            loop.close()
            logger.info("Processor shutdown complete")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
```


# processor\requirements.txt
```cmake
watchdog==6.0.0
weaviate-client==4.11.1
python-dotenv==1.0.1
uuid==1.30

```


# tests\quickstart_locally_hosted\docker-compose.yml
```yaml
---
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.29.0
    ports:
    - 8080:8080
    - 50051:50051
    volumes:
    - weaviate_data:/var/lib/weaviate
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      ENABLE_API_BASED_MODULES: 'true'
      ENABLE_MODULES: 'text2vec-ollama,generative-ollama'
      CLUSTER_HOSTNAME: 'node1'
volumes:
  weaviate_data:
...
```


# tests\quickstart_locally_hosted\quickstart_check_readiness.py
```python
# Check if we can connect to the Weaviate instance.
import weaviate

client = weaviate.connect_to_local()

print(client.is_ready())  # Should print: `True`

client.close()  # Free up resources
```


# tests\quickstart_locally_hosted\quickstart_create_collection.py
```python
# Define a Weviate collection, which is a set of objects that share the same data structure, 
# like a table in relational databases or a collection in NoSQL databases. 
# A collection also includes additional configurations that define how the data objects are stored and indexed.
# This script creates a collection with the name "Question" and configures it to use the Ollama embedding and generative models.
# If you prefer a different model provider integration, or prefer to import your own vectors, use a different configuration.

import weaviate
from weaviate.classes.config import Configure

client = weaviate.connect_to_local()

questions = client.collections.create(
    name="Question",
    vectorizer_config=Configure.Vectorizer.text2vec_ollama(     # Configure the Ollama embedding integration
        api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="nomic-embed-text",                               # The model to use
    ),
    generative_config=Configure.Generative.ollama(              # Configure the Ollama generative integration
        api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="llama3.2",                                       # The model to use
    )
)

client.close()  # Free up resources
```


# tests\quickstart_locally_hosted\quickstart_import.py
```python
# We can now add data to our collection.
# This scripts loads objects, and adds objects to the target collection (Question) using a batch process.
# (Batch imports) are the most efficient way to add large amounts of data, as it sends multiple objects in a single request. 
# 
import weaviate
import requests, json

client = weaviate.connect_to_local()

resp = requests.get(
    "https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json"
)
data = json.loads(resp.text)

questions = client.collections.get("Question")

with questions.batch.dynamic() as batch:
    for d in data:
        batch.add_object({
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        })
        if batch.number_errors > 10:
            print("Batch import stopped due to excessive errors.")
            break

failed_objects = questions.batch.failed_objects
if failed_objects:
    print(f"Number of failed imports: {len(failed_objects)}")
    print(f"First failed object: {failed_objects[0]}")

client.close()  # Free up resources
```


# tests\quickstart_locally_hosted\quickstart_neartext_query.py
```python
# Semantic search finds results based on meaning. This is called nearText in Weaviate.
# The following example searches for 2 objects whose meaning is most similar to that of biology.
# If you inspect the full response, you will see that the word biology does not appear anywhere.
#  Even so, Weaviate was able to return biology-related entries. 
# This is made possible by vector embeddings that capture meaning. 
# Under the hood, semantic search is powered by vectors, or vector embeddings.
import weaviate
import json

client = weaviate.connect_to_local()

questions = client.collections.get("Question")

response = questions.query.near_text(
    query="biology",
    limit=2
)

for obj in response.objects:
    print(json.dumps(obj.properties, indent=2))

client.close()  # Free up resources

```


# tests\quickstart_locally_hosted\quickstart_rag.py
```python
# Retrieval augmented generation (RAG), also called generative search, combines the power of generative AI models such as large language models (LLMs) with the up-to-date truthfulness of a database.
# RAG works by prompting a large language model (LLM) with a combination of a user query and data retrieved from a database.
import weaviate

client = weaviate.connect_to_local()

questions = client.collections.get("Question")

response = questions.generate.near_text(
    query="biology",
    limit=1,
    grouped_task="Write a tweet with emojis about these facts."
)

print(response.generated)  # Inspect the generated text

client.close()  # Free up resources

```


# tests\direct_weaviate_check.py
```python
import weaviate
import os
from dotenv import load_dotenv

# Load environment variables (if you have a .env file)
load_dotenv()

def check_weaviate_storage():
    """Check Weaviate directly to see if documents and embeddings are stored correctly"""
    print("🔍 Checking Weaviate storage directly...")
    
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local(port=8080)
        
        # Check if the client is ready
        if not client.is_ready():
            print("❌ Weaviate is not ready.")
            return
        
        print("✅ Connected to Weaviate successfully.")
        
        # Get collection info
        if not client.collections.exists("DocumentChunk"):
            print("❌ DocumentChunk collection does not exist.")
            return
        
        collection = client.collections.get("DocumentChunk")
        
        # Get collection stats
        print("\n1️⃣ Collection information:")
        try:
            objects_count = collection.aggregate.over_all().with_meta_count().do()
            print(f"Total stored chunks: {objects_count.total_count}")
        except Exception as e:
            print(f"Error getting aggregation: {str(e)}")
        
        # Get unique filenames
        print("\n2️⃣ Unique documents:")
        try:
            result = collection.query.fetch_objects(
                return_properties=["filename"],
                limit=1000  # Adjust as needed
            )
            
            unique_filenames = set()
            for obj in result.objects:
                unique_filenames.add(obj.properties["filename"])
            
            print(f"Number of unique documents: {len(unique_filenames)}")
            for filename in unique_filenames:
                print(f"- {filename}")
        except Exception as e:
            print(f"Error getting unique filenames: {str(e)}")
        
        # Check embedding vectors - get a sample
        print("\n3️⃣ Checking sample vectors:")
        try:
            # We'll get a sample chunk to check its vector
            sample = collection.query.fetch_objects(
                return_properties=["content", "filename", "chunkId"],
                include_vector=True,
                limit=1
            )
            
            if sample.objects:
                obj = sample.objects[0]
                vector = obj.vector
                
                print(f"Sample from: {obj.properties['filename']} (Chunk {obj.properties['chunkId']})")
                
                if vector:
                    vector_length = len(vector)
                    print(f"Vector exists with dimension: {vector_length}")
                    print(f"Vector sample (first 5 elements): {vector[:5]}")
                else:
                    print("❌ No vector found for this object.")
            else:
                print("No objects found in the collection.")
        except Exception as e:
            print(f"Error checking vectors: {str(e)}")
        
        # Test a simple nearest search
        print("\n4️⃣ Testing nearest neighbor search:")
        try:
            search_term = "RAG system architecture"
            print(f"Searching for: '{search_term}'")
            
            results = collection.query.near_text(
                query=search_term,
                limit=3,
                return_properties=["content", "filename", "chunkId"]
            )
            
            print(f"Found {len(results.objects)} results")
            for i, obj in enumerate(results.objects):
                print(f"\nResult {i+1}: {obj.properties['filename']} (Chunk {obj.properties['chunkId']})")
                # Truncate content for display
                content = obj.properties['content']
                if len(content) > 100:
                    content = content[:97] + "..."
                print(f"  {content}")
                
        except Exception as e:
            print(f"Error performing search: {str(e)}")
        
        print("\n✅ Weaviate check completed!")
        
    except Exception as e:
        print(f"❌ Error connecting to Weaviate: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    check_weaviate_storage()
```


# tests\document_storage_verification.py
```python
import requests
import json
import time
from typing import Dict, List, Any

API_URL = "http://localhost:8000"  # Update if using a different address

def check_system_status() -> Dict[str, str]:
    """Check if all system components are running correctly"""
    try:
        response = requests.get(f"{API_URL}/status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def count_indexed_documents() -> int:
    """Count how many unique documents are indexed in the system"""
    # This is a demonstration - you'd need to add an endpoint 
    # to your API that provides this information
    try:
        response = requests.get(f"{API_URL}/documents/count")
        response.raise_for_status()
        return response.json().get("count", 0)
    except Exception:
        return -1  # Indicates error or endpoint doesn't exist

def test_vector_search(query: str) -> Dict[str, Any]:
    """Test vector search functionality with a specific query"""
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"question": query},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_semantic_relationships(queries: List[str]) -> Dict[str, List]:
    """
    Test if semantically related queries return overlapping results,
    which indicates good vector embedding quality
    """
    results = {}
    all_filenames = set()
    
    for query in queries:
        search_result = test_vector_search(query)
        if "error" in search_result:
            results[query] = {"error": search_result["error"]}
            continue
            
        filenames = [obj.get("filename") for obj in search_result.get("results", [])]
        results[query] = filenames
        all_filenames.update(filenames)
    
    # Calculate overlap between results
    overlap_results = {}
    for i, query1 in enumerate(queries):
        overlaps = {}
        for j, query2 in enumerate(queries):
            if i != j:
                set1 = set(results[query1])
                set2 = set(results[query2])
                if set1 and set2:  # Ensure non-empty sets
                    overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                    overlaps[query2] = f"{overlap:.2f}"
        overlap_results[query1] = overlaps
    
    return {
        "individual_results": results,
        "unique_documents": list(all_filenames),
        "semantic_overlaps": overlap_results
    }

def test_rag_quality(query: str) -> Dict[str, Any]:
    """Test RAG generation quality with a specific query"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"question": query},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def run_verification_tests():
    """Run a comprehensive set of verification tests"""
    print("🔍 Running verification tests for EU-compliant RAG system...")
    
    # Step 1: Check if all components are running
    print("\n1️⃣  Checking system status...")
    status = check_system_status()
    print(json.dumps(status, indent=2))
    
    if "api" not in status or status["api"] != "running":
        print("❌ API is not running. Verification cannot continue.")
        return
    
    # Step 2: Check document storage
    print("\n2️⃣  Testing vector search...")
    test_queries = [
        "What is gdpr?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = test_vector_search(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
            
        print(f"Found {len(result.get('results', []))} relevant chunks")
        for i, chunk in enumerate(result.get("results", [])[:2]):  # Show first 2 results
            print(f"Result {i+1}: {chunk.get('filename')} (Chunk {chunk.get('chunkId')})")
            # Truncate content for display
            content = chunk.get("content", "")
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"  {content}")
    
    # Step 3: Test semantic relationships
    print("\n3️⃣  Testing semantic relationships...")
    semantic_queries = [
        "EU data regulations",
        "GDPR compliance",
        "European privacy laws",
        "Document processing", # Unrelated query for contrast
    ]
    
    semantic_results = test_semantic_relationships(semantic_queries)
    print("\nSemantic overlaps between queries (0.00 to 1.00):")
    for query, overlaps in semantic_results["semantic_overlaps"].items():
        print(f"Query: '{query}'")
        for other_query, score in overlaps.items():
            print(f"  - Overlap with '{other_query}': {score}")
    
    # Step 4: Test RAG generation
    print("\n4️⃣  Testing RAG generation...")
    rag_query = "What is the GDPR?"
    rag_result = test_rag_quality(rag_query)
    
    if "error" in rag_result:
        print(f"❌ Error: {rag_result['error']}")
    else:
        print(f"Query: '{rag_query}'")
        print("\nGenerated answer:")
        print(rag_result.get("answer", "No answer generated"))
        print("\nSources:")
        for source in rag_result.get("sources", []):
            print(f"- {source.get('filename')} (Chunk {source.get('chunkId')})")
    
    print("\n✅ Verification tests completed!")

if __name__ == "__main__":
    run_verification_tests()
```


# tests\pdf-extraction-tests.py
```python
import os
import sys
from pypdf import PdfReader

def test_pdf_direct(pdf_path):
    """
    Test PDF extraction directly without mocking or complex setup.
    This will show exactly what text is being extracted from the PDF.
    """
    print(f"Testing direct PDF extraction on: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file does not exist at {pdf_path}")
        return
    
    try:
        # Open the PDF and get basic info
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"PDF has {num_pages} pages")
        
        # Extract text from each page
        total_text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            chars = len(page_text) if page_text else 0
            has_text = chars > 0
            
            print(f"Page {i+1}: {chars} chars, Has text: {has_text}")
            
            if has_text:
                # Save the first 3 pages to see content
                if i < 3:
                    print(f"\n--- Sample from page {i+1} ---")
                    print(page_text[:300] + "..." if len(page_text) > 300 else page_text)
                    print("---\n")
                
                total_text += page_text
        
        print(f"\nTotal extracted text: {len(total_text)} characters")
        
        # Save full text to file for inspection
        output_file = os.path.join(os.path.dirname(pdf_path), "extracted_text.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(total_text)
        
        print(f"Full text saved to: {output_file}")
        
        # Also save first 3 pages to separate files for detailed inspection
        for i in range(min(3, num_pages)):
            page_text = reader.pages[i].extract_text()
            if page_text:
                page_file = os.path.join(os.path.dirname(pdf_path), f"page_{i+1}_text.txt")
                with open(page_file, "w", encoding="utf-8") as f:
                    f.write(page_text)
                print(f"Page {i+1} text saved to: {page_file}")
        
    except Exception as e:
        print(f"Error during PDF extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pdf_path = r"Vaccaro_2024.pdf"
    test_pdf_direct(pdf_path)

```


# vue-frontend\public\document-chat-icon.svg
```text
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <!-- Document base -->
  <path d="M14,4c-1.1,0-2,0.9-2,2v52c0,1.1,0.9,2,2,2h36c1.1,0,2-0.9,2-2V20L38,4H14z" fill="#f5f7fa" stroke="#4a6cf7" stroke-width="2.5"/>
  
  <!-- Document fold -->
  <path d="M38,4v14c0,1.1,0.9,2,2,2h12" fill="none" stroke="#4a6cf7" stroke-width="2.5"/>
  
  <!-- Document lines -->
  <line x1="20" y1="28" x2="44" y2="28" stroke="#bbb" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="20" y1="36" x2="44" y2="36" stroke="#bbb" stroke-width="2.5" stroke-linecap="round"/>
  
  <!-- Chat bubble -->
  <path d="M48,47c0,1.1-0.9,2-2,2H26.8L20,55v-6h-4c-1.1,0-2-0.9-2-2V33c0-1.1,0.9-2,2-2h30c1.1,0,2,0.9,2,2V47z" fill="#4a6cf7" stroke="#4a6cf7" stroke-width="0.5"/>
  
  <!-- EU stars (simplified) -->
  <g fill="#FFD700" transform="scale(0.12) translate(240, 295)">
    <circle cx="30" cy="30" r="6"/>
    <circle cx="55" cy="30" r="6"/>
    <circle cx="80" cy="30" r="6"/>
  </g>
</svg>

```


# vue-frontend\public\index.html
```text
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/document-chat-icon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>EU-Compliant Document Chat</title>
    <!-- Load runtime config -->
    <script src="/config.js"></script>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```


# vue-frontend\public\vite.svg
```text
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="31.88" height="32" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 257"><defs><linearGradient id="IconifyId1813088fe1fbc01fb466" x1="-.828%" x2="57.636%" y1="7.652%" y2="78.411%"><stop offset="0%" stop-color="#41D1FF"></stop><stop offset="100%" stop-color="#BD34FE"></stop></linearGradient><linearGradient id="IconifyId1813088fe1fbc01fb467" x1="43.376%" x2="50.316%" y1="2.242%" y2="89.03%"><stop offset="0%" stop-color="#FFEA83"></stop><stop offset="8.333%" stop-color="#FFDD35"></stop><stop offset="100%" stop-color="#FFA800"></stop></linearGradient></defs><path fill="url(#IconifyId1813088fe1fbc01fb466)" d="M255.153 37.938L134.897 252.976c-2.483 4.44-8.862 4.466-11.382.048L.875 37.958c-2.746-4.814 1.371-10.646 6.827-9.67l120.385 21.517a6.537 6.537 0 0 0 2.322-.004l117.867-21.483c5.438-.991 9.574 4.796 6.877 9.62Z"></path><path fill="url(#IconifyId1813088fe1fbc01fb467)" d="M185.432.063L96.44 17.501a3.268 3.268 0 0 0-2.634 3.014l-5.474 92.456a3.268 3.268 0 0 0 3.997 3.378l24.777-5.718c2.318-.535 4.413 1.507 3.936 3.838l-7.361 36.047c-.495 2.426 1.782 4.5 4.151 3.78l15.304-4.649c2.372-.72 4.652 1.36 4.15 3.788l-11.698 56.621c-.732 3.542 3.979 5.473 5.943 2.437l1.313-2.028l72.516-144.72c1.215-2.423-.88-5.186-3.54-4.672l-25.505 4.922c-2.396.462-4.435-1.77-3.759-4.114l16.646-57.705c.677-2.35-1.37-4.583-3.769-4.113Z"></path></svg>
```


# vue-frontend\src\assets\main.css
```text
/* Base styles */
:root {
    --primary-color: #4a6cf7;
    --secondary-color: #e53e3e;
    --text-color: #2c3e50;
    --light-gray: #f5f7fb;
    --border-color: #eaeaea;
  }
  
  body {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: var(--text-color);
    margin: 0;
    padding: 0;
  }
  
  h1, h2, h3, h4, h5, h6 {
    margin-top: 0;
  }
  
  /* Common utility classes */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
  }
  
  .btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.2s, opacity 0.2s;
  }
  
  .btn-primary {
    background-color: var(--primary-color);
    color: white;
  }
  
  .btn-primary:hover {
    background-color: #3a5cd7;
  }
  
  .btn-danger {
    background-color: var(--secondary-color);
    color: white;
  }
  
  .btn-danger:hover {
    background-color: #c53030;
  }
  
  .btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }/* Base styles */
:root {
    --primary-color: #4a6cf7;
    --secondary-color: #e53e3e;
    --text-color: #2c3e50;
    --light-gray: #f5f7fb;
    --border-color: #eaeaea;
  }
  
  body {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: var(--text-color);
    margin: 0;
    padding: 0;
  }
  
  h1, h2, h3, h4, h5, h6 {
    margin-top: 0;
  }
  
  /* Common utility classes */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
  }
  
  .btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.2s, opacity 0.2s;
  }
  
  .btn-primary {
    background-color: var(--primary-color);
    color: white;
  }
  
  .btn-primary:hover {
    background-color: #3a5cd7;
  }
  
  .btn-danger {
    background-color: var(--secondary-color);
    color: white;
  }
  
  .btn-danger:hover {
    background-color: #c53030;
  }
  
  .btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
```


# vue-frontend\src\assets\vue.svg
```text
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="iconify iconify--logos" width="37.07" height="36" preserveAspectRatio="xMidYMid meet" viewBox="0 0 256 198"><path fill="#41B883" d="M204.8 0H256L128 220.8L0 0h97.92L128 51.2L157.44 0h47.36Z"></path><path fill="#41B883" d="m0 0l128 220.8L256 0h-51.2L128 132.48L50.56 0H0Z"></path><path fill="#35495E" d="M50.56 0L128 133.12L204.8 0h-47.36L128 51.2L97.92 0H50.56Z"></path></svg>
```


# vue-frontend\src\components\chat\ChatInput.vue
```text
<template>
    <div class="chat-input-container">
      <input
        type="text"
        v-model="message"
        @keyup.enter="sendMessage"
        placeholder="Ask a question about your documents..."
        :disabled="disabled"
      />
      <button
        class="send-button"
        @click="sendMessage"
        :disabled="!message.trim() || disabled"
      >
        <span>Send</span>
      </button>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue';
  
  const props = defineProps({
    disabled: {
      type: Boolean,
      default: false
    }
  });
  
  const emit = defineEmits(['send']);
  const message = ref('');
  
  function sendMessage() {
    if (!message.value.trim() || props.disabled) return;
    
    emit('send', message.value);
    message.value = '';
  }
  </script>
  
  <style scoped>
  .chat-input-container {
    display: flex;
    gap: 8px;
    width: 100%;
  }
  
  input {
    flex: 1;
    padding: 12px 16px;
    border: 1px solid #ddd;
    border-radius: 8px;
    font-size: 16px;
  }
  
  input:focus {
    outline: none;
    border-color: #4a6cf7;
  }
  
  input:disabled {
    background-color: #f5f5f5;
    cursor: not-allowed;
  }
  
  .send-button {
    padding: 0 20px;
    background-color: #4a6cf7;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .send-button:hover:not(:disabled) {
    background-color: #3a5cd7;
  }
  
  .send-button:disabled {
    background-color: #a0aec0;
    cursor: not-allowed;
  }
  </style>
```


# vue-frontend\src\components\chat\ChatMessage.vue
```text
<template>
  <div :class="['message', message.role]">
    <div class="message-content">
      <div v-if="message.role === 'assistant'" class="avatar">🤖</div>
      <div v-else class="avatar">👤</div>
      
      <div class="content">
        <div v-if="message.error" class="error-message">
          {{ message.content }}
        </div>
        <div v-else v-html="formattedContent"></div>
        
        <!-- Sources -->
        <div v-if="message.sources && message.sources.length" class="sources">
          <h4>Sources:</h4>
          <ul>
            <li v-for="(source, index) in message.sources" :key="index">
              {{ formatCitation(source) }}
            </li>
          </ul>
        </div>
        
        <!-- Feedback buttons (only for assistant messages) -->
        <div v-if="message.role === 'assistant' && !message.error && !feedbackSubmitted" class="feedback">
          <button 
            class="feedback-btn positive" 
            @click="submitFeedback('positive')"
            :disabled="feedbackSubmitted"
          >
            👍 Helpful
          </button>
          <button 
            class="feedback-btn negative" 
            @click="showFeedbackForm = true"
            :disabled="feedbackSubmitted"
          >
            👎 Not Helpful
          </button>
        </div>

        <div v-if="feedbackSubmitted" class="feedback-confirmation">
          Thank you for your feedback!
        </div>        
        
        <!-- Detailed feedback form -->
        <div v-if="showFeedbackForm && !feedbackSubmitted" class="feedback-form">
          <textarea 
            v-model="feedbackText" 
            placeholder="What was wrong with this response?"
            :disabled="feedbackSubmitted"
          ></textarea>
          <div class="form-actions">
            <button @click="submitFeedback('negative')" :disabled="feedbackSubmitted">Submit</button>
            <button @click="showFeedbackForm = false" :disabled="feedbackSubmitted">Cancel</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useChatStore } from '../../stores/chat';
import { marked } from 'marked';

const props = defineProps({
  message: {
    type: Object,
    required: true
  }
});

const chatStore = useChatStore();
const showFeedbackForm = ref(false);
const feedbackText = ref('');

const feedbackSubmitted = ref(false);

const formattedContent = computed(() => {
  return marked(props.message.content);
});

function formatCitation(source) {
  // Extract filename and chunkId
  const filename = source.filename || 'Unknown';
  const chunkId = source.chunkId || 'Unknown';
  
  // Base citation starts with filename or title
  let citation = filename;
  
  // If we have metadata, use it to create a richer citation
  if (source.metadata) {
    const metadata = source.metadata;
    citation = metadata.title ? `${metadata.title}` : filename;
    
    if (metadata.itemType) {
      citation += ` [${metadata.itemType}]`;
    }
    
    // Add authors if available
    if (metadata.creators && metadata.creators.length > 0) {
      const authors = metadata.creators
        .filter(c => c.creatorType === 'author')
        .map(a => `${a.lastName}, ${a.firstName}`);
      
      if (authors.length > 0) {
        citation += ` by ${authors.join(', ')}`;
      }
    }
    
    // Add date if available
    if (metadata.date) {
      citation += ` (${metadata.date})`;
    }
  }
  
  // Add section/heading if available
  if (source.heading) {
    citation += `, Section: "${source.heading}"`;
  }
  
  // Add page if available
  if (source.page) {
    citation += `, Page ${source.page}`;
  }
  
  // Add chunk ID as a fallback reference
  if (!source.heading && !source.page) {
    citation += ` (Chunk ${chunkId})`;
  }
  
  return citation;
}

async function submitFeedback(rating) {
  try {
    // Get the original request ID from the message if available
    // or use the message ID as a fallback
    const originalRequestId = props.message.requestId || props.message.id;
    
    await chatStore.submitFeedback({
      originalRequestId: originalRequestId,
      messageId: props.message.id, 
      rating: rating,
      feedbackText: feedbackText.value
    });
    
    // Set feedbackSubmitted to true to disable and hide the feedback controls
    feedbackSubmitted.value = true;
    
    // Hide the feedback form
    showFeedbackForm.value = false;
    
    // Reset feedback text
    feedbackText.value = '';
  } catch (error) {
    console.error('Failed to submit feedback:', error);
    // Show an error message briefly
    alert('Failed to submit feedback. Please try again.');
  }
}
</script>

<style scoped>
.message {
  margin-bottom: 16px;
  display: flex;
}

.message-content {
  display: flex;
  max-width: 80%;
}

.assistant .message-content {
  background-color: #f0f7ff;
  border-radius: 12px;
  padding: 12px;
}

.user .message-content {
  margin-left: auto;
  background-color: #e6f7e6;
  border-radius: 12px;
  padding: 12px;
}

.avatar {
  font-size: 24px;
  margin-right: 12px;
  align-self: flex-start;
}

.sources {
  margin-top: 16px;
  font-size: 0.85rem;
  color: #555;
}

.sources ul {
  margin: 0;
  padding-left: 16px;
}

.feedback {
  margin-top: 12px;
  display: flex;
  gap: 8px;
}

.feedback-btn {
  background: none;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color 0.2s;
}

.feedback-btn.positive:hover:not(:disabled) {
  background-color: #e6f7e6;
}

.feedback-btn.negative:hover:not(:disabled) {
  background-color: #fff0f0;
}

.feedback-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.feedback-form {
  margin-top: 12px;
}

.feedback-form textarea {
  width: 100%;
  min-height: 80px;
  border: 1px solid #ccc;
  border-radius: 4px;
  padding: 8px;
  margin-bottom: 8px;
}

.feedback-form textarea:disabled {
  background-color: #f5f5f5;
  cursor: not-allowed;
}

.form-actions {
  display: flex;
  gap: 8px;
}

.form-actions button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-message {
  color: #d32f2f;
}

.feedback-confirmation {
  margin-top: 12px;
  padding: 8px;
  background-color: #e6f7e6;
  border-radius: 4px;
  font-size: 0.9rem;
  color: #2e7d32;
  text-align: center;
}
</style>
```


# vue-frontend\src\components\layout\Sidebar.vue
```text
<template>
  <div class="sidebar">
    <div class="sidebar-header">
      <h3>Document Chat</h3>
    </div>
    
    <div class="sidebar-content">
      <div class="system-status">
        <h4>System Status</h4>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(systemStatus.api)]"></span>
          <span>API Service: {{ systemStatus.api }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(systemStatus.weaviate)]"></span>
          <span>Vector Database: {{ systemStatus.weaviate }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', getStatusClass(systemStatus.mistral_api)]"></span>
          <span>LLM Service: {{ systemStatus.mistral_api }}</span>
        </div>
        <div class="status-item">
          <span :class="['status-indicator', loggingEnabled ? 'status-warning' : 'status-success']"></span>
          <span>Chat Logging: {{ loggingEnabled ? 'Enabled' : 'Disabled' }}</span>
          <span v-if="loggingEnabled" class="logging-warning" @click="showPrivacyInfo">⚠️</span>
        </div>        
      </div>
      
      <div class="sidebar-actions">
        <button class="action-button" @click="newConversation">
          🔄 New Conversation
        </button>
        <button class="action-button logout" @click="logout">
          🚪 Logout
        </button>
      </div>
    </div>
    <div v-if="showPrivacyModal" class="privacy-modal">
      <div class="privacy-content">
        <h4>Chat Logging Information</h4>
        <p>This system is currently logging chat interactions for research purposes.</p>
        <p>All logs are anonymized and automatically deleted after 30 days.</p>
        <p>For more information, please see the <a href="/privacy" target="_blank">Privacy Notice</a>.</p>
        <button @click="showPrivacyModal = false">Close</button>
      </div>
    </div>    
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useChatStore } from '../../stores/chat';
import authService from '../../services/authService';

const router = useRouter();
const chatStore = useChatStore();
const systemStatus = ref({
  api: 'unknown',
  weaviate: 'unknown',
  mistral_api: 'unknown'
});

const loggingEnabled = ref(false);
const showPrivacyModal = ref(false);

onMounted(async () => {
  // Set up config check
  let configCheckInterval = null;
  
  function checkConfig() {
    if (window.APP_CONFIG && typeof window.APP_CONFIG.enableChatLogging !== 'undefined') {
      console.log("Found APP_CONFIG", window.APP_CONFIG);
      loggingEnabled.value = window.APP_CONFIG.enableChatLogging === true || 
                            window.APP_CONFIG.enableChatLogging === "true";
      
      // Clear interval once config is found
      if (configCheckInterval) {
        clearInterval(configCheckInterval);
        configCheckInterval = null;
      }
    }
  }
  
  // Check immediately
  checkConfig();
  
  // Set up interval to check periodically
  configCheckInterval = setInterval(checkConfig, 100);
  
  // Clean up after 2 seconds max
  setTimeout(() => {
    if (configCheckInterval) {
      clearInterval(configCheckInterval);
      configCheckInterval = null;
    }
  }, 2000);
  
  // Fetch system status (independent of config check)
  try {
    const status = await chatStore.checkSystemStatus();
    if (status) {
      systemStatus.value = status;
    }
  } catch (error) {
    console.warn('Could not fetch system status:', error);
  }
});

function showPrivacyInfo() {
  showPrivacyModal.value = true;
}

function getStatusClass(status) {
  if (status === 'connected' || status === 'configured' || status === 'running') {
    return 'status-success';
  } else if (status === 'error' || status === 'disconnected') {
    return 'status-error';
  }
  return 'status-unknown';
}

function newConversation() {
  chatStore.clearChat();
}

function logout() {
  authService.logout();
  router.push('/login');
}
</script>

<style scoped>
.sidebar {
  width: 300px;
  height: 100%;
  background-color: #f5f7fb;
  border-right: 1px solid #eaeaea;
  display: flex;
  flex-direction: column;
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #eaeaea;
}

.sidebar-header h3 {
  margin: 0;
}

.sidebar-content {
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.system-status {
  margin-bottom: 24px;
}

.system-status h4 {
  margin-top: 0;
  margin-bottom: 12px;
  font-size: 16px;
}

.status-item {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-right: 8px;
}

.status-success {
  background-color: #48bb78;
}

.status-error {
  background-color: #e53e3e;
}

.status-unknown {
  background-color: #a0aec0;
}

.sidebar-actions {
  margin-top: auto;
}

.action-button {
  width: 100%;
  padding: 10px;
  margin-bottom: 8px;
  border: none;
  border-radius: 4px;
  background-color: #4a6cf7;
  color: white;
  cursor: pointer;
  font-size: 14px;
  text-align: left;
}

.action-button:hover {
  background-color: #3a5cd7;
}

.action-button.logout {
  background-color: #e53e3e;
}

.action-button.logout:hover {
  background-color: #c53030;
}

.status-warning {
  background-color: #f6ad55;
}

.logging-warning {
  margin-left: 8px;
  cursor: pointer;
}

/* Modal Styles */
.privacy-modal {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 100;
}

.privacy-content {
  background-color: white;
  padding: 16px;
  border-radius: 8px;
  max-width: 80%;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.privacy-content h4 {
  margin-top: 0;
  margin-bottom: 12px;
}

.privacy-content button {
  margin-top: 12px;
  padding: 8px 16px;
  background-color: #4a6cf7;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.privacy-content button:hover {
  background-color: #3a5cd7;
}
</style>
```


# vue-frontend\src\components\shared\Loading.vue
```text
<template>
    <div class="loading-container">
      <div class="loading-spinner"></div>
      <span v-if="text" class="loading-text">{{ text }}</span>
    </div>
  </template>
  
  <script setup>
  defineProps({
    text: {
      type: String,
      default: 'Loading...'
    }
  });
  </script>
  
  <style scoped>
  .loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  
  .loading-spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(0, 0, 0, 0.1);
    border-radius: 50%;
    border-top: 4px solid #4a6cf7;
    animation: spin 1s linear infinite;
  }
  
  .loading-text {
    margin-top: 8px;
    color: #666;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  </style>
```


# vue-frontend\src\router\index.js
```text
import { createRouter, createWebHistory } from 'vue-router'
import ChatView from '../views/ChatView.vue'
import LoginView from '../views/LoginView.vue'
import authService from '../services/authService'
import RegisterView from '../views/RegisterView.vue'

const routes = [
  {
    path: '/',
    name: 'Chat',
    component: ChatView,
    meta: { requiresAuth: true }
  },
  {
    path: '/login',
    name: 'Login',
    component: LoginView
  },
  {
    path: '/privacy',
    name: 'Privacy',
    component: () => import('../views/PrivacyView.vue') // Lazy loaded
  },
  {
    path: '/register',
    name: 'Register',
    component: RegisterView
  } 
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// Navigation guard for authentication
router.beforeEach((to, from, next) => {
  const isAuthenticated = authService.isAuthenticated()
  
  if (to.matched.some(record => record.meta.requiresAuth) && !isAuthenticated) {
    next({ name: 'Login' })
  } else {
    next()
  }
})

export default router
```


# vue-frontend\src\services\api.js
```text
import axios from 'axios';
import authService from './authService';

// Get config from window.APP_CONFIG (injected at runtime)
const config = window.APP_CONFIG || {
  apiUrl: '/api',  // Use relative path for browser requests
  apiKey: ''
};

const apiClient = axios.create({
  baseURL: config.apiUrl,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': config.apiKey  // Make sure the API key is included
  }
});

// Add request interceptor to include authorization token
apiClient.interceptors.request.use(
  config => {
    // Add authentication header if available
    const authHeader = authService.getAuthHeader();
    if (authHeader.Authorization) {
      config.headers.Authorization = authHeader.Authorization;
    }
    
    // Make sure the API key is always included
    if (window.APP_CONFIG && window.APP_CONFIG.apiKey) {
      config.headers['X-API-Key'] = window.APP_CONFIG.apiKey;
    }
    
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
  response => response,
  error => {
    if (error.response && error.response.status === 401) {
      // If we get an unauthorized error, log the user out
      authService.logout();
      // Redirect to login page
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Ensure proper error handling
apiClient.interceptors.response.use(
  response => response,
  error => {
    // Ignore canceled request errors
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }
    
    // Handle other errors
    console.error('API error:', error);
    return Promise.reject(error);
  }
);

export default apiClient;
```


# vue-frontend\src\services\authService.js
```text
import api from './api';

export default {
  async login(username, password) {
    try {
      const response = await api.post('/login', { username, password });
      
      // Store auth token and user info in localStorage
      localStorage.setItem('auth_token', response.data.access_token);
      localStorage.setItem('token_type', response.data.token_type);
      localStorage.setItem('username', response.data.username);
      localStorage.setItem('full_name', response.data.full_name || '');
      localStorage.setItem('isAuthenticated', 'true');
      
      return response.data;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  },
  
  logout() {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_type');
    localStorage.removeItem('username');
    localStorage.removeItem('full_name');
    localStorage.removeItem('isAuthenticated');
  },
  
  getAuthHeader() {
    const token = localStorage.getItem('auth_token');
    const tokenType = localStorage.getItem('token_type') || 'Bearer';
    
    if (token) {
      return { Authorization: `${tokenType} ${token}` };
    }
    return {};
  },
  
  isAuthenticated() {
    return localStorage.getItem('isAuthenticated') === 'true';
  },
  
  getCurrentUser() {
    return {
      username: localStorage.getItem('username'),
      fullName: localStorage.getItem('full_name')
    };
  }
}
```


# vue-frontend\src\services\chatService.js
```text
import api from './api';

export default {
  async sendMessage(message, conversationHistory) {
    try {
      const response = await api.post('/chat', {
        question: message,
        conversation_history: conversationHistory
      });
      return response.data;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  },
  
  async submitFeedback(feedbackData) {
    try {
      console.log('Sending feedback data to API:', feedbackData);
  
      // Validate required fields before sending
      if (!feedbackData.request_id || !feedbackData.message_id || !feedbackData.rating || !feedbackData.timestamp) {
        throw new Error('Missing required feedback fields');
      }
      
      // Ensure rating is in correct format
      if (feedbackData.rating !== 'positive' && feedbackData.rating !== 'negative') {
        throw new Error('Rating must be "positive" or "negative"');
      }
  
      const response = await api.post('/feedback', feedbackData);
      return response.data;
    } catch (error) {
      console.error('Error submitting feedback:', error);
      console.error('Error response:', error.response?.data);
      throw error;
    }
  },
  
  // Add this method for getting system status
  async getSystemStatus() {
    try {
      const response = await api.get('/status');
      return response.data;
    } catch (error) {
      console.error('Error getting system status:', error);
      throw error;
    }
  }
}
```


# vue-frontend\src\stores\chat.js
```text
import { defineStore } from 'pinia';
import chatService from '../services/chatService';
import api from '../services/api';

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [],
    conversationHistory: [],
    isLoading: false,
    error: null,
    systemStatus: {
      api: 'unknown',
      weaviate: 'unknown',
      mistral_api: 'unknown'
    }
  }),
  
  actions: {
    async sendMessage(content) {
      this.isLoading = true;
      this.error = null;
      
      try {
        // Generate a unique request ID for this conversation
        const requestId = Date.now().toString();
        
        // Add user message to UI
        this.messages.push({
          role: 'user',
          content,
          id: Date.now().toString(),
          requestId: requestId
        });
        
        // Add to conversation history for context
        this.conversationHistory.push({
          role: 'user',
          content,
          timestamp: Date.now()
        });
        
        // Send to API
        const response = await chatService.sendMessage(content, this.conversationHistory);
        
        // Add response to messages - include the same requestId for tracking
        this.messages.push({
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          id: Date.now().toString(),
          requestId: requestId  // Use the same requestId to link messages
        });
        
        // Add to conversation history
        this.conversationHistory.push({
          role: 'assistant',
          content: response.answer,
          sources: response.sources || [],
          timestamp: Date.now()
        });
        
        return response;
      } catch (error) {
        this.error = error.message || 'Failed to send message';
        // Add error message to chat
        this.messages.push({
          role: 'assistant',
          content: `Error: ${this.error}`,
          error: true,
          id: Date.now().toString()
        });
        throw error;
      } finally {
        this.isLoading = false;
      }
    },

    async submitFeedback(feedbackParams) {
      try {
        // Create a feedback object that exactly matches what the API expects
        const feedbackData = {
          request_id: Date.now().toString(), // A unique ID for this feedback submission
          message_id: feedbackParams.messageId,
          rating: feedbackParams.rating, // Must be "positive" or "negative"
          feedback_text: feedbackParams.feedbackText || null,
          categories: [], // Optional categories if implemented
          timestamp: new Date().toISOString() // Must be ISO format
        };
    
        console.log('Submitting feedback:', feedbackData);
        
        // Send the feedback to the API
        const response = await chatService.submitFeedback(feedbackData);
        console.log('Feedback submitted successfully:', response);
        
        // Store feedback with message
        const messageIndex = this.messages.findIndex(msg => msg.id === feedbackParams.messageId);
        if (messageIndex !== -1) {
          // Update message with feedback
          this.messages[messageIndex] = {
            ...this.messages[messageIndex],
            feedback: {
              rating: feedbackParams.rating,
              feedbackText: feedbackParams.feedbackText
            }
          };
        }
        
        return response;
      } catch (error) {
        console.error('Failed to submit feedback:', error);
        throw error;
      }
    },
        
    async checkSystemStatus() {
      try {
        // Use a simple GET request to the status endpoint
        const response = await api.get('/status');
        if (response && response.data) {
          this.systemStatus = response.data;
          console.log('System status updated:', this.systemStatus);
        }
        return this.systemStatus;
      } catch (error) {
        console.error('Failed to check system status:', error);
        // Don't update the system status on error
        return this.systemStatus;
      }
    },
    
    clearChat() {
      this.messages = [];
      this.conversationHistory = [];
    }
  }
});
```


# vue-frontend\src\views\ChatView.vue
```text
<template>
  <div class="chat-layout">
    <Sidebar />
    
    <div class="chat-container">
      <div class="chat-header">
        <h1>🇪🇺 Document Chat</h1>
        <button class="print-button" @click="printChat" title="Print Chat">
          🖨️ Print
        </button>
      </div>
      
      <div class="chat-messages" ref="messagesContainer">
        <p v-if="!chatStore.messages.length" class="empty-state">
          Ask a question about your documents...
        </p>
        
        <ChatMessage
          v-for="message in chatStore.messages"
          :key="message.id"
          :message="message"
        />
        
        <div v-if="chatStore.isLoading" class="loading-indicator">
          <Loading text="Thinking..." />
        </div>
      </div>
      
      <div class="chat-input">
        <ChatInput @send="sendMessage" :disabled="chatStore.isLoading" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import { useChatStore } from '../stores/chat';
import ChatMessage from '../components/chat/ChatMessage.vue';
import ChatInput from '../components/chat/ChatInput.vue';
import Sidebar from '../components/layout/Sidebar.vue';
import Loading from '../components/shared/Loading.vue';

function formatCitation(source) {
  // Extract filename and chunkId
  const filename = source.filename || 'Unknown';
  const chunkId = source.chunkId || 'Unknown';
  
  // If we have metadata, use it to create a richer citation
  if (source.metadata) {
    const metadata = source.metadata;
    let citation = metadata.title ? `${metadata.title}` : filename;
    
    if (metadata.itemType) {
      citation += ` [${metadata.itemType}]`;
    }
    
    // Add authors if available
    if (metadata.creators && metadata.creators.length > 0) {
      const authors = metadata.creators
        .filter(c => c.creatorType === 'author')
        .map(a => `${a.lastName}, ${a.firstName}`);
      
      if (authors.length > 0) {
        citation += ` by ${authors.join(', ')}`;
      }
    }
    
    // Add date if available
    if (metadata.date) {
      citation += ` (${metadata.date})`;
    }

    // Add section heading if available
    if (source.heading) {
      citation += `, Section: "${source.heading}"`;
    }    
    
    // Add page if available
    if (source.page) {
      citation += `, Page ${source.page}`;
    }

    return citation;
  }
  
  // Simple citation without metadata
  return `${filename} (Chunk ${chunkId})`;
}

const chatStore = useChatStore();
const messagesContainer = ref(null);

onMounted(() => {
  // Check system status on mount
  chatStore.checkSystemStatus();
  
  // Scroll to bottom on mount
  scrollToBottom();
});

// Watch for new messages and scroll to bottom
watch(
  () => chatStore.messages.length,
  () => {
    scrollToBottom();
  }
);

function scrollToBottom() {
  setTimeout(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
    }
  }, 100);
}

async function sendMessage(content) {
  if (!content.trim()) return;
  
  try {
    await chatStore.sendMessage(content);
    // Message added to store in action
  } catch (error) {
    console.error('Failed to send message:', error);
    // Error handled in store action
  }
}

function printChat() {
  // Create a printable version of the chat
  const printContent = document.createElement('div');
  printContent.innerHTML = `
    <h1>Document Chat Export</h1>
    <p>Generated on: ${new Date().toLocaleString()}</p>
    <hr>
    ${chatStore.messages.map(msg => {
      const role = msg.role === 'user' ? 'You' : 'Assistant';
      let html = `<p><strong>${role}:</strong> ${msg.content}</p>`;
      
      // Add sources if available
      if (msg.sources && msg.sources.length > 0) {
        html += '<p><strong>Sources:</strong></p><ul>';
        msg.sources.forEach(source => {
          // Use the same formatting as on screen
          html += `<li>${formatCitation(source)}</li>`;
        });
        html += '</ul>';
      }

      // Add feedback if available (for assistant messages)
      if (msg.role === 'assistant' && msg.feedback) {
        const feedbackText = msg.feedback.rating === 'positive' ? 
          '👍 Helpful' : '👎 Not Helpful';
        
        html += `<p><strong>Your feedback:</strong> ${feedbackText}`;
        
        if (msg.feedback.feedbackText) {
          html += ` - "${msg.feedback.feedbackText}"`;
        }
        html += `</p>`;
      }      
      
      return html + '<hr>';
    }).join('')}
  `;
  
  // Create a new window for printing
  const printWindow = window.open('', '_blank');
  printWindow.document.write(`
    <html>
      <head>
        <title>Chat Export</title>
        <style>
          body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
          hr { border: 0; border-top: 1px solid #eee; margin: 20px 0; }
          ul { margin-top: 10px; padding-left: 20px; }
          li { margin-bottom: 5px; }
        </style>
      </head>
      <body>${printContent.innerHTML}</body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus(); // Focus on the new window
  setTimeout(() => {
    printWindow.print();
  }, 500); // Short delay to ensure content is fully loaded
}
</script>

<style scoped>
.chat-layout {
  display: flex;
  height: 100vh;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.print-button {
  padding: 8px 12px;
  background-color: #4a6cf7;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.print-button:hover {
  background-color: #3a5cd7;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border-bottom: 1px solid #eaeaea;
}

.chat-header h1 {
  margin: 0;
  font-size: 24px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.chat-input {
  border-top: 1px solid #eaeaea;
  padding: 16px;
}

.empty-state {
  text-align: center;
  color: #888;
  margin-top: 40px;
}

.loading-indicator {
  display: flex;
  justify-content: center;
  margin: 16px 0;
}


</style>
```


# vue-frontend\src\views\LoginView.vue
```text
<template>
  <div class="login-container">
    <div class="login-card">
      <h1>🇪🇺 Document Chat</h1>
      <h2>Login</h2>
      
      <form @submit.prevent="login">
        <div class="form-group">
          <label for="username">Username</label>
          <input 
            type="text" 
            id="username" 
            v-model="username" 
            required
            placeholder="Enter username"
          />
        </div>
        
        <div class="form-group">
          <label for="password">Password</label>
          <input 
            type="password" 
            id="password" 
            v-model="password" 
            required
            placeholder="Enter password"
          />
        </div>

        <p v-if="error" class="error-message">{{ error }}</p>
        
        <button type="submit" :disabled="loading">
          {{ loading ? 'Logging in...' : 'Login' }}
        </button>

        <p class="register-link">
          Don't have an account? <router-link to="/register">Create one</router-link>
        </p>        

      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import authService from '../services/authService';

const router = useRouter();
const username = ref('');
const password = ref('');
const error = ref('');
const loading = ref(false);

async function login() {
  loading.value = true;
  error.value = '';
  
  try {
    await authService.login(username.value, password.value);
    router.push('/');
  } catch (err) {
    console.error('Login error:', err);
    if (err.response && err.response.data && err.response.data.detail) {
      error.value = err.response.data.detail;
    } else {
      error.value = 'Login failed. Please try again.';
    }
  } finally {
    loading.value = false;
  }
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f5f7fb;
}

.login-card {
  width: 100%;
  max-width: 400px;
  padding: 2rem;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

h1 {
  text-align: center;
  margin-bottom: 0.5rem;
}

h2 {
  text-align: center;
  margin-bottom: 2rem;
  color: #555;
}

.form-group {
  margin-bottom: 1.5rem;
}

label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

input {
  width: 100%;
  padding: 0.75rem;
  font-size: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}

button {
  width: 100%;
  padding: 0.75rem;
  font-size: 1rem;
  background-color: #4a6cf7;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:hover {
  background-color: #3a5cd7;
}

button:disabled {
  background-color: #a0aec0;
  cursor: not-allowed;
}

.register-link {
  margin-top: 1rem;
  text-align: center;
  font-size: 0.9rem;
}

.error-message {
  color: #e53e3e;
  margin-bottom: 1rem;
}
</style>
```


# vue-frontend\src\views\PrivacyView.vue
```text
<template>
    <div class="privacy-container">
      <h1>Privacy Notice</h1>
      <div class="privacy-content">
        <p>When enabled, this system may log chat interactions for research and service improvement.</p>
        
        <h2>What we collect:</h2>
        <ul>
          <li>Questions asked to the system</li>
          <li>Responses provided</li>
          <li>Document references used</li>
          <li>Anonymized session identifiers</li>
        </ul>
        
        <h2>Data Protection:</h2>
        <ul>
          <li>All identifiers are anonymized</li>
          <li>Logs are automatically deleted after 30 days</li>
          <li>Data is stored securely within the EU</li>
          <li>You can request deletion of your data</li>
        </ul>
        
        <button @click="goBack">Back to Chat</button>
      </div>
    </div>
  </template>
  
  <script setup>
  import { useRouter } from 'vue-router';
  
  const router = useRouter();
  
  function goBack() {
    router.push('/');
  }
  </script>
  
  <style scoped>
  .privacy-container {
    padding: 2rem;
    max-width: 800px;
    margin: 0 auto;
  }
  
  .privacy-content {
    background-color: white;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  h1 {
    margin-bottom: 2rem;
  }
  
  h2 {
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
  }
  
  ul {
    margin-bottom: 1.5rem;
  }
  
  button {
    margin-top: 2rem;
    padding: 0.5rem 1rem;
    background-color: #4a6cf7;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
  }
  </style>
```


# vue-frontend\src\views\RegisterView.vue
```text
<template>
    <div class="register-container">
      <div class="register-card">
        <h1>🇪🇺 Document Chat</h1>
        <h2>Create Account</h2>
        
        <form @submit.prevent="register">
          <div class="form-group">
            <label for="username">Username</label>
            <input 
              type="text" 
              id="username" 
              v-model="username" 
              required
              placeholder="Choose a username"
            />
          </div>
          
          <div class="form-group">
            <label for="password">Password</label>
            <input 
              type="password" 
              id="password" 
              v-model="password" 
              required
              placeholder="Create a strong password"
            />
            <small class="password-requirements">
              Password must be at least 8 characters with uppercase, lowercase, 
              numbers, and special characters.
            </small>
          </div>
          
          <div class="form-group">
            <label for="confirmPassword">Confirm Password</label>
            <input 
              type="password" 
              id="confirmPassword" 
              v-model="confirmPassword" 
              required
              placeholder="Confirm your password"
            />
          </div>
          
          <div class="form-group">
            <label for="fullName">Full Name (Optional)</label>
            <input 
              type="text" 
              id="fullName" 
              v-model="fullName" 
              placeholder="Your full name"
            />
          </div>
          
          <div class="form-group">
            <label for="email">Email (Optional)</label>
            <input 
              type="email" 
              id="email" 
              v-model="email" 
              placeholder="Your email address"
            />
          </div>
          <div class="form-group captcha">              
            <label>Please solve this math problem: {{ captchaQuestion }}</label>
            <input 
              type="number" 
              v-model="captchaAnswer" 
              required
              placeholder="Enter answer"
            />
            <button type="button" @click="refreshCaptcha" class="refresh-captcha">
              Refresh
            </button>
          </div>          
          
          <p v-if="error" class="error-message">{{ error }}</p>
          
          <button type="submit" :disabled="loading">
            {{ loading ? 'Creating Account...' : 'Create Account' }}
          </button>
          
          <p class="login-link">
            Already have an account? <router-link to="/login">Log in</router-link>
          </p>
        </form>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import { useRouter } from 'vue-router';
  import api from '../services/api';
  
  const router = useRouter();
  const username = ref('');
  const password = ref('');
  const confirmPassword = ref('');
  const fullName = ref('');
  const email = ref('');
  const error = ref('');
  const loading = ref(false);

  const captchaQuestion = ref('');
  const captchaAnswer = ref('');
  const captchaHash = ref('');
  const captchaTimestamp = ref(0);
  
  async function fetchCaptcha() {
    try {
      captchaAnswer.value = ''; // Clear the previous answer
      const response = await api.get('/captcha');
      captchaQuestion.value = response.data.question;
      captchaHash.value = response.data.hash;
      captchaTimestamp.value = response.data.timestamp;
    } catch (err) {
      console.error('Failed to load CAPTCHA:', err);
      captchaQuestion.value = 'Error loading math problem. Please refresh the page.';
    }
  }

  function refreshCaptcha() {
    captchaAnswer.value = '';
    fetchCaptcha();
  }

  // Load CAPTCHA when the component is mounted
  onMounted(() => {
    fetchCaptcha();
  });

  async function register() {
    if (password.value !== confirmPassword.value) {
      error.value = 'Passwords do not match';
      return;
    }

    loading.value = true;
    
    try {
      // Create FormData object
      const formData = new FormData();
      formData.append('username', username.value);
      formData.append('password', password.value);
      if (fullName.value) formData.append('full_name', fullName.value);
      if (email.value) formData.append('email', email.value);
      formData.append('captcha_answer', captchaAnswer.value);
      formData.append('captcha_hash', captchaHash.value);
      formData.append('captcha_timestamp', captchaTimestamp.value);
      
      // Send as FormData
      await api.post('/register', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      router.push('/login?registered=true');
    } catch (err) {
      console.error('Registration error:', err);
    
      if (err.response) {
        if (err.response.status === 400 && err.response.data && 
            err.response.data.detail && err.response.data.detail.includes('CAPTCHA')) {
          // For CAPTCHA errors, show a friendlier message
          error.value = 'The answer to the math problem was incorrect. Please try again.';
        } else if (err.response.data && err.response.data.detail) {
          // Other validation errors
          error.value = err.response.data.detail;
        } else {
          // Generic error with status code
          error.value = `Error ${err.response.status}: Please try again.`;
        }
      } else {
        // Network or other errors
        error.value = 'Registration failed. Please try again later.';
      }
      
      // Refresh CAPTCHA if there was an error
      fetchCaptcha();
    } finally {
      loading.value = false;
    }
  }
  </script>
  
  <style scoped>
  /* Reuse styles from LoginView.vue with some adjustments */
  .register-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    background-color: #f5f7fb;
    padding: 20px;
  }
  
  .register-card {
    width: 100%;
    max-width: 500px;
    padding: 2rem;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  /* Rest of your styles from LoginView.vue, plus: */
  .password-requirements {
    display: block;
    margin-top: 4px;
    color: #666;
    font-size: 0.8rem;
  }
  
  .login-link {
    margin-top: 1rem;
    text-align: center;
    font-size: 0.9rem;
  }
  
  .error-message {
    color: #e53e3e;
    margin-bottom: 1rem;
  }
  </style>
```


# vue-frontend\src\App.vue
```text
<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  height: 100vh;
  margin: 0;
  padding: 0;
}

body, html {
  margin: 0;
  padding: 0;
  height: 100%;
}
</style>
```


# vue-frontend\src\main.js
```text
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'

import './assets/main.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')
```


# vue-frontend\src\style.css
```text
:root {
  font-family: system-ui, Avenir, Helvetica, Arial, sans-serif;
  line-height: 1.5;
  font-weight: 400;

  color-scheme: light dark;
  color: rgba(255, 255, 255, 0.87);
  background-color: #242424;

  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

a {
  font-weight: 500;
  color: #646cff;
  text-decoration: inherit;
}
a:hover {
  color: #535bf2;
}

body {
  margin: 0;
  display: flex;
  place-items: center;
  min-width: 320px;
  min-height: 100vh;
}

h1 {
  font-size: 3.2em;
  line-height: 1.1;
}

button {
  border-radius: 8px;
  border: 1px solid transparent;
  padding: 0.6em 1.2em;
  font-size: 1em;
  font-weight: 500;
  font-family: inherit;
  background-color: #1a1a1a;
  cursor: pointer;
  transition: border-color 0.25s;
}
button:hover {
  border-color: #646cff;
}
button:focus,
button:focus-visible {
  outline: 4px auto -webkit-focus-ring-color;
}

.card {
  padding: 2em;
}

#app {
  max-width: 1280px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

@media (prefers-color-scheme: light) {
  :root {
    color: #213547;
    background-color: #ffffff;
  }
  a:hover {
    color: #747bff;
  }
  button {
    background-color: #f9f9f9;
  }
}

```


# vue-frontend\Dockerfile
```text
# Build stage
FROM node:18-alpine AS build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production stage
FROM nginx:stable-alpine AS production-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 80
CMD ["/entrypoint.sh"]
```


# vue-frontend\README.md
```markdown
# Vue 3 + Vite

This template should help get you started developing with Vue 3 in Vite. The template uses Vue 3 `<script setup>` SFCs, check out the [script setup docs](https://v3.vuejs.org/api/sfc-script-setup.html#sfc-script-setup) to learn more.

Learn more about IDE Support for Vue in the [Vue Docs Scaling up Guide](https://vuejs.org/guide/scaling-up/tooling.html#ide-support).

```


# vue-frontend\entrypoint.sh
```text
#!/bin/sh
set -e

API_URL='/api'

# Read API key from file
if [ -f "$INTERNAL_API_KEY_FILE" ]; then
  API_KEY=$(cat $INTERNAL_API_KEY_FILE)
  echo "API key found"
else
  echo "Warning: No API key file found at $INTERNAL_API_KEY_FILE"
  API_KEY=""
fi

# Parse ENABLE_CHAT_LOGGING to ensure it's a proper boolean value for JavaScript
LOGGING_ENABLED="false"
if [ "${ENABLE_CHAT_LOGGING}" = "true" ]; then
  LOGGING_ENABLED="true"
fi

# Make sure we escape any special characters in the API key for JavaScript
ESCAPED_API_KEY=$(echo "$API_KEY" | sed 's/[\&/]/\\&/g')

# Create a custom Nginx configuration file with the API key
cat > /etc/nginx/conf.d/default.conf << EOF
server {
    listen 80;
    server_name localhost;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    root /usr/share/nginx/html;
    index index.html;
    
    location / {
        try_files \$uri \$uri/ /index.html;
    }
    
    # Proxy API requests and add API key
    location /api/ {
        proxy_pass http://api:8000/api/v1/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        # Set the API key directly (from secrets)
        proxy_set_header X-API-Key "${API_KEY}";
    }
}
EOF

# Properly escape special characters for the JS code
CONFIG_SCRIPT="<script>window.APP_CONFIG = { apiUrl: '${API_URL}', apiKey: '${ESCAPED_API_KEY}', enableChatLogging: ${LOGGING_ENABLED} };</script>"
sed -i "s|</head>|${CONFIG_SCRIPT}</head>|" /usr/share/nginx/html/index.html

echo "Generated custom Nginx configuration with API key"

# Start nginx
exec nginx -g "daemon off;"
```


# vue-frontend\index.html
```text
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/document-chat-icon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>EU-Compliant Document Chat</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>

```


# vue-frontend\nginx.conf
```text
server {
    listen 80;
    server_name localhost;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    root /usr/share/nginx/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Proxy API requests and add API key
    location /api/ {
        # Read API key from environment variable or config
        proxy_pass http://api:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-API-Key "${API_KEY}";
    }
}
```


# vue-frontend\package-lock.json
```text
{
  "name": "vue-frontend",
  "version": "0.0.0",
  "lockfileVersion": 3,
  "requires": true,
  "packages": {
    "": {
      "name": "vue-frontend",
      "version": "0.0.0",
      "dependencies": {
        "axios": "^1.8.4",
        "marked": "^15.0.7",
        "pinia": "^3.0.1",
        "vue": "^3.5.13",
        "vue-router": "^4.5.0"
      },
      "devDependencies": {
        "@vitejs/plugin-vue": "^5.2.1",
        "vite": "^6.2.0"
      }
    },
    "node_modules/@babel/helper-string-parser": {
      "version": "7.25.9",
      "resolved": "https://registry.npmjs.org/@babel/helper-string-parser/-/helper-string-parser-7.25.9.tgz",
      "integrity": "sha512-4A/SCr/2KLd5jrtOMFzaKjVtAei3+2r/NChoBNoZ3EyP/+GlhoaEGoWOZUmFmoITP7zOJyHIMm+DYRd8o3PvHA==",
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/helper-validator-identifier": {
      "version": "7.25.9",
      "resolved": "https://registry.npmjs.org/@babel/helper-validator-identifier/-/helper-validator-identifier-7.25.9.tgz",
      "integrity": "sha512-Ed61U6XJc3CVRfkERJWDz4dJwKe7iLmmJsbOGu9wSloNSFttHV0I8g6UAgb7qnK5ly5bGLPd4oXZlxCdANBOWQ==",
      "license": "MIT",
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@babel/parser": {
      "version": "7.26.10",
      "resolved": "https://registry.npmjs.org/@babel/parser/-/parser-7.26.10.tgz",
      "integrity": "sha512-6aQR2zGE/QFi8JpDLjUZEPYOs7+mhKXm86VaKFiLP35JQwQb6bwUE+XbvkH0EptsYhbNBSUGaUBLKqxH1xSgsA==",
      "license": "MIT",
      "dependencies": {
        "@babel/types": "^7.26.10"
      },
      "bin": {
        "parser": "bin/babel-parser.js"
      },
      "engines": {
        "node": ">=6.0.0"
      }
    },
    "node_modules/@babel/types": {
      "version": "7.26.10",
      "resolved": "https://registry.npmjs.org/@babel/types/-/types-7.26.10.tgz",
      "integrity": "sha512-emqcG3vHrpxUKTrxcblR36dcrcoRDvKmnL/dCL6ZsHaShW80qxCAcNhzQZrpeM765VzEos+xOi4s+r4IXzTwdQ==",
      "license": "MIT",
      "dependencies": {
        "@babel/helper-string-parser": "^7.25.9",
        "@babel/helper-validator-identifier": "^7.25.9"
      },
      "engines": {
        "node": ">=6.9.0"
      }
    },
    "node_modules/@esbuild/aix-ppc64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/aix-ppc64/-/aix-ppc64-0.25.1.tgz",
      "integrity": "sha512-kfYGy8IdzTGy+z0vFGvExZtxkFlA4zAxgKEahG9KE1ScBjpQnFsNOX8KTU5ojNru5ed5CVoJYXFtoxaq5nFbjQ==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "aix"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-arm": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/android-arm/-/android-arm-0.25.1.tgz",
      "integrity": "sha512-dp+MshLYux6j/JjdqVLnMglQlFu+MuVeNrmT5nk6q07wNhCdSnB7QZj+7G8VMUGh1q+vj2Bq8kRsuyA00I/k+Q==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/android-arm64/-/android-arm64-0.25.1.tgz",
      "integrity": "sha512-50tM0zCJW5kGqgG7fQ7IHvQOcAn9TKiVRuQ/lN0xR+T2lzEFvAi1ZcS8DiksFcEpf1t/GYOeOfCAgDHFpkiSmA==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/android-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/android-x64/-/android-x64-0.25.1.tgz",
      "integrity": "sha512-GCj6WfUtNldqUzYkN/ITtlhwQqGWu9S45vUXs7EIYf+7rCiiqH9bCloatO9VhxsL0Pji+PF4Lz2XXCES+Q8hDw==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/darwin-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/darwin-arm64/-/darwin-arm64-0.25.1.tgz",
      "integrity": "sha512-5hEZKPf+nQjYoSr/elb62U19/l1mZDdqidGfmFutVUjjUZrOazAtwK+Kr+3y0C/oeJfLlxo9fXb1w7L+P7E4FQ==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/darwin-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/darwin-x64/-/darwin-x64-0.25.1.tgz",
      "integrity": "sha512-hxVnwL2Dqs3fM1IWq8Iezh0cX7ZGdVhbTfnOy5uURtao5OIVCEyj9xIzemDi7sRvKsuSdtCAhMKarxqtlyVyfA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/freebsd-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/freebsd-arm64/-/freebsd-arm64-0.25.1.tgz",
      "integrity": "sha512-1MrCZs0fZa2g8E+FUo2ipw6jw5qqQiH+tERoS5fAfKnRx6NXH31tXBKI3VpmLijLH6yriMZsxJtaXUyFt/8Y4A==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/freebsd-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/freebsd-x64/-/freebsd-x64-0.25.1.tgz",
      "integrity": "sha512-0IZWLiTyz7nm0xuIs0q1Y3QWJC52R8aSXxe40VUxm6BB1RNmkODtW6LHvWRrGiICulcX7ZvyH6h5fqdLu4gkww==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-arm": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-arm/-/linux-arm-0.25.1.tgz",
      "integrity": "sha512-NdKOhS4u7JhDKw9G3cY6sWqFcnLITn6SqivVArbzIaf3cemShqfLGHYMx8Xlm/lBit3/5d7kXvriTUGa5YViuQ==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-arm64/-/linux-arm64-0.25.1.tgz",
      "integrity": "sha512-jaN3dHi0/DDPelk0nLcXRm1q7DNJpjXy7yWaWvbfkPvI+7XNSc/lDOnCLN7gzsyzgu6qSAmgSvP9oXAhP973uQ==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-ia32": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-ia32/-/linux-ia32-0.25.1.tgz",
      "integrity": "sha512-OJykPaF4v8JidKNGz8c/q1lBO44sQNUQtq1KktJXdBLn1hPod5rE/Hko5ugKKZd+D2+o1a9MFGUEIUwO2YfgkQ==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-loong64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-loong64/-/linux-loong64-0.25.1.tgz",
      "integrity": "sha512-nGfornQj4dzcq5Vp835oM/o21UMlXzn79KobKlcs3Wz9smwiifknLy4xDCLUU0BWp7b/houtdrgUz7nOGnfIYg==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-mips64el": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-mips64el/-/linux-mips64el-0.25.1.tgz",
      "integrity": "sha512-1osBbPEFYwIE5IVB/0g2X6i1qInZa1aIoj1TdL4AaAb55xIIgbg8Doq6a5BzYWgr+tEcDzYH67XVnTmUzL+nXg==",
      "cpu": [
        "mips64el"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-ppc64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-ppc64/-/linux-ppc64-0.25.1.tgz",
      "integrity": "sha512-/6VBJOwUf3TdTvJZ82qF3tbLuWsscd7/1w+D9LH0W/SqUgM5/JJD0lrJ1fVIfZsqB6RFmLCe0Xz3fmZc3WtyVg==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-riscv64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-riscv64/-/linux-riscv64-0.25.1.tgz",
      "integrity": "sha512-nSut/Mx5gnilhcq2yIMLMe3Wl4FK5wx/o0QuuCLMtmJn+WeWYoEGDN1ipcN72g1WHsnIbxGXd4i/MF0gTcuAjQ==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-s390x": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-s390x/-/linux-s390x-0.25.1.tgz",
      "integrity": "sha512-cEECeLlJNfT8kZHqLarDBQso9a27o2Zd2AQ8USAEoGtejOrCYHNtKP8XQhMDJMtthdF4GBmjR2au3x1udADQQQ==",
      "cpu": [
        "s390x"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/linux-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/linux-x64/-/linux-x64-0.25.1.tgz",
      "integrity": "sha512-xbfUhu/gnvSEg+EGovRc+kjBAkrvtk38RlerAzQxvMzlB4fXpCFCeUAYzJvrnhFtdeyVCDANSjJvOvGYoeKzFA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/netbsd-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/netbsd-arm64/-/netbsd-arm64-0.25.1.tgz",
      "integrity": "sha512-O96poM2XGhLtpTh+s4+nP7YCCAfb4tJNRVZHfIE7dgmax+yMP2WgMd2OecBuaATHKTHsLWHQeuaxMRnCsH8+5g==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "netbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/netbsd-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/netbsd-x64/-/netbsd-x64-0.25.1.tgz",
      "integrity": "sha512-X53z6uXip6KFXBQ+Krbx25XHV/NCbzryM6ehOAeAil7X7oa4XIq+394PWGnwaSQ2WRA0KI6PUO6hTO5zeF5ijA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "netbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openbsd-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/openbsd-arm64/-/openbsd-arm64-0.25.1.tgz",
      "integrity": "sha512-Na9T3szbXezdzM/Kfs3GcRQNjHzM6GzFBeU1/6IV/npKP5ORtp9zbQjvkDJ47s6BCgaAZnnnu/cY1x342+MvZg==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/openbsd-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/openbsd-x64/-/openbsd-x64-0.25.1.tgz",
      "integrity": "sha512-T3H78X2h1tszfRSf+txbt5aOp/e7TAz3ptVKu9Oyir3IAOFPGV6O9c2naym5TOriy1l0nNf6a4X5UXRZSGX/dw==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "openbsd"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/sunos-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/sunos-x64/-/sunos-x64-0.25.1.tgz",
      "integrity": "sha512-2H3RUvcmULO7dIE5EWJH8eubZAI4xw54H1ilJnRNZdeo8dTADEZ21w6J22XBkXqGJbe0+wnNJtw3UXRoLJnFEg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "sunos"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-arm64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-arm64/-/win32-arm64-0.25.1.tgz",
      "integrity": "sha512-GE7XvrdOzrb+yVKB9KsRMq+7a2U/K5Cf/8grVFRAGJmfADr/e/ODQ134RK2/eeHqYV5eQRFxb1hY7Nr15fv1NQ==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-ia32": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-ia32/-/win32-ia32-0.25.1.tgz",
      "integrity": "sha512-uOxSJCIcavSiT6UnBhBzE8wy3n0hOkJsBOzy7HDAuTDE++1DJMRRVCPGisULScHL+a/ZwdXPpXD3IyFKjA7K8A==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@esbuild/win32-x64": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/@esbuild/win32-x64/-/win32-x64-0.25.1.tgz",
      "integrity": "sha512-Y1EQdcfwMSeQN/ujR5VayLOJ1BHaK+ssyk0AEzPjC+t1lITgsnccPqFjb6V+LsTp/9Iov4ysfjxLaGJ9RPtkVg==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ],
      "engines": {
        "node": ">=18"
      }
    },
    "node_modules/@jridgewell/sourcemap-codec": {
      "version": "1.5.0",
      "resolved": "https://registry.npmjs.org/@jridgewell/sourcemap-codec/-/sourcemap-codec-1.5.0.tgz",
      "integrity": "sha512-gv3ZRaISU3fjPAgNsriBRqGWQL6quFx04YMPW/zD8XMLsU32mhCCbfbO6KZFLjvYpCZ8zyDEgqsgf+PwPaM7GQ==",
      "license": "MIT"
    },
    "node_modules/@rollup/rollup-android-arm-eabi": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-android-arm-eabi/-/rollup-android-arm-eabi-4.36.0.tgz",
      "integrity": "sha512-jgrXjjcEwN6XpZXL0HUeOVGfjXhPyxAbbhD0BlXUB+abTOpbPiN5Wb3kOT7yb+uEtATNYF5x5gIfwutmuBA26w==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ]
    },
    "node_modules/@rollup/rollup-android-arm64": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-android-arm64/-/rollup-android-arm64-4.36.0.tgz",
      "integrity": "sha512-NyfuLvdPdNUfUNeYKUwPwKsE5SXa2J6bCt2LdB/N+AxShnkpiczi3tcLJrm5mA+eqpy0HmaIY9F6XCa32N5yzg==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "android"
      ]
    },
    "node_modules/@rollup/rollup-darwin-arm64": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-darwin-arm64/-/rollup-darwin-arm64-4.36.0.tgz",
      "integrity": "sha512-JQ1Jk5G4bGrD4pWJQzWsD8I1n1mgPXq33+/vP4sk8j/z/C2siRuxZtaUA7yMTf71TCZTZl/4e1bfzwUmFb3+rw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@rollup/rollup-darwin-x64": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-darwin-x64/-/rollup-darwin-x64-4.36.0.tgz",
      "integrity": "sha512-6c6wMZa1lrtiRsbDziCmjE53YbTkxMYhhnWnSW8R/yqsM7a6mSJ3uAVT0t8Y/DGt7gxUWYuFM4bwWk9XCJrFKA==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ]
    },
    "node_modules/@rollup/rollup-freebsd-arm64": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-freebsd-arm64/-/rollup-freebsd-arm64-4.36.0.tgz",
      "integrity": "sha512-KXVsijKeJXOl8QzXTsA+sHVDsFOmMCdBRgFmBb+mfEb/7geR7+C8ypAml4fquUt14ZyVXaw2o1FWhqAfOvA4sg==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ]
    },
    "node_modules/@rollup/rollup-freebsd-x64": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-freebsd-x64/-/rollup-freebsd-x64-4.36.0.tgz",
      "integrity": "sha512-dVeWq1ebbvByI+ndz4IJcD4a09RJgRYmLccwlQ8bPd4olz3Y213uf1iwvc7ZaxNn2ab7bjc08PrtBgMu6nb4pQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "freebsd"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm-gnueabihf": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm-gnueabihf/-/rollup-linux-arm-gnueabihf-4.36.0.tgz",
      "integrity": "sha512-bvXVU42mOVcF4le6XSjscdXjqx8okv4n5vmwgzcmtvFdifQ5U4dXFYaCB87namDRKlUL9ybVtLQ9ztnawaSzvg==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm-musleabihf": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm-musleabihf/-/rollup-linux-arm-musleabihf-4.36.0.tgz",
      "integrity": "sha512-JFIQrDJYrxOnyDQGYkqnNBtjDwTgbasdbUiQvcU8JmGDfValfH1lNpng+4FWlhaVIR4KPkeddYjsVVbmJYvDcg==",
      "cpu": [
        "arm"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm64-gnu": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm64-gnu/-/rollup-linux-arm64-gnu-4.36.0.tgz",
      "integrity": "sha512-KqjYVh3oM1bj//5X7k79PSCZ6CvaVzb7Qs7VMWS+SlWB5M8p3FqufLP9VNp4CazJ0CsPDLwVD9r3vX7Ci4J56A==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-arm64-musl": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-arm64-musl/-/rollup-linux-arm64-musl-4.36.0.tgz",
      "integrity": "sha512-QiGnhScND+mAAtfHqeT+cB1S9yFnNQ/EwCg5yE3MzoaZZnIV0RV9O5alJAoJKX/sBONVKeZdMfO8QSaWEygMhw==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-loongarch64-gnu": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-loongarch64-gnu/-/rollup-linux-loongarch64-gnu-4.36.0.tgz",
      "integrity": "sha512-1ZPyEDWF8phd4FQtTzMh8FQwqzvIjLsl6/84gzUxnMNFBtExBtpL51H67mV9xipuxl1AEAerRBgBwFNpkw8+Lg==",
      "cpu": [
        "loong64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-powerpc64le-gnu": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-powerpc64le-gnu/-/rollup-linux-powerpc64le-gnu-4.36.0.tgz",
      "integrity": "sha512-VMPMEIUpPFKpPI9GZMhJrtu8rxnp6mJR3ZzQPykq4xc2GmdHj3Q4cA+7avMyegXy4n1v+Qynr9fR88BmyO74tg==",
      "cpu": [
        "ppc64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-riscv64-gnu": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-riscv64-gnu/-/rollup-linux-riscv64-gnu-4.36.0.tgz",
      "integrity": "sha512-ttE6ayb/kHwNRJGYLpuAvB7SMtOeQnVXEIpMtAvx3kepFQeowVED0n1K9nAdraHUPJ5hydEMxBpIR7o4nrm8uA==",
      "cpu": [
        "riscv64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-s390x-gnu": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-s390x-gnu/-/rollup-linux-s390x-gnu-4.36.0.tgz",
      "integrity": "sha512-4a5gf2jpS0AIe7uBjxDeUMNcFmaRTbNv7NxI5xOCs4lhzsVyGR/0qBXduPnoWf6dGC365saTiwag8hP1imTgag==",
      "cpu": [
        "s390x"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-x64-gnu": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-x64-gnu/-/rollup-linux-x64-gnu-4.36.0.tgz",
      "integrity": "sha512-5KtoW8UWmwFKQ96aQL3LlRXX16IMwyzMq/jSSVIIyAANiE1doaQsx/KRyhAvpHlPjPiSU/AYX/8m+lQ9VToxFQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-linux-x64-musl": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-linux-x64-musl/-/rollup-linux-x64-musl-4.36.0.tgz",
      "integrity": "sha512-sycrYZPrv2ag4OCvaN5js+f01eoZ2U+RmT5as8vhxiFz+kxwlHrsxOwKPSA8WyS+Wc6Epid9QeI/IkQ9NkgYyQ==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "linux"
      ]
    },
    "node_modules/@rollup/rollup-win32-arm64-msvc": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-arm64-msvc/-/rollup-win32-arm64-msvc-4.36.0.tgz",
      "integrity": "sha512-qbqt4N7tokFwwSVlWDsjfoHgviS3n/vZ8LK0h1uLG9TYIRuUTJC88E1xb3LM2iqZ/WTqNQjYrtmtGmrmmawB6A==",
      "cpu": [
        "arm64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-ia32-msvc": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-ia32-msvc/-/rollup-win32-ia32-msvc-4.36.0.tgz",
      "integrity": "sha512-t+RY0JuRamIocMuQcfwYSOkmdX9dtkr1PbhKW42AMvaDQa+jOdpUYysroTF/nuPpAaQMWp7ye+ndlmmthieJrQ==",
      "cpu": [
        "ia32"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@rollup/rollup-win32-x64-msvc": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/@rollup/rollup-win32-x64-msvc/-/rollup-win32-x64-msvc-4.36.0.tgz",
      "integrity": "sha512-aRXd7tRZkWLqGbChgcMMDEHjOKudo1kChb1Jt1IfR8cY/KIpgNviLeJy5FUb9IpSuQj8dU2fAYNMPW/hLKOSTw==",
      "cpu": [
        "x64"
      ],
      "dev": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "win32"
      ]
    },
    "node_modules/@types/estree": {
      "version": "1.0.6",
      "resolved": "https://registry.npmjs.org/@types/estree/-/estree-1.0.6.tgz",
      "integrity": "sha512-AYnb1nQyY49te+VRAVgmzfcgjYS91mY5P0TKUDCLEM+gNnA+3T6rWITXRLYCpahpqSQbN5cE+gHpnPyXjHWxcw==",
      "dev": true,
      "license": "MIT"
    },
    "node_modules/@vitejs/plugin-vue": {
      "version": "5.2.3",
      "resolved": "https://registry.npmjs.org/@vitejs/plugin-vue/-/plugin-vue-5.2.3.tgz",
      "integrity": "sha512-IYSLEQj4LgZZuoVpdSUCw3dIynTWQgPlaRP6iAvMle4My0HdYwr5g5wQAfwOeHQBmYwEkqF70nRpSilr6PoUDg==",
      "dev": true,
      "license": "MIT",
      "engines": {
        "node": "^18.0.0 || >=20.0.0"
      },
      "peerDependencies": {
        "vite": "^5.0.0 || ^6.0.0",
        "vue": "^3.2.25"
      }
    },
    "node_modules/@vue/compiler-core": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/compiler-core/-/compiler-core-3.5.13.tgz",
      "integrity": "sha512-oOdAkwqUfW1WqpwSYJce06wvt6HljgY3fGeM9NcVA1HaYOij3mZG9Rkysn0OHuyUAGMbEbARIpsG+LPVlBJ5/Q==",
      "license": "MIT",
      "dependencies": {
        "@babel/parser": "^7.25.3",
        "@vue/shared": "3.5.13",
        "entities": "^4.5.0",
        "estree-walker": "^2.0.2",
        "source-map-js": "^1.2.0"
      }
    },
    "node_modules/@vue/compiler-dom": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/compiler-dom/-/compiler-dom-3.5.13.tgz",
      "integrity": "sha512-ZOJ46sMOKUjO3e94wPdCzQ6P1Lx/vhp2RSvfaab88Ajexs0AHeV0uasYhi99WPaogmBlRHNRuly8xV75cNTMDA==",
      "license": "MIT",
      "dependencies": {
        "@vue/compiler-core": "3.5.13",
        "@vue/shared": "3.5.13"
      }
    },
    "node_modules/@vue/compiler-sfc": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/compiler-sfc/-/compiler-sfc-3.5.13.tgz",
      "integrity": "sha512-6VdaljMpD82w6c2749Zhf5T9u5uLBWKnVue6XWxprDobftnletJ8+oel7sexFfM3qIxNmVE7LSFGTpv6obNyaQ==",
      "license": "MIT",
      "dependencies": {
        "@babel/parser": "^7.25.3",
        "@vue/compiler-core": "3.5.13",
        "@vue/compiler-dom": "3.5.13",
        "@vue/compiler-ssr": "3.5.13",
        "@vue/shared": "3.5.13",
        "estree-walker": "^2.0.2",
        "magic-string": "^0.30.11",
        "postcss": "^8.4.48",
        "source-map-js": "^1.2.0"
      }
    },
    "node_modules/@vue/compiler-ssr": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/compiler-ssr/-/compiler-ssr-3.5.13.tgz",
      "integrity": "sha512-wMH6vrYHxQl/IybKJagqbquvxpWCuVYpoUJfCqFZwa/JY1GdATAQ+TgVtgrwwMZ0D07QhA99rs/EAAWfvG6KpA==",
      "license": "MIT",
      "dependencies": {
        "@vue/compiler-dom": "3.5.13",
        "@vue/shared": "3.5.13"
      }
    },
    "node_modules/@vue/devtools-api": {
      "version": "7.7.2",
      "resolved": "https://registry.npmjs.org/@vue/devtools-api/-/devtools-api-7.7.2.tgz",
      "integrity": "sha512-1syn558KhyN+chO5SjlZIwJ8bV/bQ1nOVTG66t2RbG66ZGekyiYNmRO7X9BJCXQqPsFHlnksqvPhce2qpzxFnA==",
      "license": "MIT",
      "dependencies": {
        "@vue/devtools-kit": "^7.7.2"
      }
    },
    "node_modules/@vue/devtools-kit": {
      "version": "7.7.2",
      "resolved": "https://registry.npmjs.org/@vue/devtools-kit/-/devtools-kit-7.7.2.tgz",
      "integrity": "sha512-CY0I1JH3Z8PECbn6k3TqM1Bk9ASWxeMtTCvZr7vb+CHi+X/QwQm5F1/fPagraamKMAHVfuuCbdcnNg1A4CYVWQ==",
      "license": "MIT",
      "dependencies": {
        "@vue/devtools-shared": "^7.7.2",
        "birpc": "^0.2.19",
        "hookable": "^5.5.3",
        "mitt": "^3.0.1",
        "perfect-debounce": "^1.0.0",
        "speakingurl": "^14.0.1",
        "superjson": "^2.2.1"
      }
    },
    "node_modules/@vue/devtools-shared": {
      "version": "7.7.2",
      "resolved": "https://registry.npmjs.org/@vue/devtools-shared/-/devtools-shared-7.7.2.tgz",
      "integrity": "sha512-uBFxnp8gwW2vD6FrJB8JZLUzVb6PNRG0B0jBnHsOH8uKyva2qINY8PTF5Te4QlTbMDqU5K6qtJDr6cNsKWhbOA==",
      "license": "MIT",
      "dependencies": {
        "rfdc": "^1.4.1"
      }
    },
    "node_modules/@vue/reactivity": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/reactivity/-/reactivity-3.5.13.tgz",
      "integrity": "sha512-NaCwtw8o48B9I6L1zl2p41OHo/2Z4wqYGGIK1Khu5T7yxrn+ATOixn/Udn2m+6kZKB/J7cuT9DbWWhRxqixACg==",
      "license": "MIT",
      "dependencies": {
        "@vue/shared": "3.5.13"
      }
    },
    "node_modules/@vue/runtime-core": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/runtime-core/-/runtime-core-3.5.13.tgz",
      "integrity": "sha512-Fj4YRQ3Az0WTZw1sFe+QDb0aXCerigEpw418pw1HBUKFtnQHWzwojaukAs2X/c9DQz4MQ4bsXTGlcpGxU/RCIw==",
      "license": "MIT",
      "dependencies": {
        "@vue/reactivity": "3.5.13",
        "@vue/shared": "3.5.13"
      }
    },
    "node_modules/@vue/runtime-dom": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/runtime-dom/-/runtime-dom-3.5.13.tgz",
      "integrity": "sha512-dLaj94s93NYLqjLiyFzVs9X6dWhTdAlEAciC3Moq7gzAc13VJUdCnjjRurNM6uTLFATRHexHCTu/Xp3eW6yoog==",
      "license": "MIT",
      "dependencies": {
        "@vue/reactivity": "3.5.13",
        "@vue/runtime-core": "3.5.13",
        "@vue/shared": "3.5.13",
        "csstype": "^3.1.3"
      }
    },
    "node_modules/@vue/server-renderer": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/server-renderer/-/server-renderer-3.5.13.tgz",
      "integrity": "sha512-wAi4IRJV/2SAW3htkTlB+dHeRmpTiVIK1OGLWV1yeStVSebSQQOwGwIq0D3ZIoBj2C2qpgz5+vX9iEBkTdk5YA==",
      "license": "MIT",
      "dependencies": {
        "@vue/compiler-ssr": "3.5.13",
        "@vue/shared": "3.5.13"
      },
      "peerDependencies": {
        "vue": "3.5.13"
      }
    },
    "node_modules/@vue/shared": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/@vue/shared/-/shared-3.5.13.tgz",
      "integrity": "sha512-/hnE/qP5ZoGpol0a5mDi45bOd7t3tjYJBjsgCsivow7D48cJeV5l05RD82lPqi7gRiphZM37rnhW1l6ZoCNNnQ==",
      "license": "MIT"
    },
    "node_modules/asynckit": {
      "version": "0.4.0",
      "resolved": "https://registry.npmjs.org/asynckit/-/asynckit-0.4.0.tgz",
      "integrity": "sha512-Oei9OH4tRh0YqU3GxhX79dM/mwVgvbZJaSNaRk+bshkj0S5cfHcgYakreBjrHwatXKbz+IoIdYLxrKim2MjW0Q==",
      "license": "MIT"
    },
    "node_modules/axios": {
      "version": "1.8.4",
      "resolved": "https://registry.npmjs.org/axios/-/axios-1.8.4.tgz",
      "integrity": "sha512-eBSYY4Y68NNlHbHBMdeDmKNtDgXWhQsJcGqzO3iLUM0GraQFSS9cVgPX5I9b3lbdFKyYoAEGAZF1DwhTaljNAw==",
      "license": "MIT",
      "dependencies": {
        "follow-redirects": "^1.15.6",
        "form-data": "^4.0.0",
        "proxy-from-env": "^1.1.0"
      }
    },
    "node_modules/birpc": {
      "version": "0.2.19",
      "resolved": "https://registry.npmjs.org/birpc/-/birpc-0.2.19.tgz",
      "integrity": "sha512-5WeXXAvTmitV1RqJFppT5QtUiz2p1mRSYU000Jkft5ZUCLJIk4uQriYNO50HknxKwM6jd8utNc66K1qGIwwWBQ==",
      "license": "MIT",
      "funding": {
        "url": "https://github.com/sponsors/antfu"
      }
    },
    "node_modules/call-bind-apply-helpers": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/call-bind-apply-helpers/-/call-bind-apply-helpers-1.0.2.tgz",
      "integrity": "sha512-Sp1ablJ0ivDkSzjcaJdxEunN5/XvksFJ2sMBFfq6x0ryhQV/2b/KwFe21cMpmHtPOSij8K99/wSfoEuTObmuMQ==",
      "license": "MIT",
      "dependencies": {
        "es-errors": "^1.3.0",
        "function-bind": "^1.1.2"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/combined-stream": {
      "version": "1.0.8",
      "resolved": "https://registry.npmjs.org/combined-stream/-/combined-stream-1.0.8.tgz",
      "integrity": "sha512-FQN4MRfuJeHf7cBbBMJFXhKSDq+2kAArBlmRBvcvFE5BB1HZKXtSFASDhdlz9zOYwxh8lDdnvmMOe/+5cdoEdg==",
      "license": "MIT",
      "dependencies": {
        "delayed-stream": "~1.0.0"
      },
      "engines": {
        "node": ">= 0.8"
      }
    },
    "node_modules/copy-anything": {
      "version": "3.0.5",
      "resolved": "https://registry.npmjs.org/copy-anything/-/copy-anything-3.0.5.tgz",
      "integrity": "sha512-yCEafptTtb4bk7GLEQoM8KVJpxAfdBJYaXyzQEgQQQgYrZiDp8SJmGKlYza6CYjEDNstAdNdKA3UuoULlEbS6w==",
      "license": "MIT",
      "dependencies": {
        "is-what": "^4.1.8"
      },
      "engines": {
        "node": ">=12.13"
      },
      "funding": {
        "url": "https://github.com/sponsors/mesqueeb"
      }
    },
    "node_modules/csstype": {
      "version": "3.1.3",
      "resolved": "https://registry.npmjs.org/csstype/-/csstype-3.1.3.tgz",
      "integrity": "sha512-M1uQkMl8rQK/szD0LNhtqxIPLpimGm8sOBwU7lLnCpSbTyY3yeU1Vc7l4KT5zT4s/yOxHH5O7tIuuLOCnLADRw==",
      "license": "MIT"
    },
    "node_modules/delayed-stream": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/delayed-stream/-/delayed-stream-1.0.0.tgz",
      "integrity": "sha512-ZySD7Nf91aLB0RxL4KGrKHBXl7Eds1DAmEdcoVawXnLD7SDhpNgtuII2aAkg7a7QS41jxPSZ17p4VdGnMHk3MQ==",
      "license": "MIT",
      "engines": {
        "node": ">=0.4.0"
      }
    },
    "node_modules/dunder-proto": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/dunder-proto/-/dunder-proto-1.0.1.tgz",
      "integrity": "sha512-KIN/nDJBQRcXw0MLVhZE9iQHmG68qAVIBg9CqmUYjmQIhgij9U5MFvrqkUL5FbtyyzZuOeOt0zdeRe4UY7ct+A==",
      "license": "MIT",
      "dependencies": {
        "call-bind-apply-helpers": "^1.0.1",
        "es-errors": "^1.3.0",
        "gopd": "^1.2.0"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/entities": {
      "version": "4.5.0",
      "resolved": "https://registry.npmjs.org/entities/-/entities-4.5.0.tgz",
      "integrity": "sha512-V0hjH4dGPh9Ao5p0MoRY6BVqtwCjhz6vI5LT8AJ55H+4g9/4vbHx1I54fS0XuclLhDHArPQCiMjDxjaL8fPxhw==",
      "license": "BSD-2-Clause",
      "engines": {
        "node": ">=0.12"
      },
      "funding": {
        "url": "https://github.com/fb55/entities?sponsor=1"
      }
    },
    "node_modules/es-define-property": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/es-define-property/-/es-define-property-1.0.1.tgz",
      "integrity": "sha512-e3nRfgfUZ4rNGL232gUgX06QNyyez04KdjFrF+LTRoOXmrOgFKDg4BCdsjW8EnT69eqdYGmRpJwiPVYNrCaW3g==",
      "license": "MIT",
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/es-errors": {
      "version": "1.3.0",
      "resolved": "https://registry.npmjs.org/es-errors/-/es-errors-1.3.0.tgz",
      "integrity": "sha512-Zf5H2Kxt2xjTvbJvP2ZWLEICxA6j+hAmMzIlypy4xcBg1vKVnx89Wy0GbS+kf5cwCVFFzdCFh2XSCFNULS6csw==",
      "license": "MIT",
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/es-object-atoms": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/es-object-atoms/-/es-object-atoms-1.1.1.tgz",
      "integrity": "sha512-FGgH2h8zKNim9ljj7dankFPcICIK9Cp5bm+c2gQSYePhpaG5+esrLODihIorn+Pe6FGJzWhXQotPv73jTaldXA==",
      "license": "MIT",
      "dependencies": {
        "es-errors": "^1.3.0"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/es-set-tostringtag": {
      "version": "2.1.0",
      "resolved": "https://registry.npmjs.org/es-set-tostringtag/-/es-set-tostringtag-2.1.0.tgz",
      "integrity": "sha512-j6vWzfrGVfyXxge+O0x5sh6cvxAog0a/4Rdd2K36zCMV5eJ+/+tOAngRO8cODMNWbVRdVlmGZQL2YS3yR8bIUA==",
      "license": "MIT",
      "dependencies": {
        "es-errors": "^1.3.0",
        "get-intrinsic": "^1.2.6",
        "has-tostringtag": "^1.0.2",
        "hasown": "^2.0.2"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/esbuild": {
      "version": "0.25.1",
      "resolved": "https://registry.npmjs.org/esbuild/-/esbuild-0.25.1.tgz",
      "integrity": "sha512-BGO5LtrGC7vxnqucAe/rmvKdJllfGaYWdyABvyMoXQlfYMb2bbRuReWR5tEGE//4LcNJj9XrkovTqNYRFZHAMQ==",
      "dev": true,
      "hasInstallScript": true,
      "license": "MIT",
      "bin": {
        "esbuild": "bin/esbuild"
      },
      "engines": {
        "node": ">=18"
      },
      "optionalDependencies": {
        "@esbuild/aix-ppc64": "0.25.1",
        "@esbuild/android-arm": "0.25.1",
        "@esbuild/android-arm64": "0.25.1",
        "@esbuild/android-x64": "0.25.1",
        "@esbuild/darwin-arm64": "0.25.1",
        "@esbuild/darwin-x64": "0.25.1",
        "@esbuild/freebsd-arm64": "0.25.1",
        "@esbuild/freebsd-x64": "0.25.1",
        "@esbuild/linux-arm": "0.25.1",
        "@esbuild/linux-arm64": "0.25.1",
        "@esbuild/linux-ia32": "0.25.1",
        "@esbuild/linux-loong64": "0.25.1",
        "@esbuild/linux-mips64el": "0.25.1",
        "@esbuild/linux-ppc64": "0.25.1",
        "@esbuild/linux-riscv64": "0.25.1",
        "@esbuild/linux-s390x": "0.25.1",
        "@esbuild/linux-x64": "0.25.1",
        "@esbuild/netbsd-arm64": "0.25.1",
        "@esbuild/netbsd-x64": "0.25.1",
        "@esbuild/openbsd-arm64": "0.25.1",
        "@esbuild/openbsd-x64": "0.25.1",
        "@esbuild/sunos-x64": "0.25.1",
        "@esbuild/win32-arm64": "0.25.1",
        "@esbuild/win32-ia32": "0.25.1",
        "@esbuild/win32-x64": "0.25.1"
      }
    },
    "node_modules/estree-walker": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/estree-walker/-/estree-walker-2.0.2.tgz",
      "integrity": "sha512-Rfkk/Mp/DL7JVje3u18FxFujQlTNR2q6QfMSMB7AvCBx91NGj/ba3kCfza0f6dVDbw7YlRf/nDrn7pQrCCyQ/w==",
      "license": "MIT"
    },
    "node_modules/follow-redirects": {
      "version": "1.15.9",
      "resolved": "https://registry.npmjs.org/follow-redirects/-/follow-redirects-1.15.9.tgz",
      "integrity": "sha512-gew4GsXizNgdoRyqmyfMHyAmXsZDk6mHkSxZFCzW9gwlbtOW44CDtYavM+y+72qD/Vq2l550kMF52DT8fOLJqQ==",
      "funding": [
        {
          "type": "individual",
          "url": "https://github.com/sponsors/RubenVerborgh"
        }
      ],
      "license": "MIT",
      "engines": {
        "node": ">=4.0"
      },
      "peerDependenciesMeta": {
        "debug": {
          "optional": true
        }
      }
    },
    "node_modules/form-data": {
      "version": "4.0.2",
      "resolved": "https://registry.npmjs.org/form-data/-/form-data-4.0.2.tgz",
      "integrity": "sha512-hGfm/slu0ZabnNt4oaRZ6uREyfCj6P4fT/n6A1rGV+Z0VdGXjfOhVUpkn6qVQONHGIFwmveGXyDs75+nr6FM8w==",
      "license": "MIT",
      "dependencies": {
        "asynckit": "^0.4.0",
        "combined-stream": "^1.0.8",
        "es-set-tostringtag": "^2.1.0",
        "mime-types": "^2.1.12"
      },
      "engines": {
        "node": ">= 6"
      }
    },
    "node_modules/fsevents": {
      "version": "2.3.3",
      "resolved": "https://registry.npmjs.org/fsevents/-/fsevents-2.3.3.tgz",
      "integrity": "sha512-5xoDfX+fL7faATnagmWPpbFtwh/R77WmMMqqHGS65C3vvB0YHrgF+B1YmZ3441tMj5n63k0212XNoJwzlhffQw==",
      "dev": true,
      "hasInstallScript": true,
      "license": "MIT",
      "optional": true,
      "os": [
        "darwin"
      ],
      "engines": {
        "node": "^8.16.0 || ^10.6.0 || >=11.0.0"
      }
    },
    "node_modules/function-bind": {
      "version": "1.1.2",
      "resolved": "https://registry.npmjs.org/function-bind/-/function-bind-1.1.2.tgz",
      "integrity": "sha512-7XHNxH7qX9xG5mIwxkhumTox/MIRNcOgDrxWsMt2pAr23WHp6MrRlN7FBSFpCpr+oVO0F744iUgR82nJMfG2SA==",
      "license": "MIT",
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/get-intrinsic": {
      "version": "1.3.0",
      "resolved": "https://registry.npmjs.org/get-intrinsic/-/get-intrinsic-1.3.0.tgz",
      "integrity": "sha512-9fSjSaos/fRIVIp+xSJlE6lfwhES7LNtKaCBIamHsjr2na1BiABJPo0mOjjz8GJDURarmCPGqaiVg5mfjb98CQ==",
      "license": "MIT",
      "dependencies": {
        "call-bind-apply-helpers": "^1.0.2",
        "es-define-property": "^1.0.1",
        "es-errors": "^1.3.0",
        "es-object-atoms": "^1.1.1",
        "function-bind": "^1.1.2",
        "get-proto": "^1.0.1",
        "gopd": "^1.2.0",
        "has-symbols": "^1.1.0",
        "hasown": "^2.0.2",
        "math-intrinsics": "^1.1.0"
      },
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/get-proto": {
      "version": "1.0.1",
      "resolved": "https://registry.npmjs.org/get-proto/-/get-proto-1.0.1.tgz",
      "integrity": "sha512-sTSfBjoXBp89JvIKIefqw7U2CCebsc74kiY6awiGogKtoSGbgjYE/G/+l9sF3MWFPNc9IcoOC4ODfKHfxFmp0g==",
      "license": "MIT",
      "dependencies": {
        "dunder-proto": "^1.0.1",
        "es-object-atoms": "^1.0.0"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/gopd": {
      "version": "1.2.0",
      "resolved": "https://registry.npmjs.org/gopd/-/gopd-1.2.0.tgz",
      "integrity": "sha512-ZUKRh6/kUFoAiTAtTYPZJ3hw9wNxx+BIBOijnlG9PnrJsCcSjs1wyyD6vJpaYtgnzDrKYRSqf3OO6Rfa93xsRg==",
      "license": "MIT",
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/has-symbols": {
      "version": "1.1.0",
      "resolved": "https://registry.npmjs.org/has-symbols/-/has-symbols-1.1.0.tgz",
      "integrity": "sha512-1cDNdwJ2Jaohmb3sg4OmKaMBwuC48sYni5HUw2DvsC8LjGTLK9h+eb1X6RyuOHe4hT0ULCW68iomhjUoKUqlPQ==",
      "license": "MIT",
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/has-tostringtag": {
      "version": "1.0.2",
      "resolved": "https://registry.npmjs.org/has-tostringtag/-/has-tostringtag-1.0.2.tgz",
      "integrity": "sha512-NqADB8VjPFLM2V0VvHUewwwsw0ZWBaIdgo+ieHtK3hasLz4qeCRjYcqfB6AQrBggRKppKF8L52/VqdVsO47Dlw==",
      "license": "MIT",
      "dependencies": {
        "has-symbols": "^1.0.3"
      },
      "engines": {
        "node": ">= 0.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/ljharb"
      }
    },
    "node_modules/hasown": {
      "version": "2.0.2",
      "resolved": "https://registry.npmjs.org/hasown/-/hasown-2.0.2.tgz",
      "integrity": "sha512-0hJU9SCPvmMzIBdZFqNPXWa6dqh7WdH0cII9y+CyS8rG3nL48Bclra9HmKhVVUHyPWNH5Y7xDwAB7bfgSjkUMQ==",
      "license": "MIT",
      "dependencies": {
        "function-bind": "^1.1.2"
      },
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/hookable": {
      "version": "5.5.3",
      "resolved": "https://registry.npmjs.org/hookable/-/hookable-5.5.3.tgz",
      "integrity": "sha512-Yc+BQe8SvoXH1643Qez1zqLRmbA5rCL+sSmk6TVos0LWVfNIB7PGncdlId77WzLGSIB5KaWgTaNTs2lNVEI6VQ==",
      "license": "MIT"
    },
    "node_modules/is-what": {
      "version": "4.1.16",
      "resolved": "https://registry.npmjs.org/is-what/-/is-what-4.1.16.tgz",
      "integrity": "sha512-ZhMwEosbFJkA0YhFnNDgTM4ZxDRsS6HqTo7qsZM08fehyRYIYa0yHu5R6mgo1n/8MgaPBXiPimPD77baVFYg+A==",
      "license": "MIT",
      "engines": {
        "node": ">=12.13"
      },
      "funding": {
        "url": "https://github.com/sponsors/mesqueeb"
      }
    },
    "node_modules/magic-string": {
      "version": "0.30.17",
      "resolved": "https://registry.npmjs.org/magic-string/-/magic-string-0.30.17.tgz",
      "integrity": "sha512-sNPKHvyjVf7gyjwS4xGTaW/mCnF8wnjtifKBEhxfZ7E/S8tQ0rssrwGNn6q8JH/ohItJfSQp9mBtQYuTlH5QnA==",
      "license": "MIT",
      "dependencies": {
        "@jridgewell/sourcemap-codec": "^1.5.0"
      }
    },
    "node_modules/marked": {
      "version": "15.0.7",
      "resolved": "https://registry.npmjs.org/marked/-/marked-15.0.7.tgz",
      "integrity": "sha512-dgLIeKGLx5FwziAnsk4ONoGwHwGPJzselimvlVskE9XLN4Orv9u2VA3GWw/lYUqjfA0rUT/6fqKwfZJapP9BEg==",
      "license": "MIT",
      "bin": {
        "marked": "bin/marked.js"
      },
      "engines": {
        "node": ">= 18"
      }
    },
    "node_modules/math-intrinsics": {
      "version": "1.1.0",
      "resolved": "https://registry.npmjs.org/math-intrinsics/-/math-intrinsics-1.1.0.tgz",
      "integrity": "sha512-/IXtbwEk5HTPyEwyKX6hGkYXxM9nbj64B+ilVJnC/R6B0pH5G4V3b0pVbL7DBj4tkhBAppbQUlf6F6Xl9LHu1g==",
      "license": "MIT",
      "engines": {
        "node": ">= 0.4"
      }
    },
    "node_modules/mime-db": {
      "version": "1.52.0",
      "resolved": "https://registry.npmjs.org/mime-db/-/mime-db-1.52.0.tgz",
      "integrity": "sha512-sPU4uV7dYlvtWJxwwxHD0PuihVNiE7TyAbQ5SWxDCB9mUYvOgroQOwYQQOKPJ8CIbE+1ETVlOoK1UC2nU3gYvg==",
      "license": "MIT",
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/mime-types": {
      "version": "2.1.35",
      "resolved": "https://registry.npmjs.org/mime-types/-/mime-types-2.1.35.tgz",
      "integrity": "sha512-ZDY+bPm5zTTF+YpCrAU9nK0UgICYPT0QtT1NZWFv4s++TNkcgVaT0g6+4R2uI4MjQjzysHB1zxuWL50hzaeXiw==",
      "license": "MIT",
      "dependencies": {
        "mime-db": "1.52.0"
      },
      "engines": {
        "node": ">= 0.6"
      }
    },
    "node_modules/mitt": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/mitt/-/mitt-3.0.1.tgz",
      "integrity": "sha512-vKivATfr97l2/QBCYAkXYDbrIWPM2IIKEl7YPhjCvKlG3kE2gm+uBo6nEXK3M5/Ffh/FLpKExzOQ3JJoJGFKBw==",
      "license": "MIT"
    },
    "node_modules/nanoid": {
      "version": "3.3.11",
      "resolved": "https://registry.npmjs.org/nanoid/-/nanoid-3.3.11.tgz",
      "integrity": "sha512-N8SpfPUnUp1bK+PMYW8qSWdl9U+wwNWI4QKxOYDy9JAro3WMX7p2OeVRF9v+347pnakNevPmiHhNmZ2HbFA76w==",
      "funding": [
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "bin": {
        "nanoid": "bin/nanoid.cjs"
      },
      "engines": {
        "node": "^10 || ^12 || ^13.7 || ^14 || >=15.0.1"
      }
    },
    "node_modules/perfect-debounce": {
      "version": "1.0.0",
      "resolved": "https://registry.npmjs.org/perfect-debounce/-/perfect-debounce-1.0.0.tgz",
      "integrity": "sha512-xCy9V055GLEqoFaHoC1SoLIaLmWctgCUaBaWxDZ7/Zx4CTyX7cJQLJOok/orfjZAh9kEYpjJa4d0KcJmCbctZA==",
      "license": "MIT"
    },
    "node_modules/picocolors": {
      "version": "1.1.1",
      "resolved": "https://registry.npmjs.org/picocolors/-/picocolors-1.1.1.tgz",
      "integrity": "sha512-xceH2snhtb5M9liqDsmEw56le376mTZkEX/jEb/RxNFyegNul7eNslCXP9FDj/Lcu0X8KEyMceP2ntpaHrDEVA==",
      "license": "ISC"
    },
    "node_modules/pinia": {
      "version": "3.0.1",
      "resolved": "https://registry.npmjs.org/pinia/-/pinia-3.0.1.tgz",
      "integrity": "sha512-WXglsDzztOTH6IfcJ99ltYZin2mY8XZCXujkYWVIJlBjqsP6ST7zw+Aarh63E1cDVYeyUcPCxPHzJpEOmzB6Wg==",
      "license": "MIT",
      "dependencies": {
        "@vue/devtools-api": "^7.7.2"
      },
      "funding": {
        "url": "https://github.com/sponsors/posva"
      },
      "peerDependencies": {
        "typescript": ">=4.4.4",
        "vue": "^2.7.0 || ^3.5.11"
      },
      "peerDependenciesMeta": {
        "typescript": {
          "optional": true
        }
      }
    },
    "node_modules/postcss": {
      "version": "8.5.3",
      "resolved": "https://registry.npmjs.org/postcss/-/postcss-8.5.3.tgz",
      "integrity": "sha512-dle9A3yYxlBSrt8Fu+IpjGT8SY8hN0mlaA6GY8t0P5PjIOZemULz/E2Bnm/2dcUOena75OTNkHI76uZBNUUq3A==",
      "funding": [
        {
          "type": "opencollective",
          "url": "https://opencollective.com/postcss/"
        },
        {
          "type": "tidelift",
          "url": "https://tidelift.com/funding/github/npm/postcss"
        },
        {
          "type": "github",
          "url": "https://github.com/sponsors/ai"
        }
      ],
      "license": "MIT",
      "dependencies": {
        "nanoid": "^3.3.8",
        "picocolors": "^1.1.1",
        "source-map-js": "^1.2.1"
      },
      "engines": {
        "node": "^10 || ^12 || >=14"
      }
    },
    "node_modules/proxy-from-env": {
      "version": "1.1.0",
      "resolved": "https://registry.npmjs.org/proxy-from-env/-/proxy-from-env-1.1.0.tgz",
      "integrity": "sha512-D+zkORCbA9f1tdWRK0RaCR3GPv50cMxcrz4X8k5LTSUD1Dkw47mKJEZQNunItRTkWwgtaUSo1RVFRIG9ZXiFYg==",
      "license": "MIT"
    },
    "node_modules/rfdc": {
      "version": "1.4.1",
      "resolved": "https://registry.npmjs.org/rfdc/-/rfdc-1.4.1.tgz",
      "integrity": "sha512-q1b3N5QkRUWUl7iyylaaj3kOpIT0N2i9MqIEQXP73GVsN9cw3fdx8X63cEmWhJGi2PPCF23Ijp7ktmd39rawIA==",
      "license": "MIT"
    },
    "node_modules/rollup": {
      "version": "4.36.0",
      "resolved": "https://registry.npmjs.org/rollup/-/rollup-4.36.0.tgz",
      "integrity": "sha512-zwATAXNQxUcd40zgtQG0ZafcRK4g004WtEl7kbuhTWPvf07PsfohXl39jVUvPF7jvNAIkKPQ2XrsDlWuxBd++Q==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "@types/estree": "1.0.6"
      },
      "bin": {
        "rollup": "dist/bin/rollup"
      },
      "engines": {
        "node": ">=18.0.0",
        "npm": ">=8.0.0"
      },
      "optionalDependencies": {
        "@rollup/rollup-android-arm-eabi": "4.36.0",
        "@rollup/rollup-android-arm64": "4.36.0",
        "@rollup/rollup-darwin-arm64": "4.36.0",
        "@rollup/rollup-darwin-x64": "4.36.0",
        "@rollup/rollup-freebsd-arm64": "4.36.0",
        "@rollup/rollup-freebsd-x64": "4.36.0",
        "@rollup/rollup-linux-arm-gnueabihf": "4.36.0",
        "@rollup/rollup-linux-arm-musleabihf": "4.36.0",
        "@rollup/rollup-linux-arm64-gnu": "4.36.0",
        "@rollup/rollup-linux-arm64-musl": "4.36.0",
        "@rollup/rollup-linux-loongarch64-gnu": "4.36.0",
        "@rollup/rollup-linux-powerpc64le-gnu": "4.36.0",
        "@rollup/rollup-linux-riscv64-gnu": "4.36.0",
        "@rollup/rollup-linux-s390x-gnu": "4.36.0",
        "@rollup/rollup-linux-x64-gnu": "4.36.0",
        "@rollup/rollup-linux-x64-musl": "4.36.0",
        "@rollup/rollup-win32-arm64-msvc": "4.36.0",
        "@rollup/rollup-win32-ia32-msvc": "4.36.0",
        "@rollup/rollup-win32-x64-msvc": "4.36.0",
        "fsevents": "~2.3.2"
      }
    },
    "node_modules/source-map-js": {
      "version": "1.2.1",
      "resolved": "https://registry.npmjs.org/source-map-js/-/source-map-js-1.2.1.tgz",
      "integrity": "sha512-UXWMKhLOwVKb728IUtQPXxfYU+usdybtUrK/8uGE8CQMvrhOpwvzDBwj0QhSL7MQc7vIsISBG8VQ8+IDQxpfQA==",
      "license": "BSD-3-Clause",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/speakingurl": {
      "version": "14.0.1",
      "resolved": "https://registry.npmjs.org/speakingurl/-/speakingurl-14.0.1.tgz",
      "integrity": "sha512-1POYv7uv2gXoyGFpBCmpDVSNV74IfsWlDW216UPjbWufNf+bSU6GdbDsxdcxtfwb4xlI3yxzOTKClUosxARYrQ==",
      "license": "BSD-3-Clause",
      "engines": {
        "node": ">=0.10.0"
      }
    },
    "node_modules/superjson": {
      "version": "2.2.2",
      "resolved": "https://registry.npmjs.org/superjson/-/superjson-2.2.2.tgz",
      "integrity": "sha512-5JRxVqC8I8NuOUjzBbvVJAKNM8qoVuH0O77h4WInc/qC2q5IreqKxYwgkga3PfA22OayK2ikceb/B26dztPl+Q==",
      "license": "MIT",
      "dependencies": {
        "copy-anything": "^3.0.2"
      },
      "engines": {
        "node": ">=16"
      }
    },
    "node_modules/vite": {
      "version": "6.2.2",
      "resolved": "https://registry.npmjs.org/vite/-/vite-6.2.2.tgz",
      "integrity": "sha512-yW7PeMM+LkDzc7CgJuRLMW2Jz0FxMOsVJ8Lv3gpgW9WLcb9cTW+121UEr1hvmfR7w3SegR5ItvYyzVz1vxNJgQ==",
      "dev": true,
      "license": "MIT",
      "dependencies": {
        "esbuild": "^0.25.0",
        "postcss": "^8.5.3",
        "rollup": "^4.30.1"
      },
      "bin": {
        "vite": "bin/vite.js"
      },
      "engines": {
        "node": "^18.0.0 || ^20.0.0 || >=22.0.0"
      },
      "funding": {
        "url": "https://github.com/vitejs/vite?sponsor=1"
      },
      "optionalDependencies": {
        "fsevents": "~2.3.3"
      },
      "peerDependencies": {
        "@types/node": "^18.0.0 || ^20.0.0 || >=22.0.0",
        "jiti": ">=1.21.0",
        "less": "*",
        "lightningcss": "^1.21.0",
        "sass": "*",
        "sass-embedded": "*",
        "stylus": "*",
        "sugarss": "*",
        "terser": "^5.16.0",
        "tsx": "^4.8.1",
        "yaml": "^2.4.2"
      },
      "peerDependenciesMeta": {
        "@types/node": {
          "optional": true
        },
        "jiti": {
          "optional": true
        },
        "less": {
          "optional": true
        },
        "lightningcss": {
          "optional": true
        },
        "sass": {
          "optional": true
        },
        "sass-embedded": {
          "optional": true
        },
        "stylus": {
          "optional": true
        },
        "sugarss": {
          "optional": true
        },
        "terser": {
          "optional": true
        },
        "tsx": {
          "optional": true
        },
        "yaml": {
          "optional": true
        }
      }
    },
    "node_modules/vue": {
      "version": "3.5.13",
      "resolved": "https://registry.npmjs.org/vue/-/vue-3.5.13.tgz",
      "integrity": "sha512-wmeiSMxkZCSc+PM2w2VRsOYAZC8GdipNFRTsLSfodVqI9mbejKeXEGr8SckuLnrQPGe3oJN5c3K0vpoU9q/wCQ==",
      "license": "MIT",
      "dependencies": {
        "@vue/compiler-dom": "3.5.13",
        "@vue/compiler-sfc": "3.5.13",
        "@vue/runtime-dom": "3.5.13",
        "@vue/server-renderer": "3.5.13",
        "@vue/shared": "3.5.13"
      },
      "peerDependencies": {
        "typescript": "*"
      },
      "peerDependenciesMeta": {
        "typescript": {
          "optional": true
        }
      }
    },
    "node_modules/vue-router": {
      "version": "4.5.0",
      "resolved": "https://registry.npmjs.org/vue-router/-/vue-router-4.5.0.tgz",
      "integrity": "sha512-HDuk+PuH5monfNuY+ct49mNmkCRK4xJAV9Ts4z9UFc4rzdDnxQLyCMGGc8pKhZhHTVzfanpNwB/lwqevcBwI4w==",
      "license": "MIT",
      "dependencies": {
        "@vue/devtools-api": "^6.6.4"
      },
      "funding": {
        "url": "https://github.com/sponsors/posva"
      },
      "peerDependencies": {
        "vue": "^3.2.0"
      }
    },
    "node_modules/vue-router/node_modules/@vue/devtools-api": {
      "version": "6.6.4",
      "resolved": "https://registry.npmjs.org/@vue/devtools-api/-/devtools-api-6.6.4.tgz",
      "integrity": "sha512-sGhTPMuXqZ1rVOk32RylztWkfXTRhuS7vgAKv0zjqk8gbsHkJ7xfFf+jbySxt7tWObEJwyKaHMikV/WGDiQm8g==",
      "license": "MIT"
    }
  }
}

```


# vue-frontend\package.json
```text
{
  "name": "vue-frontend",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "axios": "^1.8.4",
    "marked": "^15.0.7",
    "pinia": "^3.0.1",
    "vue": "^3.5.13",
    "vue-router": "^4.5.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^5.2.1",
    "vite": "^6.2.0"
  }
}

```


# vue-frontend\vite.config.js
```text
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
})

```


# LICENSE
```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```


# README.md
```markdown
# EU-Compliant Document Chat System

A GDPR-compliant Retrieval-Augmented Generation (RAG) system designed for academic environments to securely query document collections.

![Simplified Architecture](docs/diagrams/architecture-diagram.svg)

## Key Features

- **EU Data Sovereignty**: All components comply with EU data protection regulations
- **Simple Document Management**: Add text files to a watched folder for automatic processing
- **Metadata Support**: Include bibliographic data for academic publications and other documents
- **Natural Language Querying**: Ask questions about your documents in natural language
- **Source Citations**: All answers include references to source documents
- **GDPR Compliance**: Built with privacy by design principles
- **Enhanced Security**:
  - Authentication for web interface
  - Request validation and sanitization
  - API key rotation mechanisms
  - Docker Secrets for credential management
  - Network isolation between components
  - Security headers via reverse proxy
  - Rate limiting and abuse prevention
  
## Technology Stack

- **Vector Database**: Weaviate (Netherlands-based)
- **LLM Provider**: Mistral AI (France-based)
- **Backend**: FastAPI (Python)
- **Frontend** Vue.js + Nginx
- **Deployment**: Docker containers on Hetzner (German cloud provider)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB of available RAM
- Mistral AI API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ducroq/doc-chat.git
   cd doc-chat
   ```

2. Create a `.env` file with your Mistral AI credentials:
   ```
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_MODEL=mistral-tiny
   MISTRAL_DAILY_TOKEN_BUDGET=10000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   ```

3. Set up your Mistral AI API key securely using Docker Secrets:
   ```bash
   mkdir -p ./secrets
   echo "your_api_key_here" > ./secrets/mistral_api_key.txt
   chmod 600 ./secrets/mistral_api_key.txt
   ```

4. Start the system:

   - On Windows
   ```bash
   .\start.ps1
   ```

   - On Linux
   ```bash
   chmod +x start.sh stop.sh
   ./start.sh
   ```
   
5. Access the interfaces:
   - Web interface: http://localhost:8081 (served by Nginx)
   - API documentation: http://localhost:8000/docs
   - Weaviate console: http://localhost:8080

### Adding Documents with Metadata

Simply place files in the `data/` directory. The system will automatically process and index them.

1. Place your `.txt` files in the `data/` directory
2. For each text file, create a corresponding metadata file with the same base name:
   ```
   data/
   example.txt
   example.metadata.json
   ```
3. Format the metadata file using a Zotero-inspired schema:
   ```json
   {
   "itemType": "journalArticle",
   "title": "Example Paper Title",
   "creators": [
      {"firstName": "John", "lastName": "Smith", "creatorType": "author"}
   ],
   "date": "2023",
   "publicationTitle": "Journal Name",
   "tags": ["tag1", "tag2"]
   }
   ```
   
The system will automatically associate metadata with documents and display it when providing answers.

### Authentication System

The system includes a secure authentication system:

- JWT-based authentication for API and web interfaces
- User management via command-line tool
- Bcrypt password hashing
- Role-based access control

To set up initial authentication after installation:

```bash
# Create a JWT secret key
openssl rand -hex 32 > ./secrets/jwt_secret_key.txt
chmod 600 ./secrets/jwt_secret_key.txt

# Create an admin user 
python manage_users.py create admin --generate-password --admin
```

For detailed information on authentication, see the [Authentication System Documentation](docs/security.md#authentication-system).

## Research & Analytics Features

- **Chat Logging**: Optional logging of interactions for research purposes
- **Privacy-First Design**: GDPR-compliant with anonymization and automatic data retention policies
- **Transparent Processing**: Clear user notifications when logging is enabled

## Documentation

For more detailed information about the system, check the following documentation:

- [Architecture Overview](docs/architecture.md)
- [Authentication](docs/authentication.md)
- [Deployment Guide](docs/deployment-guide.md)
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)
- [Security](docs/security.md)
- [Privacy Notice](docs/privacy-notice.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.



```


# docker-compose.yml
```yaml
secrets:
  mistral_api_key:
    file: ./secrets/mistral_api_key.txt
  internal_api_key:
    file: ./secrets/internal_api_key.txt
  jwt_secret_key:
    file: ./secrets/jwt_secret_key.txt

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.29.0
    networks:
      - backend
    command:
      - --host
      - 0.0.0.0
      - --port
      - '8080'
      - --scheme
      - http
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: on-failure

  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    networks:
      - backend
    environment:
      ENABLE_CUDA: '0'
    restart: on-failure

  processor:
    networks:
      - backend
    build: ./processor
    volumes:
      - ./data:/data
    environment:
      - WEAVIATE_URL=http://weaviate:8080
      - DATA_FOLDER=/data
    depends_on:
      - weaviate
      - t2v-transformers
    restart: on-failure
    user: "1000:1000"  # Use non-root user
    security_opt:
      - no-new-privileges:true

  api:
    networks:
      - frontend
      - backend
    secrets:
      - mistral_api_key
      - internal_api_key
      - jwt_secret_key
    build: 
      context: ./api
      dockerfile: Dockerfile
    ports:
      - 8000:8000
    environment:
      - WEAVIATE_URL=http://weaviate:8080
      - MISTRAL_MODEL=mistral-large-latest
      - MISTRAL_DAILY_TOKEN_BUDGET=100000
      - MISTRAL_MAX_REQUESTS_PER_MINUTE=30
      - MISTRAL_MAX_TOKENS_PER_REQUEST=5000
      - ENABLE_CHAT_LOGGING=true
      - ANONYMIZE_CHAT_LOGS=true
      - LOG_RETENTION_DAYS=30
      - CHAT_LOG_DIR=chat_data
      - MISTRAL_API_KEY_FILE=/run/secrets/mistral_api_key
      - INTERNAL_API_KEY_FILE=/run/secrets/internal_api_key
      - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret_key
      - LOG_DIR=/app/logs
    depends_on:
      - weaviate
      - t2v-transformers
    volumes:
      - ./chat_data:/app/chat_data
      - ./logs:/app/logs
      - ./users.json:/app/users.json
    user: "1000:1000"  # Use non-root user
    security_opt:
      - no-new-privileges:true

  vue-frontend:
    build:
      context: ./vue-frontend
      dockerfile: Dockerfile
    ports:
      - "8081:80"
    networks:
      - frontend
    secrets:
      - internal_api_key
    environment:
      - INTERNAL_API_KEY_FILE=/run/secrets/internal_api_key
      - ENABLE_CHAT_LOGGING=true
    depends_on:
      - api
    # user: "1000:1000"  # Use non-root user
    security_opt:
      - no-new-privileges:true

volumes:
  weaviate_data:


```


# manage_users.py
```python
# Save this as manage_users.py in the project root directory
import bcrypt
import json
import os
import argparse
from utils.utils import validate_password

USER_DB_FILE = "users.json"

def load_users():
    """Load users from the JSON file"""
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to the JSON file"""
    with open(USER_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def create_user(username, password, full_name=None, email=None, is_admin=False, force=False):
    """Create a new user or update an existing one"""
    # Validate password unless force is True
    if not force:
        is_valid, message = validate_password(password)
        if not is_valid:
            print(f"Password validation failed: {message}")
            print("Use --force to override password validation")
            return False
    
    users = load_users()
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    # Create or update user entry
    users[username] = {
        "username": username,
        "full_name": full_name or username,
        "email": email,
        "hashed_password": hashed_password,
        "disabled": False,
        "is_admin": is_admin
    }
    
    save_users(users)
    print(f"User '{username}' {'updated' if username in users else 'created'} successfully.")
    return True

def disable_user(username, disable=True):
    """Disable or enable a user"""
    users = load_users()
    
    if username not in users:
        print(f"User '{username}' does not exist!")
        return
    
    users[username]["disabled"] = disable
    save_users(users)
    
    status = "disabled" if disable else "enabled"
    print(f"User '{username}' {status} successfully.")

def delete_user(username):
    """Delete a user"""
    users = load_users()
    
    if username not in users:
        print(f"User '{username}' does not exist!")
        return
    
    del users[username]
    save_users(users)
    print(f"User '{username}' deleted successfully.")

def list_users():
    """List all users"""
    users = load_users()
    
    if not users:
        print("No users found.")
        return
    
    print("\nUser List:")
    print("=" * 60)
    print(f"{'Username':<15} {'Full Name':<20} {'Email':<20} {'Status':<10}")
    print("-" * 60)
    
    for username, user in users.items():
        status = "Disabled" if user.get("disabled", False) else "Active"
        if user.get("is_admin", False):
            status += " (Admin)"
        
        print(f"{username:<15} {user.get('full_name', ''):<20} {user.get('email', ''):<20} {status:<10}")
    
    print("=" * 60)

def generate_password():
    """Generate a strong password"""
    import random
    import string
    
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = '@#$%^&+=!'
    
    # Ensure at least one character from each set
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special)
    ]
    
    # Add remaining characters
    all_chars = lowercase + uppercase + digits + special
    password.extend(random.choice(all_chars) for _ in range(8))
    
    # Shuffle the password
    random.shuffle(password)
    
    # Convert to string
    return ''.join(password)

def main():
    parser = argparse.ArgumentParser(description="User management for EU-Compliant Document Chat")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Create user command
    create_parser = subparsers.add_parser("create", help="Create or update a user")
    create_parser.add_argument("username", help="Username")
    create_parser.add_argument("password", help="Password", nargs="?")
    create_parser.add_argument("--full-name", help="Full name")
    create_parser.add_argument("--email", help="Email address")
    create_parser.add_argument("--admin", action="store_true", help="Set as admin user")
    create_parser.add_argument("--force", action="store_true", help="Force creation even if password doesn't meet requirements")
    create_parser.add_argument("--generate-password", action="store_true", help="Generate a strong password")
    
    # List users command
    list_parser = subparsers.add_parser("list", help="List all users")
    
    # Disable user command
    disable_parser = subparsers.add_parser("disable", help="Disable a user")
    disable_parser.add_argument("username", help="Username")
    
    # Enable user command
    enable_parser = subparsers.add_parser("enable", help="Enable a user")
    enable_parser.add_argument("username", help="Username")
    
    # Delete user command
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("username", help="Username")
    
    # Reset password command
    reset_parser = subparsers.add_parser("reset-password", help="Reset a user's password")
    reset_parser.add_argument("username", help="Username")
    reset_parser.add_argument("password", help="New password", nargs="?")
    reset_parser.add_argument("--generate", action="store_true", help="Generate a strong password")
    reset_parser.add_argument("--force", action="store_true", help="Force reset even if password doesn't meet requirements")
    
    args = parser.parse_args()
    
    if args.command == "create":
        password = args.password
        if args.generate_password or (not password and args.password is None):
            password = generate_password()
            print(f"Generated password: {password}")
        elif not password:
            import getpass
            password = getpass.getpass("Enter password: ")
            confirmation = getpass.getpass("Confirm password: ")
            if password != confirmation:
                print("Passwords do not match!")
                return
        
        create_user(args.username, password, args.full_name, args.email, args.admin, args.force)
    
    elif args.command == "list":
        list_users()
    
    elif args.command == "disable":
        disable_user(args.username, True)
    
    elif args.command == "enable":
        disable_user(args.username, False)
    
    elif args.command == "delete":
        delete_user(args.username)
    
    elif args.command == "reset-password":
        users = load_users()
        if args.username not in users:
            print(f"User '{args.username}' does not exist!")
            return
        
        password = args.password
        if args.generate or (not password and args.password is None):
            password = generate_password()
            print(f"Generated password: {password}")
        elif not password:
            import getpass
            password = getpass.getpass("Enter new password: ")
            confirmation = getpass.getpass("Confirm new password: ")
            if password != confirmation:
                print("Passwords do not match!")
                return
        
        is_admin = users[args.username].get("is_admin", False)
        full_name = users[args.username].get("full_name")
        email = users[args.username].get("email")
        
        create_user(args.username, password, full_name, email, is_admin, args.force)
        print(f"Password reset for user '{args.username}'")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```


# start.ps1
```powershell
param ()

Write-Host "Starting EU-compliant RAG system..." -ForegroundColor Cyan

# Check if Docker is running first
try {
    $dockerStatus = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Docker is running." -ForegroundColor Green
    }
} catch {
    Write-Host "Error checking Docker status: $_" -ForegroundColor Red
    Write-Host "Please ensure Docker is installed and running, then try again." -ForegroundColor Red
    exit 1
}

# Start Weaviate and text vectorizer
Write-Host "Starting Weaviate and text vectorizer..." -ForegroundColor Yellow
docker-compose up -d weaviate t2v-transformers

# Wait for the text vectorizer to be ready first
Write-Host "Waiting for text vectorizer to be ready..." -ForegroundColor Yellow
$t2vReady = $false
$attempts = 0
$maxAttempts = 20

while (-not $t2vReady -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Checking text vectorizer... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        # Start a temporary container that has curl to check readiness
        $output = docker run --rm --network doc_chat_backend curlimages/curl -s http://t2v-transformers:8080/.well-known/ready
        if ($output -ne $null -or $LASTEXITCODE -eq 0) {
            $t2vReady = $true
            Write-Host "Text vectorizer is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Text vectorizer not ready yet: $_" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if (-not $t2vReady) {
    Write-Host "Text vectorizer did not become ready within the timeout period." -ForegroundColor Yellow
    Write-Host "Continuing anyway, but there might be initialization issues..." -ForegroundColor Yellow
}

# Now check if Weaviate is ready
Write-Host "Waiting for Weaviate to be ready..." -ForegroundColor Yellow
$weaviateReady = $false
$attempts = 0
$maxAttempts = 30

while (-not $weaviateReady -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Checking Weaviate... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        # Use a temporary container with curl to check readiness
        $output = docker run --rm --network doc_chat_backend curlimages/curl -s http://weaviate:8080/v1/.well-known/ready
        if ($output -ne $null -or $LASTEXITCODE -eq 0) {
            $weaviateReady = $true
            Write-Host "Weaviate is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Weaviate not ready yet: $_" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if (-not $weaviateReady) {
    Write-Host "Weaviate did not become ready within the timeout period." -ForegroundColor Red
    exit 1
}

# Allow additional time for Weaviate to fully initialize after reporting ready
Write-Host "Giving Weaviate extra time to fully initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the processor
Write-Host "Starting document processor..." -ForegroundColor Yellow
docker-compose up -d processor

# Wait for processor to initialize
Write-Host "Waiting for processor to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Start the API and frontend
Write-Host "Starting API and Vue.js frontend..." -ForegroundColor Yellow
docker-compose up -d api vue-frontend
$frontendUrl = "http://localhost:8081"

# Wait for API to initialize and verify connection to Weaviate
Write-Host "Waiting for API to connect to Weaviate..." -ForegroundColor Yellow
$apiReady = $false
$attempts = 0
$maxAttempts = 20

# First, get the API key from the file
$apiKeyPath = "./secrets/internal_api_key.txt" 
if (Test-Path $apiKeyPath) {
    $apiKey = Get-Content $apiKeyPath -Raw
    $apiKey = $apiKey.Trim()
    Write-Host "API key loaded from file" -ForegroundColor Gray
} else {
    Write-Host "Warning: API key file not found at $apiKeyPath" -ForegroundColor Yellow
    $apiKey = ""
}

while (-not $apiReady -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Checking API status... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        # Use the correct endpoint and include the API key header
        $headers = @{
            "X-API-Key" = $apiKey
        }
        
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/status" -Headers $headers -UseBasicParsing
        $status = $response.Content | ConvertFrom-Json
        
        if ($status.api -eq "running" -and $status.weaviate -eq "connected") {
            $apiReady = $true
            Write-Host "API successfully connected to Weaviate!" -ForegroundColor Green
        } else {
            Write-Host "  API not fully connected yet, waiting..." -ForegroundColor Gray
            Start-Sleep -Seconds 3
        }
    } catch {
        $errorMessage = $_.Exception.Message
        if ($errorMessage -match "429") { 
            Write-Host "  API rate limit exceeded. Waiting longer..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        }
        elseif ($errorMessage -match "403" -or $errorMessage -match "Invalid API key") {
            Write-Host "  API key authentication failed. Check your API key." -ForegroundColor Red
            Start-Sleep -Seconds 3
        }
        else {
            Write-Host "  API not ready yet: $errorMessage" -ForegroundColor Gray
            Start-Sleep -Seconds 3
        }
    }
}

if (-not $apiReady) {
    Write-Host "API did not connect to Weaviate properly within the timeout period." -ForegroundColor Yellow
    Write-Host "You may need to restart the API container: docker-compose restart api" -ForegroundColor Yellow
} else {
    Write-Host "All services started and connected successfully!" -ForegroundColor Green
}

# Display access information
Write-Host "Vue.js frontend: $frontendUrl" -ForegroundColor Cyan
Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "System statistics: http://localhost:8000/statistics" -ForegroundColor Cyan
Write-Host "Weaviate console: http://localhost:8080" -ForegroundColor Cyan
Write-Host "Processor logs: docker-compose logs -f processor" -ForegroundColor Cyan
Write-Host "Vue.js logs: docker-compose logs -f vue-frontend" -ForegroundColor Cyan
# Write-Host ""
# Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Cyan
```


# start.sh
```text
#!/bin/bash

echo "Starting EU-compliant RAG system..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
else
    echo "Docker is running."
fi

# Start Weaviate and text vectorizer
echo "Starting Weaviate and text vectorizer..."
docker-compose up -d weaviate t2v-transformers

# Wait for the text vectorizer to be ready
echo "Waiting for text vectorizer to be ready..."
t2v_ready=false
attempts=0
max_attempts=20

while [ "$t2v_ready" = false ] && [ $attempts -lt $max_attempts ]; do
    ((attempts++))
    echo "Checking text vectorizer... ($attempts/$max_attempts)"
    
    if docker run --rm --network doc_chat_backend curlimages/curl -s --fail http://t2v-transformers:8080/.well-known/ready > /dev/null 2>&1; then
        t2v_ready=true
        echo "Text vectorizer is ready!"
    else
        echo "  Text vectorizer not ready yet, waiting..."
        sleep 3
    fi
done

if [ "$t2v_ready" = false ]; then
    echo "Warning: Text vectorizer did not become ready within the timeout period."
    echo "Continuing anyway, but there might be initialization issues..."
fi

# Check if Weaviate is ready
echo "Waiting for Weaviate to be ready..."
weaviate_ready=false
attempts=0
max_attempts=30

while [ "$weaviate_ready" = false ] && [ $attempts -lt $max_attempts ]; do
    ((attempts++))
    echo "Checking Weaviate... ($attempts/$max_attempts)"
    
    if docker run --rm --network doc_chat_backend curlimages/curl -s --fail http://weaviate:8080/v1/.well-known/ready > /dev/null 2>&1; then
        weaviate_ready=true
        echo "Weaviate is ready!"
    else
        echo "  Weaviate not ready yet, waiting..."
        sleep 3
    fi
done

if [ "$weaviate_ready" = false ]; then
    echo "Error: Weaviate did not become ready within the timeout period."
    exit 1
fi

# Allow additional time for Weaviate to fully initialize
echo "Giving Weaviate extra time to fully initialize..."
sleep 5

# Start the processor
echo "Starting document processor..."
docker-compose up -d processor

# Wait for processor to initialize
echo "Waiting for processor to initialize..."
sleep 10

# Start the API and frontend
echo "Starting API and Vue.js frontend..."
docker-compose up -d api vue-frontend
frontend_url="http://localhost:8081"

# Wait for API to initialize and verify connection to Weaviate
echo "Waiting for API to connect to Weaviate..."
api_ready=false
attempts=0
max_attempts=20

# First, get the API key from the file
api_key_path="./secrets/internal_api_key.txt"
if [ -f "$api_key_path" ]; then
    api_key=$(cat "$api_key_path" | tr -d '\n\r')
    echo "API key loaded from file"
else
    echo "Warning: API key file not found at $api_key_path"
    api_key=""
fi

while [ "$api_ready" = false ] && [ $attempts -lt $max_attempts ]; do
    ((attempts++))
    echo "Checking API status... ($attempts/$max_attempts)"
    
    # Use curl with the API key header
    response=$(curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $api_key" http://localhost:8000/api/v1/status 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        # If we got a 200 response, check the actual content
        content=$(curl -s -H "X-API-Key: $api_key" http://localhost:8000/api/v1/status)
        if echo "$content" | grep -q '"api":"running"' && echo "$content" | grep -q '"weaviate":"connected"'; then
            api_ready=true
            echo "API successfully connected to Weaviate!"
        else
            echo "  API not fully connected yet, waiting..."
            sleep 3
        fi
    elif [ "$response" = "429" ]; then
        echo "  API rate limit exceeded. Waiting longer..."
        sleep 10  # Wait longer for rate limit to reset
    elif [ "$response" = "403" ]; then
        echo "  API key authentication failed. Check your API key."
        sleep 3
    else
        echo "  API not ready yet: Status code $response"
        sleep 3
    fi
done

if [ "$api_ready" = false ]; then
    echo "Warning: API did not connect to Weaviate properly within the timeout period."
    echo "You may need to restart the API container: docker-compose restart api"
else
    echo "All services started and connected successfully!"
fi

# Display access information
echo "Vue.js frontend: $frontend_url"
echo "API documentation: http://localhost:8000/docs"
echo "System statistics: http://localhost:8000/statistics"
echo "Weaviate console: http://localhost:8080"
echo "Processor logs: docker-compose logs -f processor"
echo "Vue.js logs: docker-compose logs -f vue-frontend"
echo ""
echo "Use ./stop.sh to stop all services."
```


# stop.ps1
```powershell
param ()

Write-Host "Starting graceful shutdown of EU-compliant RAG system..." -ForegroundColor Cyan

# Make a call to the API to flush logs
try {
    Write-Host "Requesting API to flush logs..." -ForegroundColor Yellow
    
    # Get the API key from the internal_api_key file
    $apiKeyPath = "./secrets/internal_api_key.txt"
    if (Test-Path $apiKeyPath) {
        $apiKey = Get-Content $apiKeyPath -Raw
        
        # Make the request with the API key header
        $headers = @{
            "X-API-Key" = $apiKey.Trim()
            "Content-Type" = "application/json"
        }
        
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/admin/flush-logs" -Method POST -Headers $headers
        Write-Host "Log flush result: $($response.status) - $($response.message)" -ForegroundColor Green
    } else {
        Write-Host "API key file not found at $apiKeyPath - skipping log flush" -ForegroundColor Yellow
    }
    
    # Small delay to allow the API time to process the request
    Start-Sleep -Seconds 2
} catch {
    Write-Host "Failed to request log flush: $_" -ForegroundColor Yellow
}

# Step 1: Stop the frontend and API first (they depend on other services)
Write-Host "Stopping frontend and API..." -ForegroundColor Yellow
docker-compose stop vue-frontend api
Write-Host "Frontend and API stopped." -ForegroundColor Green

# Step 2: Stop the processor 
Write-Host "Stopping document processor..." -ForegroundColor Yellow
docker-compose stop processor
Write-Host "Document processor stopped." -ForegroundColor Green

# Step 3: Stop Weaviate and text vectorizer 
Write-Host "Stopping Weaviate and text vectorizer..." -ForegroundColor Yellow
docker-compose stop weaviate t2v-transformers
Write-Host "Weaviate and text vectorizer stopped." -ForegroundColor Green

Write-Host "All services have been gracefully stopped." -ForegroundColor Green

# Fully remove everything at the end
Write-Host "Removing all containers..." -ForegroundColor Yellow
docker-compose down
Write-Host "All containers removed." -ForegroundColor Green
```


# stop.sh
```text
#!/bin/bash

echo "Starting graceful shutdown of EU-compliant RAG system..."

# Make a call to the API to flush logs
echo "Requesting API to flush logs..."
api_key_path="./secrets/internal_api_key.txt"
if [ -f "$api_key_path" ]; then
    api_key=$(cat "$api_key_path" | tr -d '\n\r')
    
    # Make the request with the API key header
    response=$(curl -s -X POST -H "X-API-Key: $api_key" -H "Content-Type: application/json" http://localhost:8000/api/v1/admin/flush-logs)
    
    if [ $? -eq 0 ]; then
        echo "Log flush completed: $response"
    else
        echo "Failed to flush logs: $response"
    fi
    
    # Small delay to allow the API time to process the request
    sleep 2
else
    echo "API key file not found at $api_key_path - skipping log flush"
fi

# Step 1: Stop the frontend and API first (they depend on other services)
echo "Stopping frontend and API..."
docker-compose stop vue-frontend api
echo "Frontend and API stopped."

# Step 2: Stop the processor
echo "Stopping document processor..."
docker-compose stop processor
echo "Document processor stopped."

# Step 3: Stop Weaviate and text vectorizer
echo "Stopping Weaviate and text vectorizer..."
docker-compose stop weaviate t2v-transformers
echo "Weaviate and text vectorizer stopped."

echo "All services have been gracefully stopped."

# Fully remove everything at the end
echo "Removing all containers..."
docker-compose down
echo "All containers removed."
```

