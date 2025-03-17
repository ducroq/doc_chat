import os
import json
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Set, Tuple
from pydantic import BaseModel, field_validator

# Configure module logger
logger = logging.getLogger(__name__)


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


class ChatLogger:
    """
    Privacy-compliant chat logger for research purposes.
    Implements GDPR requirements including opt-in logging, log rotation,
    anonymization, and deletion capabilities.
    
    Logs are saved to a local folder within the project with configurable
    retention periods and privacy controls.
    """
    
    # Class constants
    DEFAULT_LOG_DIR = "chat_data"
    DEFAULT_RETENTION_DAYS = 30
    DEFAULT_BUFFER_SIZE = 10
    ANONYMIZE_PREFIX = "anon_"
    
    def __init__(
        self, 
        log_dir: str = DEFAULT_LOG_DIR,
        buffer_size: int = DEFAULT_BUFFER_SIZE
    ):
        """
        Initialize the chat logger with privacy controls.
        
        Args:
            log_dir: Directory where logs will be stored
            buffer_size: Number of log entries to buffer before writing to disk
        """
        self.log_dir = Path(log_dir)
        self.enabled = os.getenv("ENABLE_CHAT_LOGGING", "false").lower() == "true"
        self.anonymize = os.getenv("ANONYMIZE_CHAT_LOGS", "true").lower() == "true"
        self.retention_days = int(os.getenv("LOG_RETENTION_DAYS", str(self.DEFAULT_RETENTION_DAYS)))
        self.buffer_size = buffer_size
        self.log_buffer: List[str] = []
        
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
            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                request_id=request_id or str(uuid.uuid4()),
                user_id=anonymized_user_id if self.anonymize and user_id else user_id,
                query=query,
                response={
                    "answer": response.get("answer"),
                    "sources": self._anonymize_sources(response.get("sources", []))
                },
                metadata=metadata
            )
            
            return self._write_log_entry(log_entry.model_dump())  # Using model_dump() instead of dict()
        except ValueError as e:
            logger.error(f"Error creating log entry: {str(e)}")
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
        if self.enabled and self.log_buffer:
            self._flush_buffer()
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