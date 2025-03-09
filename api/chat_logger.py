import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class ChatLogger:
    """
    Privacy-compliant chat logger for research purposes.
    Implements GDPR requirements including opt-in logging, log rotation,
    anonymization, and deletion capabilities.
    
    Logs are saved to a local folder within the project.
    """
    
    def __init__(self, log_dir: str = "chat_data"):
        """
        Initialize the chat logger with privacy controls.
        
        Args:
            log_dir: Directory where logs will be stored
        """
        self.log_dir = Path(log_dir)
        self.enabled = os.getenv("ENABLE_CHAT_LOGGING", "false").lower() == "true"
        self.anonymize = os.getenv("ANONYMIZE_CHAT_LOGS", "true").lower() == "true"
        self.retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
        
        if self.enabled:
            self.log_dir.mkdir(exist_ok=True, parents=True)
            self.log_file = self.log_dir / f"chat_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
            logger.info(f"Chat logging enabled. Logs will be saved to {self.log_file}")
            logger.info(f"Log anonymization: {self.anonymize}, Retention period: {self.retention_days} days")
            
            # Run initial rotation to clean up old logs
            self._rotate_logs()
        else:
            logger.info("Chat logging is disabled")
    
    def log_interaction(self, query: str, response: Dict[str, Any], 
                        request_id: Optional[str] = None, 
                        user_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
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
        """
        if not self.enabled:
            return False
            
        # Apply anonymization if enabled
        if self.anonymize and user_id:
            # Create a deterministic but anonymized ID
            anon_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user_id))
            user_id = f"anon_{anon_id[-12:]}"
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id or str(uuid.uuid4()),
            "user_id": user_id,  # Will be None if not provided or anonymized if enabled
            "query": query,
            "response": {
                "answer": response.get("answer"),
                "sources": self._anonymize_sources(response.get("sources", []))
            }
        }
        
        # Add metadata if provided
        if metadata:
            log_entry["metadata"] = metadata
        
        try:
            # Make sure log directory exists
            self.log_dir.mkdir(exist_ok=True, parents=True)
            
            # Write to log file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Handle log rotation if needed
            self._rotate_logs()
            return True
        except Exception as e:
            logger.error(f"Error logging chat interaction: {str(e)}")
            return False

    def _anonymize_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Anonymize potentially sensitive information in document sources.
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
            
            anonymized_sources.append(anon_source)
            
        return anonymized_sources

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
            for log_file in self.log_dir.glob("chat_log_*.jsonl"):
                try:
                    # Extract date from filename
                    date_str = log_file.stem.replace("chat_log_", "")
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    # Delete if older than retention period
                    if file_date < cutoff_date:
                        logger.info(f"Deleting old log file: {log_file}")
                        log_file.unlink()
                except (ValueError, OSError) as e:
                    logger.warning(f"Error processing log file {log_file}: {str(e)}")
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
            
        success = True
        
        try:
            # Process each log file
            for log_file in self.log_dir.glob("chat_log_*.jsonl"):
                try:
                    # Create a temporary file
                    temp_file = log_file.with_suffix(".tmp")
                    
                    # If anonymization is enabled, calculate the anonymized ID
                    target_id = user_id
                    if self.anonymize:
                        anon_id = str(uuid.uuid5(uuid.NAMESPACE_OID, user_id))
                        target_id = f"anon_{anon_id[-12:]}"
                    
                    # Filter out entries for this user
                    with open(log_file, 'r', encoding='utf-8') as input_file, \
                         open(temp_file, 'w', encoding='utf-8') as output_file:
                        for line in input_file:
                            try:
                                entry = json.loads(line.strip())
                                if entry.get("user_id") != target_id:
                                    output_file.write(line)
                            except json.JSONDecodeError:
                                # Keep lines that can't be parsed
                                output_file.write(line)
                    
                    # Replace original with filtered version
                    temp_file.replace(log_file)
                    
                except Exception as e:
                    logger.error(f"Error processing file {log_file} during user data deletion: {str(e)}")
                    success = False
            
            logger.info(f"Completed deletion of user data for user ID: {user_id}")
            return success
            
        except Exception as e:
            logger.error(f"Error during user data deletion: {str(e)}")
            return False
            
    def get_log_files(self, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> List[Path]:
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
        
        return matching_files