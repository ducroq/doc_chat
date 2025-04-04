"""
Processor tracker for tracking which files have been processed.
"""
import os
import json
import time
from typing import Dict, Any, List, Optional, Set

from config import settings
from utils.logging_config import get_logger
from utils.errors import TrackerError

logger = get_logger(__name__)

class ProcessingTracker:
    """
    Tracks files that have been processed to avoid redundant processing.
    
    Attributes:
        tracker_file_path: Path to the JSON file storing processing records
        processed_files: Dictionary of processed files with metadata
        data_folder: Base folder for documents
    """
    
    def __init__(self, tracker_file_path: Optional[str] = None, data_folder: Optional[str] = None):
        """
        Initialize a tracker that keeps record of processed files.
        
        Args:
            tracker_file_path: Path to the JSON file storing processing records
            data_folder: Base folder for documents to calculate relative paths
        """
        self.tracker_file_path = tracker_file_path or settings.TRACKER_FILE
        self.data_folder = data_folder or settings.DATA_FOLDER
        logger.info(f"Initializing file processing tracker at {self.tracker_file_path}")
        
        # Create tracker directory if it doesn't exist
        tracker_dir = os.path.dirname(self.tracker_file_path)
        if tracker_dir and not os.path.exists(tracker_dir):
            try:
                os.makedirs(tracker_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create tracker directory: {str(e)}")
        
        self.processed_files = self._load_tracker()
        logger.info(f"Tracker initialized with {len(self.processed_files)} previously processed files")

    def _load_tracker(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the tracker file or create it if it doesn't exist.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of processed files with metadata
            
        Raises:
            TrackerError: If the tracker file cannot be loaded
        """
        if os.path.exists(self.tracker_file_path):
            try:
                logger.info(f"Loading existing tracker file from {self.tracker_file_path}")
                with open(self.tracker_file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded tracker with {len(data)} records")
                    return data
            except json.JSONDecodeError as e:
                error_msg = f"Error decoding tracker file JSON: {str(e)}"
                logger.error(error_msg)
                raise TrackerError(error_msg, {"file_path": self.tracker_file_path})
            except PermissionError as e:
                error_msg = f"Permission error reading tracker file: {str(e)}"
                logger.error(error_msg)
                raise TrackerError(error_msg, {"file_path": self.tracker_file_path})
            except Exception as e:
                error_msg = f"Error loading tracker file: {str(e)}"
                logger.error(error_msg)
                raise TrackerError(error_msg, {"file_path": self.tracker_file_path})
                
        logger.info("No existing tracker file found, starting with empty tracking")
        return {}

    def _save_tracker(self) -> bool:
        """
        Save the tracker data to file.
        
        Returns:
            bool: Whether the save was successful
            
        Raises:
            TrackerError: If the tracker file cannot be saved
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracker_file_path) or '.', exist_ok=True)
            
            with open(self.tracker_file_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
                
            logger.debug(f"Saved processing tracker to {self.tracker_file_path}")
            return True
        except PermissionError as e:
            error_msg = f"Permission error saving tracker file: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg, {"file_path": self.tracker_file_path})
        except Exception as e:
            error_msg = f"Error saving tracker file: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg, {"file_path": self.tracker_file_path})
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on modification time.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file is new or modified since last processing
            
        Raises:
            TrackerError: If the file status cannot be determined
        """
        try:
            file_mod_time = os.path.getmtime(file_path)
            # Use relative path as key instead of just filename
            file_key = self._get_file_key(file_path)
            
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
            error_msg = f"Error checking file status: {str(e)}"
            logger.error(error_msg)
            # If in doubt, process the file
            return True

    def mark_as_processed(self, file_path: str) -> bool:
        """
        Mark a file as processed with current timestamps.
        
        Args:
            file_path: Path to the file to mark as processed
            
        Returns:
            bool: Whether marking was successful
            
        Raises:
            TrackerError: If the file cannot be marked as processed
        """
        try:
            file_mod_time = os.path.getmtime(file_path)
            file_key = self._get_file_key(file_path)
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
            error_msg = f"Error marking file as processed: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg, {"file_path": file_path})
    
    def remove_file(self, file_path: str) -> bool:
        """
        Remove a file from the tracking record.
        
        Args:
            file_path: Path to the file to remove from tracking
            
        Returns:
            bool: Whether removal was successful
            
        Raises:
            TrackerError: If the file cannot be removed from tracking
        """
        try:
            file_key = self._get_file_key(file_path)
            if file_key in self.processed_files:
                logger.info(f"Removing {file_key} from processing tracker")
                del self.processed_files[file_key]
                self._save_tracker()
                return True
            logger.debug(f"File {file_key} not found in tracker")
            return False
        except Exception as e:
            error_msg = f"Error removing file from tracker: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg, {"file_path": file_path})
    
    def get_all_tracked_files(self) -> Set[str]:
        """
        Return a set of all tracked file paths (absolute).
        
        Returns:
            Set[str]: Set of tracked absolute file paths
            
        Raises:
            TrackerError: If the tracked files cannot be retrieved
        """
        try:
            # Convert relative paths back to absolute paths
            tracked_files = set()
            
            for file_key, file_data in self.processed_files.items():
                # If we have a stored path, use it
                if 'path' in file_data and os.path.isabs(file_data['path']):
                    tracked_files.add(file_data['path'])
                else:
                    # Otherwise reconstruct from data folder and relative path
                    abs_path = os.path.normpath(os.path.join(self.data_folder, file_key))
                    tracked_files.add(abs_path)
            
            return tracked_files
        except Exception as e:
            error_msg = f"Error getting tracked files: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg)
    
    def clear_tracking_data(self) -> bool:
        """
        Clear all tracking data.
        
        Returns:
            bool: Whether clearing was successful
            
        Raises:
            TrackerError: If the tracking data cannot be cleared
        """
        try:
            self.processed_files = {}
            self._save_tracker()
            logger.info("Cleared all file tracking data")
            return True
        except Exception as e:
            error_msg = f"Error clearing tracking data: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg)
    
    def get_tracking_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tracked files.
        
        Returns:
            Dict[str, Any]: Statistics about tracked files
            
        Raises:
            TrackerError: If the tracking statistics cannot be retrieved
        """
        try:
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
        except Exception as e:
            error_msg = f"Error getting tracking stats: {str(e)}"
            logger.error(error_msg)
            raise TrackerError(error_msg)

    def _get_file_key(self, file_path: str) -> str:
        """
        Generate a unique key for a file based on its relative path from data_folder.
        
        Args:
            file_path: Absolute path to the file
            
        Returns:
            str: Relative path to use as a key
        """
        abs_data_folder = os.path.abspath(self.data_folder)
        abs_file_path = os.path.abspath(file_path)
        
        # Make sure file is within data folder
        if abs_file_path.startswith(abs_data_folder):
            rel_path = os.path.relpath(abs_file_path, abs_data_folder)
            return rel_path
        else:
            # Fallback to basename if the file is not within data folder
            logger.warning(f"File {file_path} is not within data folder {self.data_folder}")
            return os.path.basename(file_path)