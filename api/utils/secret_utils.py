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
