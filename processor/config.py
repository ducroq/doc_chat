import os
import secrets
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from typing import ClassVar

from logging import getLogger

logger = getLogger(__name__)

class ChunkingStrategy(str, Enum):
    """Enum for different chunking strategies."""
    SIMPLE = "simple"           # Simple character-based chunking
    PARAGRAPH = "paragraph"     # Paragraph-based chunking
    SECTION = "section"         # Section-based (using headings)
    SEMANTIC = "semantic"       # Semantic chunking using AI

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    # Basic settings
    DEBUG: bool = False
    
    # Weaviate connection
    WEAVIATE_URL: str = "http://weaviate:8080"
    
    # Document processing
    DATA_FOLDER: str = "/data"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNKING_STRATEGY: ChunkingStrategy = ChunkingStrategy.SECTION
    
    # Performance settings
    MAX_RETRIES: int = 10
    RETRY_DELAY: int = 5
    MAX_WORKER_THREADS: int = 5
    BATCH_SIZE: int = 10
    
    # File processing
    FILE_EXTENSIONS: List[str] = [".md", ".txt"]
    PROCESS_SUBFOLDERS: bool = True
    TRACKER_FILE: str = ""  # We'll set this after initialization
    
    # Logging
    LOG_DIR: str = "logs"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    
    model_config = {
        "env_prefix": "",  # No prefix for environment variables
        "extra": "ignore"  # Ignore extra attributes
    }
    
    def __str__(self) -> str:
        """Return string representation of settings for logging."""
        settings_dict = {k: v for k, v in self.__dict__.items() 
                         if not k.startswith('_') and k.isupper()}
        return "\n".join(f"{k}={v}" for k, v in settings_dict.items())
    
    def as_dict(self) -> Dict[str, Any]:
        """Return settings as a dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k.isupper()}
    
    def model_post_init(self, __context: Any) -> None:
        """Post initialization hook to set derived values."""
        # Set values that depend on other settings
        self.TRACKER_FILE = os.path.join(self.DATA_FOLDER, ".processed_files.json")
        
        # Convert environment variables
        if os.environ.get("DEBUG"):
            self.DEBUG = os.environ.get("DEBUG", "false").lower() in ["true", "1", "yes", "t"]
        
        if os.environ.get("WEAVIATE_URL"):
            self.WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
            
        if os.environ.get("DATA_FOLDER"):
            self.DATA_FOLDER = os.environ.get("DATA_FOLDER")
            # Update tracker file location
            self.TRACKER_FILE = os.path.join(self.DATA_FOLDER, ".processed_files.json")
            
        if os.environ.get("CHUNK_SIZE"):
            self.CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
            
        if os.environ.get("CHUNK_OVERLAP"):
            self.CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
            
        if os.environ.get("CHUNKING_STRATEGY"):
            self.CHUNKING_STRATEGY = ChunkingStrategy(os.environ.get("CHUNKING_STRATEGY", "section"))
            
        if os.environ.get("MAX_RETRIES"):
            self.MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "10"))
            
        if os.environ.get("RETRY_DELAY"):
            self.RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "5"))
            
        if os.environ.get("MAX_WORKER_THREADS"):
            self.MAX_WORKER_THREADS = int(os.environ.get("MAX_WORKER_THREADS", "5"))
            
        if os.environ.get("BATCH_SIZE"):
            self.BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))
            
        if os.environ.get("FILE_EXTENSIONS"):
            self.FILE_EXTENSIONS = os.environ.get("FILE_EXTENSIONS", ".md,.txt").split(",")
            
        if os.environ.get("PROCESS_SUBFOLDERS"):
            self.PROCESS_SUBFOLDERS = os.environ.get("PROCESS_SUBFOLDERS", "true").lower() in ["true", "1", "yes", "t"]
            
        if os.environ.get("LOG_DIR"):
            self.LOG_DIR = os.environ.get("LOG_DIR", "logs")
            
        if os.environ.get("LOG_FORMAT"):
            self.LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Create settings instance
settings = Settings()