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
    
    # Logging
    LOG_DIR: str = "logs"
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"        
    
    class Config:
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