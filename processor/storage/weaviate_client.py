"""
Weaviate client for the document processor.
Handles connection and schema setup.
"""
import time
import asyncio
from typing import Optional

import weaviate
from weaviate.config import AdditionalConfig, Timeout

from config import settings
from utils.logging_config import get_logger
from utils.errors import WeaviateConnectionError

logger = get_logger(__name__)

async def connect_with_retry(
    weaviate_url: str = settings.WEAVIATE_URL,
    max_retries: int = settings.MAX_RETRIES,
    retry_delay: int = settings.RETRY_DELAY
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
        WeaviateConnectionError: If connection fails after max retries
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
    raise WeaviateConnectionError(error_message, {
        "weaviate_url": weaviate_url,
        "max_retries": max_retries,
        "original_error": str(last_exception)
    })

def setup_schema(client: weaviate.Client) -> bool:
    """
    Set up the Weaviate schema for document chunks.
    
    Args:
        client: Connected Weaviate client
        
    Returns:
        bool: True if setup was successful, False otherwise
        
    Raises:
        WeaviateConnectionError: If schema setup fails
    """
    try:
        from weaviate.classes.config import Configure, DataType
        
        start_time = time.time()
        logger.info("Setting up Weaviate schema for document chunks")
        
        # Check if the collection already exists
        if not client.collections.exists("DocumentChunk"):
            logger.info("DocumentChunk collection does not exist, creating new collection")
            creation_start = time.time()
            
            # Collection doesn't exist, create it            
            client.collections.create(
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
        error_message = f"Error setting up Weaviate schema: {str(e)}"
        logger.error(error_message)
        import traceback
        logger.error(traceback.format_exc())
        raise WeaviateConnectionError(error_message, {"original_error": str(e)})