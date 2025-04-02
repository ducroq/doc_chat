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
    