"""
Main entry point for the document processor application.
"""
import os
import time
import asyncio
import signal

from config import settings
from utils.logging_config import setup_logging, get_logger
from utils.errors import WeaviateConnectionError, FileProcessingError
from storage.weaviate_client import connect_with_retry, setup_schema
from storage.document_storage import DocumentStorage
from core.processor_tracker import ProcessingTracker
from core.document_processor import DocumentProcessor

# Set up logging
logger = setup_logging()

# Flag to control graceful shutdown
running = True

async def process_documents():
    """
    Main document processing function.
    Connects to Weaviate, sets up schema, and processes documents.
    """
    try:
        # Create Weaviate client with retry
        logger.info(f"Connecting to Weaviate at {settings.WEAVIATE_URL}")
        client = await connect_with_retry(settings.WEAVIATE_URL)
        
        # Setup schema
        logger.info("Setting up Weaviate schema")
        schema_success = setup_schema(client)
        if not schema_success:
            logger.error("Failed to set up Weaviate schema")
            return
        
        # Create storage and tracker instances
        storage = DocumentStorage(client)
        tracker = ProcessingTracker(settings.TRACKER_FILE, settings.DATA_FOLDER)
        
        # Create document processor
        processor = DocumentProcessor(
            storage=storage,
            tracker=tracker,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            chunking_strategy=settings.CHUNKING_STRATEGY,
            process_subfolders=settings.PROCESS_SUBFOLDERS
        )
        
        # Process all documents
        logger.info(f"Starting document processing for data folder: {settings.DATA_FOLDER}")
        stats = await processor.process_all_documents(settings.DATA_FOLDER)
        
        # Log processing results
        logger.info("Document processing summary:")
        for key, value in stats.items():
            if key == "duration" and value is not None:
                logger.info(f"  {key}: {value:.2f}s")
            else:
                logger.info(f"  {key}: {value}")
        
        # Get document count
        doc_count = await storage.get_document_count()
        logger.info(f"Total documents in database: {doc_count}")
        
        # Display information about staying alive
        logger.info("Document processing complete. Service will remain running.")
        logger.info("No file watching is active. To process new files, restart the container.")
        
        # Keep the container running
        while running:
            await asyncio.sleep(60)  # Check every minute if we should still be running
    
    except WeaviateConnectionError as e:
        logger.error(f"Weaviate connection error: {str(e)}")
    except FileProcessingError as e:
        logger.error(f"File processing error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Shutting down processor")
        if 'client' in locals():
            client.close()

def handle_sigterm(signum, frame):
    """
    Handle SIGTERM signal for graceful shutdown.
    """
    global running
    logger.info("Received shutdown signal, exiting gracefully...")
    running = False

async def main():
    """
    Main entry point.
    """
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    # Log settings
    logger.info("Starting document processor with the following settings:")
    for key, value in settings.as_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Run document processing
    await process_documents()

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())