import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import weaviate
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TextFileHandler(FileSystemEventHandler):
    def __init__(self, weaviate_client):
        self.weaviate_client = weaviate_client
    
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        logger.info(f"New text file detected: {event.src_path}")
        self.process_file(event.src_path)
        
    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        logger.info(f"Text file modified: {event.src_path}")
        self.process_file(event.src_path)
        
    def process_file(self, file_path):
        # Placeholder for the actual file processing
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"File content length: {len(content)} characters")
                # In a real implementation, you would:
                # 1. Split the content into chunks
                # 2. Store chunks in Weaviate
                logger.info(f"File {file_path} processed successfully")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

def setup_weaviate_schema(client):
    """Set up the Weaviate schema for document chunks."""
    logger.info("Setting up Weaviate schema")
    
    # Define the class for document chunks
    class_obj = {
        "class": "DocumentChunk",
        "vectorizer": "text2vec-transformers",
        "moduleConfig": {
            "text2vec-transformers": {
                "vectorizeClassName": False
            }
        },
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": False,
                        "vectorizePropertyName": False
                    }
                }
            },
            {
                "name": "filename",
                "dataType": ["string"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True,
                        "vectorizePropertyName": False
                    }
                }
            },
            {
                "name": "chunkId",
                "dataType": ["int"],
                "moduleConfig": {
                    "text2vec-transformers": {
                        "skip": True,
                        "vectorizePropertyName": False
                    }
                }
            }
        ]
    }
    
    # Check if the class already exists
    try:
        schema = client.schema.get()
        existing_classes = [c['class'] for c in schema['classes']] if 'classes' in schema else []
        
        if "DocumentChunk" not in existing_classes:
            client.schema.create_class(class_obj)
            logger.info("DocumentChunk class created in Weaviate")
        else:
            logger.info("DocumentChunk class already exists in Weaviate")
    except Exception as e:
        logger.error(f"Error setting up Weaviate schema: {str(e)}")

def main():
    # Connect to Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    try:
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        client = weaviate.Client(weaviate_url)
        
        # Check connection
        if client.is_ready():
            logger.info("Successfully connected to Weaviate")
            
            # Setup schema
            setup_weaviate_schema(client)
            
            # Set up file watcher
            data_folder = os.getenv("DATA_FOLDER", "/data")
            logger.info(f"Starting to watch folder: {data_folder}")
            
            event_handler = TextFileHandler(client)
            observer = Observer()
            observer.schedule(event_handler, data_folder, recursive=False)
            observer.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        else:
            logger.error("Failed to connect to Weaviate")
    except Exception as e:
        logger.error(f"Error initializing the processor: {str(e)}")

if __name__ == "__main__":
    main()