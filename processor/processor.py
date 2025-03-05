import os
import time
import logging
import uuid
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

def chunk_text(text, max_chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks of approximately max_chunk_size characters.
    """
    # Simple paragraph-based chunking
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_chunk_size,
        # save the current chunk and start a new one with overlap
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            
            # Find a good overlap point - ideally at a paragraph boundary
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

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
        """Process a text file and store chunks in Weaviate."""
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                logger.info(f"File content length: {len(content)} characters")
                
                # Get the filename without the path
                filename = os.path.basename(file_path)
                
                # Delete existing chunks for this file if any
                self.delete_existing_chunks(filename)
                
                # Split the content into chunks
                chunks = chunk_text(content)
                logger.info(f"Split into {len(chunks)} chunks")
                
                # Store each chunk in Weaviate
                for i, chunk_content in enumerate(chunks):
                    self.store_chunk(chunk_content, filename, i)
                    
                logger.info(f"File {filename} processed successfully")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")

    def delete_existing_chunks(self, filename):
        """Delete existing chunks for a file."""
        try:
            self.weaviate_client.batch.delete_objects(
                class_name="DocumentChunk",
                where={
                    "path": ["filename"],
                    "operator": "Equal",
                    "valueString": filename
                }
            )
            logger.info(f"Deleted existing chunks for {filename}")
        except Exception as e:
            logger.error(f"Error deleting existing chunks: {str(e)}")

    def store_chunk(self, content, filename, chunk_id):
        """Store a chunk in Weaviate."""
        try:
            properties = {
                "content": content,
                "filename": filename,
                "chunkId": chunk_id
            }
            
            # Create a UUID based on filename and chunk_id for consistency
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{chunk_id}"))
            
            self.weaviate_client.data_object.create(
                class_name="DocumentChunk",
                data_object=properties,
                uuid=obj_uuid
            )
            logger.info(f"Stored chunk {chunk_id} from {filename}")
        except Exception as e:
            logger.error(f"Error storing chunk: {str(e)}")

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
        client = weaviate.Client(
            weaviate_url,
            startup_period=60  # Increase this to 60 seconds, since Windows typically has slightly slower file operations through Docker:
        )        
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