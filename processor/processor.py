import os
import time
import logging
import uuid
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import weaviate
from weaviate.config import AdditionalConfig, Timeout
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

    def validate_and_get_encoding(self, file_path):
        """
        Validate that a file is a readable text file and return the correct encoding.
        Returns (is_valid, encoding or error_message)
        """
        # Expanded list of encodings to try
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Try to read with each encoding
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    # Try to read a sample to verify encoding works
                    sample = file.read(100)
                    return True, encoding
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return False, f"Error reading file: {str(e)}"
        
        return False, "Could not decode with any supported encoding"        
        
    def process_file(self, file_path):
        """Process a text file and store chunks in Weaviate."""
        logger.info(f"Processing file: {file_path}")
        
        # Validate file and get encoding
        is_valid, result = self.validate_and_get_encoding(file_path)
        
        if not is_valid:
            logger.error(f"Error processing file {file_path}: {result}")
            return
        
        # result is the encoding if valid
        encoding = result
        logger.info(f"File validated, using encoding: {encoding}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as file:
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
            collection = self.weaviate_client.collections.get("DocumentChunk")
            
            # Create a proper filter object using the where filter builder
            where_filter = weaviate.classes.query.Filter.by_property("filename").equal(filename)
            
            # Delete objects with the proper filter syntax
            result = collection.data.delete_many(
                where=where_filter
            )
            
            if result and hasattr(result, 'successful'):
                logger.info(f"Deleted {result.successful} existing chunks for {filename}")
            else:
                logger.info(f"No existing chunks found for {filename}")
                
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

            # Get the DocumentChunk collection
            collection = self.weaviate_client.collections.get("DocumentChunk")
            
            # First, try to delete the object if it exists
            try:
                collection.data.delete_by_id(obj_uuid)
                logger.info(f"Deleted existing object with ID {obj_uuid}")
            except Exception as delete_error:
                # It's okay if the object doesn't exist yet
                logger.debug(f"Object with ID {obj_uuid} not found for deletion: {str(delete_error)}")
            
            # Now insert the object
            collection.data.insert(
                properties=properties,
                uuid=obj_uuid
            )
            logger.info(f"Stored chunk {chunk_id} from {filename}")            
            logger.info(f"Stored chunk {chunk_id} from {filename}")
        except Exception as e:
            logger.error(f"Error storing chunk: {str(e)}")
    
    def process_existing_files(self, data_folder):
        """Process all existing text files in the data folder."""
        logger.info(f"Scanning for existing files in {data_folder}")
        text_files = glob.glob(os.path.join(data_folder, "*.txt"))
        logger.info(f"Found {len(text_files)} existing text files")
        
        for file_path in text_files:
            logger.info(f"Processing existing file: {file_path}")
            self.process_file(file_path)

def setup_weaviate_schema(client):
    """Set up the Weaviate schema for document chunks."""
    logger.info("Setting up Weaviate schema")
    
    # Check if the collection already exists
    try:
        if not client.collections.exists("DocumentChunk"):
            # Collection doesn't exist, create it
            collection = client.collections.create(
                name="DocumentChunk",
                vectorizer_config=weaviate.config.Configure.Vectorizer.text2vec_transformers(
                    vectorize_collection_name=False
                ),
                properties=[
                    weaviate.config.Property(
                        name="content",
                        data_type=weaviate.config.DataType.TEXT,
                        vectorize_property_name=False,
                        skip_vectorization=False
                    ),
                    weaviate.config.Property(
                        name="filename",
                        data_type=weaviate.config.DataType.TEXT,
                        vectorize_property_name=False,
                        skip_vectorization=True
                    ),
                    weaviate.config.Property(
                        name="chunkId",
                        data_type=weaviate.config.DataType.INT,
                        vectorize_property_name=False,
                        skip_vectorization=True
                    )
                ]
            )
            logger.info("DocumentChunk collection created in Weaviate")
        else:
            logger.info("DocumentChunk collection already exists in Weaviate")
    except Exception as e:
        logger.error(f"Error setting up Weaviate schema: {str(e)}")
            
def connect_with_retry(weaviate_url, max_retries=10, retry_delay=5):
    """Connect to Weaviate with retry mechanism."""
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
        except Exception as e:
            last_exception = e
            logger.warning(f"Connection attempt {retries+1} failed: {str(e)}")
        
        # Wait before retry
        logger.info(f"Waiting {retry_delay} seconds before retry...")
        time.sleep(retry_delay)
        retries += 1
    
    # If we get here, all retries failed
    raise Exception(f"Failed to connect to Weaviate after {max_retries} attempts. Last error: {str(last_exception)}")

def main():
    # Connect to Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    try:
        logger.info(f"Connecting to Weaviate at {weaviate_url}")

        client = connect_with_retry(weaviate_url)

        logger.info("Successfully connected to Weaviate")
        
        # Setup schema
        setup_weaviate_schema(client)             
            
        # Set up file watcher
        data_folder = os.getenv("DATA_FOLDER", "/data")
        logger.info(f"Starting to watch folder: {data_folder}")
        
        event_handler = TextFileHandler(client)
        
        # Process existing files first
        logger.info("Processing existing files in data folder")
        event_handler.process_existing_files(data_folder)
        
        # Then set up watchdog for future changes
        observer = Observer()
        observer.schedule(event_handler, data_folder, recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    except Exception as e:
        logger.error(f"Error initializing the processor: {str(e)}")

if __name__ == "__main__":
    main()