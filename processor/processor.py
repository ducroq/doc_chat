import os
import json
import time
import logging
import uuid
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProcessingTracker:
    def __init__(self, tracker_file_path="processed_files.json"):
        self.tracker_file_path = tracker_file_path
        self.processed_files = self._load_tracker()
    
    def _load_tracker(self):
        """Load the tracker file or create it if it doesn't exist."""
        if os.path.exists(self.tracker_file_path):
            try:
                with open(self.tracker_file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading tracker file: {str(e)}")
                return {}
        return {}
    
    def _save_tracker(self):
        """Save the tracker data to file."""
        try:
            with open(self.tracker_file_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tracker file: {str(e)}")
    
    def should_process_file(self, file_path):
        """
        Determine if a file should be processed based on modification time.
        Returns True if file is new or modified since last processing.
        """
        try:
            file_mod_time = os.path.getmtime(file_path)
            file_key = os.path.basename(file_path)
            
            # If file not in tracker or has been modified, process it
            if (file_key not in self.processed_files or 
                file_mod_time > self.processed_files[file_key]['last_modified']):
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking file status: {str(e)}")
            # If in doubt, process the file
            return True
    
    def mark_as_processed(self, file_path):
        """Mark a file as processed with current timestamps."""
        try:
            file_mod_time = os.path.getmtime(file_path)
            file_key = os.path.basename(file_path)
            
            self.processed_files[file_key] = {
                'path': file_path,
                'last_modified': file_mod_time,
                'last_processed': time.time()
            }
            self._save_tracker()
        except Exception as e:
            logger.error(f"Error marking file as processed: {str(e)}")

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
            # If you're using a tracker, update it to remove the file

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

    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        logger.info(f"Text file deleted: {event.src_path}")
        
        # Get the filename without the path
        filename = os.path.basename(event.src_path)
        
        # Delete the chunks from Weaviate
        self.delete_existing_chunks(filename)
        
        # If you're using a tracker, update it to remove the file
        tracker_path = os.path.join(os.path.dirname(event.src_path), ".processed_files.json")
        if os.path.exists(tracker_path):
            try:
                with open(tracker_path, 'r') as f:
                    tracker_data = json.load(f)
                
                # Remove the file from the tracker if it exists
                if filename in tracker_data:
                    del tracker_data[filename]
                    
                    # Save the updated tracker
                    with open(tracker_path, 'w') as f:
                        json.dump(tracker_data, f, indent=2)
                    
                    logger.info(f"Removed {filename} from processing tracker")
            except Exception as e:
                logger.error(f"Error updating tracker after file deletion: {str(e)}")
                
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
            # Get the collection
            collection = self.weaviate_client.collections.get("DocumentChunk")
            
            # Create a proper filter for Weaviate 4.x
            from weaviate.classes.query import Filter
            where_filter = Filter.by_property("filename").equal(filename)
            
            # Delete using the filter
            result = collection.data.delete_many(
                where=where_filter
            )
            
            # Log the result
            if hasattr(result, 'successful'):
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
        """Process only new or modified text files in the data folder."""
        logger.info(f"Scanning for files in {data_folder}")
        text_files = glob.glob(os.path.join(data_folder, "*.txt"))
        logger.info(f"Found {len(text_files)} text files")
        
        # Initialize the processing tracker
        tracker = ProcessingTracker(os.path.join(data_folder, ".processed_files.json"))
        
        files_processed = 0
        for file_path in text_files:
            if tracker.should_process_file(file_path):
                logger.info(f"Processing new/modified file: {file_path}")
                self.process_file(file_path)
                tracker.mark_as_processed(file_path)
                files_processed += 1
            else:
                logger.info(f"Skipping already processed file: {file_path}")
        
        logger.info(f"Processed {files_processed} new/modified files out of {len(text_files)} total files")
        
def setup_weaviate_schema(client):
    """Set up the Weaviate schema for document chunks."""
    logger.info("Setting up Weaviate schema")
    
    try:
        # Check if the collection already exists
        if not client.collections.exists("DocumentChunk"):
            
            # Collection doesn't exist, create it            
            client.collections.create(
                name="DocumentChunk",
                vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
                properties=[
                    {
                        "name": "content",
                        "dataType": ["text"]
                    },
                    {
                        "name": "filename",
                        "dataType": ["text"]
                    },
                    {
                        "name": "chunkId",
                        "dataType": ["int"]
                    }
                ]
            )
            
            logger.info("DocumentChunk collection created in Weaviate")
        else:
            logger.info("DocumentChunk collection already exists in Weaviate")
    except Exception as e:
        logger.error(f"Error setting up Weaviate schema: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
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