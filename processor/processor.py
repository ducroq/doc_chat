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
from weaviate.classes.config import Configure, DataType
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ------------ Utility Functions ------------

def chunk_text(text, max_chunk_size=1000, overlap=200):
    """
    Split text into overlapping chunks of approximately max_chunk_size characters.
    """
    # Simple paragraph-based chunking
    # Could we improve this by more intelligently splitting chunks?
    # What about using a transformer model to split text into meaningful chunks
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

def detect_file_encoding(file_path):
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

# ------------ Document Storage Class ------------

class DocumentStorage:
    def __init__(self, weaviate_client):
        """Initialize storage with a Weaviate client connection."""
        self.client = weaviate_client
        
    def setup_schema(self):
        """Set up the Weaviate schema for document chunks."""
        try:
            # Check if the collection already exists
            if not self.client.collections.exists("DocumentChunk"):
                # Collection doesn't exist, create it            
                self.client.collections.create(
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
                        )
                    ]
                )
                logger.info("DocumentChunk collection created in Weaviate")
            else:
                logger.info("DocumentChunk collection already exists in Weaviate")
        except Exception as e:
            logger.error(f"Error setting up Weaviate schema: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def delete_chunks(self, filename):
        """Delete all chunks associated with a specific filename."""
        try:
            # Get the collection
            collection = self.client.collections.get("DocumentChunk")
            
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
        """Store a document chunk in Weaviate."""
        try:
            properties = {
                "content": content,
                "filename": filename,
                "chunkId": chunk_id
            }
            
            # Create a UUID based on filename and chunk_id for consistency
            obj_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{chunk_id}"))

            # Get the DocumentChunk collection
            collection = self.client.collections.get("DocumentChunk")
            
            # First, try to delete the object if it exists
            try:
                collection.data.delete_by_id(obj_uuid)
                logger.debug(f"Deleted existing object with ID {obj_uuid}")
            except Exception as delete_error:
                # It's okay if the object doesn't exist yet
                logger.debug(f"Object with ID {obj_uuid} not found for deletion: {str(delete_error)}")
            
            # Now insert the object
            collection.data.insert(
                properties=properties,
                uuid=obj_uuid
            )
            logger.info(f"Stored chunk {chunk_id} from {filename}")
        except Exception as e:
            logger.error(f"Error storing chunk: {str(e)}")
    
    def get_chunks(self, query, limit=3):
        """
        Retrieve chunks relevant to a query.
        Returns a list of chunk objects.
        """
        try:
            collection = self.client.collections.get("DocumentChunk")
            results = collection.query.near_text(
                query=query,
                limit=limit,
                return_properties=["content", "filename", "chunkId"]
            )
            
            return results.objects
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return []

# ------------ Processing Tracker Class ------------

class ProcessingTracker:
    def __init__(self, tracker_file_path="processed_files.json"):
        """
        Initialize a tracker that keeps record of processed files.
        
        Args:
            tracker_file_path: Path to the JSON file storing processing records
        """
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
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracker_file_path), exist_ok=True)
            
            with open(self.tracker_file_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
                
            logger.debug(f"Saved processing tracker to {self.tracker_file_path}")
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
            logger.info(f"Marked {file_key} as processed")
        except Exception as e:
            logger.error(f"Error marking file as processed: {str(e)}")
    
    def remove_file(self, filename):
        """Remove a file from the tracking record."""
        try:
            if filename in self.processed_files:
                logger.info(f"Removing {filename} from processing tracker")
                del self.processed_files[filename]
                self._save_tracker()
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing file from tracker: {str(e)}")
            return False
    
    def get_all_tracked_files(self):
        """Return a list of all tracked filenames."""
        return list(self.processed_files.keys())
    
    def clear_tracking_data(self):
        """Clear all tracking data."""
        try:
            self.processed_files = {}
            self._save_tracker()
            logger.info("Cleared all file tracking data")
            return True
        except Exception as e:
            logger.error(f"Error clearing tracking data: {str(e)}")
            return False

# ------------ Document Processor Class ------------

class DocumentProcessor:
    def __init__(self, storage, chunk_size=1000, chunk_overlap=200):
        """
        Initialize a document processor that reads and chunks text files.
        
        Args:
            storage: DocumentStorage instance for storing document chunks
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Amount of overlap between consecutive chunks
        """
        self.storage = storage
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_file(self, file_path):
        """
        Process a text file: read, chunk, and store in vector database.
        
        Args:
            file_path: Path to the text file to process
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        logger.info(f"Processing file: {file_path}")
        
        # Validate file and get encoding
        is_valid, result = detect_file_encoding(file_path)
        
        if not is_valid:
            logger.error(f"Error processing file {file_path}: {result}")
            return False
        
        # Result is the encoding if valid
        encoding = result
        logger.info(f"File validated, using encoding: {encoding}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
                logger.info(f"File content length: {len(content)} characters")
                
                # Get the filename without the path
                filename = os.path.basename(file_path)
                
                # Delete existing chunks for this file if any
                self.storage.delete_chunks(filename)
                
                # Split the content into chunks
                chunks = chunk_text(content, self.chunk_size, self.chunk_overlap)
                logger.info(f"Split into {len(chunks)} chunks")
                
                # Store each chunk in Weaviate
                for i, chunk_content in enumerate(chunks):
                    self.storage.store_chunk(chunk_content, filename, i)
                    
                logger.info(f"File {filename} processed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return False
    
    def process_directory(self, directory_path, tracker=None):
        """
        Process all text files in a directory.
        
        Args:
            directory_path: Path to the directory containing text files
            tracker: Optional ProcessingTracker to track processed files
            
        Returns:
            dict: Statistics about the processing (total, processed, failed)
        """
        logger.info(f"Scanning for files in {directory_path}")
        text_files = glob.glob(os.path.join(directory_path, "*.txt"))
        logger.info(f"Found {len(text_files)} text files")
        
        stats = {
            "total": len(text_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }
        
        for file_path in text_files:
            # If tracker is provided, check if file needs processing
            if tracker and not tracker.should_process_file(file_path):
                logger.info(f"Skipping already processed file: {file_path}")
                stats["skipped"] += 1
                continue
                
            # Process the file
            success = self.process_file(file_path)
            
            # Update tracking and stats
            if success:
                stats["processed"] += 1
                if tracker:
                    tracker.mark_as_processed(file_path)
            else:
                stats["failed"] += 1
        
        logger.info(f"Directory processing complete. Stats: {stats}")
        return stats

# ------------ File Watcher Class ------------

class TextFileHandler(FileSystemEventHandler):
    def __init__(self, document_processor, tracker=None):
        """
        Initialize a file system event handler for text files.
        
        Args:
            document_processor: DocumentProcessor instance to process files
            tracker: Optional ProcessingTracker to track processed files
        """
        self.processor = document_processor
        self.tracker = tracker
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        logger.info(f"New text file detected: {event.src_path}")
        
        # Process the new file
        success = self.processor.process_file(event.src_path)
        
        # Update tracker if processing was successful
        if success and self.tracker:
            self.tracker.mark_as_processed(event.src_path)
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        logger.info(f"Text file modified: {event.src_path}")
        
        # Process the modified file
        success = self.processor.process_file(event.src_path)
        
        # Update tracker if processing was successful
        if success and self.tracker:
            self.tracker.mark_as_processed(event.src_path)
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        logger.info(f"Text file deleted: {event.src_path}")
        
        # Get the filename without the path
        filename = os.path.basename(event.src_path)
        
        # Delete the chunks from storage
        self.processor.storage.delete_chunks(filename)
        
        # Update tracker if available
        if self.tracker:
            self.tracker.remove_file(filename)
    
    def process_existing_files(self, data_folder):
        """
        Process existing text files in the specified folder.
        
        Args:
            data_folder: Path to the folder containing text files
            
        Returns:
            dict: Processing statistics
        """
        logger.info(f"Processing existing files in {data_folder}")
        return self.processor.process_directory(data_folder, self.tracker)
    
    def start_watching(self, data_folder, process_existing=True):
        """
        Start watching a folder for file changes.
        
        Args:
            data_folder: Path to the folder to watch
            process_existing: Whether to process existing files before watching
            
        Returns:
            Observer: The started watchdog observer
        """
        logger.info(f"Starting to watch folder: {data_folder}")
        
        # Process existing files if requested
        if process_existing:
            self.process_existing_files(data_folder)
        
        # Set up watchdog for future changes
        observer = Observer()
        observer.schedule(self, data_folder, recursive=False)
        observer.start()
        
        logger.info(f"Now watching folder {data_folder} for changes")
        return observer

# ------------ Main Function ------------

def main():
    # Connect to Weaviate
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    data_folder = os.getenv("DATA_FOLDER", "/data")
    
    try:
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        client = connect_with_retry(weaviate_url)
        logger.info("Successfully connected to Weaviate")
        
        # Create storage, processor, and file handler instances
        storage = DocumentStorage(client)
        storage.setup_schema()
        
        tracker = ProcessingTracker(os.path.join(data_folder, ".processed_files.json"))
        processor = DocumentProcessor(storage)
        
        handler = TextFileHandler(processor, tracker)
        observer = handler.start_watching(data_folder, process_existing=True)
        
        # Keep the main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping the file watcher")
            observer.stop()
        observer.join()
        
    except Exception as e:
        logger.error(f"Error initializing the processor: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()