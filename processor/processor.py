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
        start_time = time.time()
        logger.info("Setting up Weaviate schema for document chunks")
        
        try:
            # Check if the collection already exists
            if not self.client.collections.exists("DocumentChunk"):
                logger.info("DocumentChunk collection does not exist, creating new collection")
                creation_start = time.time()
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
        except Exception as e:
            logger.error(f"Error setting up Weaviate schema: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def delete_chunks(self, filename):
        """Delete all chunks associated with a specific filename."""
        start_time = time.time()
        logger.info(f"Deleting chunks for file: {filename}")
        
        try:
            # Get the collection
            collection = self.client.collections.get("DocumentChunk")
            
            # Create a proper filter for Weaviate 4.x
            from weaviate.classes.query import Filter
            where_filter = Filter.by_property("filename").equal(filename)
            
            # Delete using the filter
            deletion_start = time.time()
            result = collection.data.delete_many(
                where=where_filter
            )
            deletion_time = time.time() - deletion_start
            
            # Log the result
            if hasattr(result, 'successful'):
                logger.info(f"Deleted {result.successful} existing chunks for {filename} in {deletion_time:.2f}s")
            else:
                logger.info(f"No existing chunks found for {filename} ({deletion_time:.2f}s)")
                
            total_time = time.time() - start_time
            logger.debug(f"Total chunk deletion process took {total_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Error deleting existing chunks: {str(e)}")

    def store_chunk(self, content, filename, chunk_id, metadata=None):
        """Store a document chunk in Weaviate with metadata as a JSON string."""
        start_time = time.time()
        chunk_size = len(content)
        logger.debug(f"Storing chunk {chunk_id} from {filename} (size: {chunk_size} chars)")
        
        try:
            properties = {
                "content": content,
                "filename": filename,
                "chunkId": chunk_id
            }
            
            # Add metadata as a JSON string if provided
            if metadata and isinstance(metadata, dict):
                try:
                    properties["metadataJson"] = json.dumps(metadata)
                    logger.debug(f"Added metadata to chunk {chunk_id} from {filename}")
                except Exception as e:
                    logger.error(f"Error serializing metadata for {filename}, chunk {chunk_id}: {str(e)}")
            
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
                logger.debug(f"Object with ID {obj_uuid} not found for deletion (expected for new chunks)")
            
            # Now insert the object
            insert_start = time.time()
            collection.data.insert(
                properties=properties,
                uuid=obj_uuid
            )
            insert_time = time.time() - insert_start
            
            total_time = time.time() - start_time
            logger.debug(f"Stored chunk {chunk_id} from {filename} (size: {chunk_size} chars) in {total_time:.3f}s (insert: {insert_time:.3f}s)")
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id} from {filename}: {str(e)}")
                
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
        logger.info(f"Initializing file processing tracker at {tracker_file_path}")
        self.tracker_file_path = tracker_file_path
        self.processed_files = self._load_tracker()
        logger.info(f"Tracker initialized with {len(self.processed_files)} previously processed files")

    def _load_tracker(self):
        """Load the tracker file or create it if it doesn't exist."""
        if os.path.exists(self.tracker_file_path):
            try:
                logger.info(f"Loading existing tracker file from {self.tracker_file_path}")
                with open(self.tracker_file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded tracker with {len(data)} records")
                    return data
            except Exception as e:
                logger.error(f"Error loading tracker file: {str(e)}")
                return {}
        logger.info("No existing tracker file found, starting with empty tracking")
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
            
            logger.debug(f"Checking if file needs processing: {file_key}")
            
            # If file not in tracker or has been modified, process it
            if file_key not in self.processed_files:
                logger.info(f"File {file_key} not in tracker, will be processed")
                return True
            
            last_mod_time = self.processed_files[file_key]['last_modified']
            time_diff = file_mod_time - last_mod_time
            
            if time_diff > 0:
                logger.info(f"File {file_key} has been modified since last processing " +
                        f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mod_time))} vs. " +
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_mod_time))})")
                return True
            else:
                logger.debug(f"File {file_key} unchanged since last processing, skipping")
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
            process_time = time.time()
            
            self.processed_files[file_key] = {
                'path': file_path,
                'last_modified': file_mod_time,
                'last_processed': process_time,
                'last_processed_human': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(process_time))
            }
            self._save_tracker()
            logger.info(f"Marked {file_key} as processed (last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mod_time))})")
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
        start_time = time.time()
        
        # Log file size for context
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"Processing file: {file_path} (size: {file_size/1024:.1f} KB)")
        except Exception as e:
            logger.info(f"Processing file: {file_path} (size unknown: {str(e)})")
        
        # Validate file and get encoding
        encoding_start = time.time()
        is_valid, result = detect_file_encoding(file_path)
        encoding_time = time.time() - encoding_start
        
        if not is_valid:
            logger.error(f"Error processing file {file_path}: {result}")
            return False
        
        # Result is the encoding if valid
        encoding = result
        logger.info(f"File validated, using encoding: {encoding} (detection took {encoding_time:.2f}s)")
        
        try:
            read_start = time.time()
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            read_time = time.time() - read_start
            logger.info(f"File read complete in {read_time:.2f}s. Content length: {len(content)} characters")
            
            # Get the filename without the path
            filename = os.path.basename(file_path)

            # Check for associated metadata file
            metadata = None
            base_name = os.path.splitext(filename)[0]
            metadata_path = os.path.join(os.path.dirname(file_path), f"{base_name}.metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as meta_file:
                        metadata = json.load(meta_file)
                    logger.info(f"Loaded metadata from {metadata_path}")
                except Exception as e:
                    logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
            
            # Delete existing chunks for this file if any
            deletion_start = time.time()
            self.storage.delete_chunks(filename)
            deletion_time = time.time() - deletion_start
            logger.info(f"Previous chunks deletion completed in {deletion_time:.2f}s")
            
            # Split the content into chunks
            chunk_start = time.time()
            chunks = chunk_text(content, self.chunk_size, self.chunk_overlap)
            chunk_time = time.time() - chunk_start
            
            avg_chunk_size = sum(len(c) for c in chunks) / max(len(chunks), 1)
            logger.info(f"Text chunking complete in {chunk_time:.2f}s. Created {len(chunks)} chunks with avg size of {avg_chunk_size:.1f} chars")
            
            # Store each chunk in Weaviate
            storage_start = time.time()
            for i, chunk_content in enumerate(chunks):
                chunk_store_start = time.time()
                self.storage.store_chunk(chunk_content, filename, i, metadata)
                if i % 5 == 0 and i > 0:  # Log progress every 5 chunks
                    logger.info(f"Stored {i}/{len(chunks)} chunks from {filename}")
            
            storage_time = time.time() - storage_start
            logger.info(f"All chunks stored successfully in {storage_time:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"File {filename} processed successfully in total time: {total_time:.2f}s")
            return True
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        start_time = time.time()
        logger.info(f"Scanning for files in {directory_path}")
        text_files = glob.glob(os.path.join(directory_path, "*.txt"))
        logger.info(f"Found {len(text_files)} text files")
        
        stats = {
            "total": len(text_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }
        
        for i, file_path in enumerate(text_files):
            logger.info(f"Processing file {i+1}/{len(text_files)}: {os.path.basename(file_path)}")
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
        
        process_time = time.time() - start_time
        logger.info(f"Directory processing complete in {process_time:.2f}s. Stats: {stats}")
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
        logger.info("Initializing TextFileHandler for watching text files")
        self.processor = document_processor
        self.tracker = tracker
        self.stats = {
            "created": 0,
            "modified": 0,
            "deleted": 0,
            "processed": 0,
            "failed": 0
        }
        self.start_time = time.time()
        logger.info(f"TextFileHandler initialized with {'tracking enabled' if tracker else 'no tracking'}")

    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        event_time = time.time()
        file_size = os.path.getsize(event.src_path) if os.path.exists(event.src_path) else "unknown"
        logger.info(f"New text file detected: {event.src_path} (size: {file_size} bytes)")
        
        # Process the new file
        self.stats["created"] += 1
        success = self.processor.process_file(event.src_path)
        
        # Update tracker if processing was successful
        if success:
            self.stats["processed"] += 1
            if self.tracker:
                self.tracker.mark_as_processed(event.src_path)
        else:
            self.stats["failed"] += 1
            
        process_time = time.time() - event_time
        logger.info(f"Creation event processed in {process_time:.2f}s (success: {success})")
        
        # Log overall statistics periodically
        self._log_stats()

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        event_time = time.time()
        file_size = os.path.getsize(event.src_path) if os.path.exists(event.src_path) else "unknown"
        logger.info(f"Text file modified: {event.src_path} (size: {file_size} bytes)")
        
        # Process the modified file
        self.stats["modified"] += 1
        success = self.processor.process_file(event.src_path)
        
        # Update tracker if processing was successful
        if success:
            self.stats["processed"] += 1
            if self.tracker:
                self.tracker.mark_as_processed(event.src_path)
        else:
            self.stats["failed"] += 1
            
        process_time = time.time() - event_time
        logger.info(f"Modification event processed in {process_time:.2f}s (success: {success})")
        
        # Log overall statistics periodically
        self._log_stats()

    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        event_time = time.time()
        logger.info(f"Text file deleted: {event.src_path}")
        
        # Get the filename without the path
        filename = os.path.basename(event.src_path)
        
        # Delete the chunks from storage
        self.stats["deleted"] += 1
        self.processor.storage.delete_chunks(filename)
        
        # Update tracker if available
        tracker_updated = False
        if self.tracker:
            tracker_updated = self.tracker.remove_file(filename)
            
        process_time = time.time() - event_time
        logger.info(f"Deletion event processed in {process_time:.2f}s (tracker updated: {tracker_updated})")
        
        # Log overall statistics periodically
        self._log_stats()
    
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
    
    def _log_stats(self):
        """Log current statistics about file processing."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Log statistics every hour or after every 10 operations
        total_ops = sum(self.stats.values())
        if total_ops % 10 == 0 or (uptime // 3600) > ((uptime - 60) // 3600):
            logger.info(f"TextFileHandler statistics - Uptime: {uptime//3600:.0f}h {(uptime%3600)//60:.0f}m {uptime%60:.0f}s")
            logger.info(f"Events processed: created={self.stats['created']}, modified={self.stats['modified']}, " +
                    f"deleted={self.stats['deleted']}, successful={self.stats['processed']}, failed={self.stats['failed']}")

    def start_watching(self, data_folder, process_existing=True):
        """
        Start watching a folder for file changes.
        
        Args:
            data_folder: Path to the folder to watch
            process_existing: Whether to process existing files before watching
            
        Returns:
            Observer: The started watchdog observer
        """
        watch_start = time.time()
        logger.info(f"Starting to watch folder: {data_folder} (process existing: {process_existing})")
        
        # Process existing files if requested
        if process_existing:
            stats = self.process_existing_files(data_folder)
            logger.info(f"Processed existing files summary: total={stats['total']}, " +
                    f"processed={stats['processed']}, skipped={stats['skipped']}, failed={stats['failed']}")
        
        # Set up watchdog for future changes
        observer_start = time.time()
        observer = Observer()
        observer.schedule(self, data_folder, recursive=False)
        observer.start()
        
        watch_time = time.time() - watch_start
        logger.info(f"Now watching folder {data_folder} for changes (setup took {watch_time:.2f}s)")
        
        # Start regular stats logging
        self._start_stats_logging()
        
        return observer

    def _start_stats_logging(self):
        """Set up a background thread to log statistics periodically."""
        def log_stats_thread():
            while True:
                time.sleep(3600)  # Log stats every hour
                self._log_stats()
        
        import threading
        stats_thread = threading.Thread(target=log_stats_thread, daemon=True)
        stats_thread.start()
        logger.info("Started periodic statistics logging")

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