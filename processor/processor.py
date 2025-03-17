import os
import re
import json
import time
import logging
import uuid
import glob
import asyncio
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Set, 
    TypedDict, Callable, Iterable, Generator
)
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, DataType
from weaviate.classes.query import Filter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type definitions
class ChunkMetadata(TypedDict, total=False):
    """Type definition for chunk metadata."""
    page: Optional[int]
    heading: Optional[str]
    level: Optional[int]
    itemType: Optional[str]
    title: Optional[str]
    date: Optional[str]
    creators: Optional[List[Dict[str, str]]]

class TextChunk(TypedDict):
    """Type definition for a text chunk."""
    content: str
    page: Optional[int]
    heading: Optional[str]
    level: Optional[int]

class ProcessingResult(TypedDict):
    """Type definition for processing result."""
    success: bool
    message: str
    chunks_processed: int
    file_path: str
    metadata: Optional[Dict[str, Any]]

class ProcessingStats(TypedDict):
    """Type definition for processing statistics."""
    total: int
    processed: int
    skipped: int
    failed: int
    start_time: float
    duration: Optional[float]

class ChunkingStrategy(str, Enum):
    """Enum for different chunking strategies."""
    SIMPLE = "simple"           # Simple character-based chunking
    PARAGRAPH = "paragraph"     # Paragraph-based chunking
    SECTION = "section"         # Section-based (using headings)
    SEMANTIC = "semantic"       # Semantic chunking using AI

# Configuration class
class ProcessorConfig:
    """
    Configuration for document processor.
    
    Attributes:
        WEAVIATE_URL (str): URL for the Weaviate instance
        DATA_FOLDER (str): Folder to watch for documents
        CHUNK_SIZE (int): Default size of text chunks
        CHUNK_OVERLAP (int): Default overlap between chunks
        CHUNKING_STRATEGY (ChunkingStrategy): Strategy for chunking text
        MAX_RETRIES (int): Maximum number of retries for database operations
        RETRY_DELAY (int): Delay between retries in seconds
        FILE_EXTENSIONS (List[str]): Supported file extensions
        MAX_WORKER_THREADS (int): Maximum number of worker threads
        BATCH_SIZE (int): Number of chunks to process in a batch
    """
    
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    DATA_FOLDER = os.getenv("DATA_FOLDER", "/data")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    CHUNKING_STRATEGY = ChunkingStrategy(os.getenv("CHUNKING_STRATEGY", "section"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "10"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
    FILE_EXTENSIONS = [".md", ".txt"]  # Supported file extensions
    MAX_WORKER_THREADS = int(os.getenv("MAX_WORKER_THREADS", "5"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

config = ProcessorConfig()

# ------------ Utility Functions ------------
def chunk_text(
    text: str,
    max_chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    chunking_strategy: ChunkingStrategy = config.CHUNKING_STRATEGY
) -> List[TextChunk]:
    """
    Split text into overlapping chunks, respecting Markdown structure and sentence boundaries.
    
    This function processes Markdown documents using the following conventions:
    
    1. Page numbering:
       Use HTML comments to mark page numbers: <!-- page: 123 -->
    
    2. Heading structure:
       Standard Markdown heading syntax determines section hierarchy:
       # Heading 1
       ## Heading 2
       ### Heading 3
    
    3. Paragraphs:
       Separate paragraphs with blank lines
    
    The function preserves document structure by:
    - Keeping heading context with content
    - Respecting page boundaries
    - Maintaining heading hierarchy levels
    - Preserving paragraph and sentence boundaries when possible
    
    Args:
        text: The Markdown text to chunk
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        chunking_strategy: Strategy to use for chunking
        
    Returns:
        List[TextChunk]: List of chunks with metadata
    """
    # Input validation
    if not text:
        logger.warning("Empty text provided to chunk_text")
        return []
    
    if max_chunk_size <= 0:
        logger.warning(f"Invalid max_chunk_size: {max_chunk_size}, using default")
        max_chunk_size = config.CHUNK_SIZE
        
    if overlap < 0 or overlap >= max_chunk_size:
        logger.warning(f"Invalid overlap: {overlap}, using default")
        overlap = config.CHUNK_OVERLAP
    
    # Helper function for language-agnostic sentence splitting
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences, handling multiple languages."""
        # This pattern works for many European languages
        # It looks for periods, question marks, or exclamation points
        # followed by spaces and capital letters
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return sentences
    
    # Extract page markers
    page_pattern = re.compile(r'<!--\s*page:\s*(\d+)\s*-->')
    page_matches = list(page_pattern.finditer(text))
    
    # First, process the text into pages
    pages = []
    current_page_num = 1
    last_pos = 0
    
    # Process page markers
    for match in page_matches:
        # Add content before this page marker with current page number
        page_text = text[last_pos:match.start()]
        if page_text.strip():
            pages.append((page_text, current_page_num))
        
        # Update page number and position
        current_page_num = int(match.group(1))
        last_pos = match.end()
    
    # Add any remaining content
    if last_pos < len(text):
        pages.append((text[last_pos:], current_page_num))
    
    # Now process each page for headings and sections
    sections = []
    heading_pattern = re.compile(r'^(#+)\s+(.+)$', re.MULTILINE)
    
    for page_text, page_num in pages:
        # Find all headings in this page
        heading_matches = list(heading_pattern.finditer(page_text))
        
        if not heading_matches:
            # No headings on this page, treat whole page as one section
            sections.append({
                "text": page_text,
                "page": page_num,
                "heading": "Untitled Section",
                "level": 0
            })
            continue
        
        # Process sections based on headings
        last_heading_pos = 0
        current_heading = "Untitled Section"
        current_level = 0
        
        for i, match in enumerate(heading_matches):
            # Add content before this heading (if not at start)
            if i > 0 or match.start() > 0:
                section_text = page_text[last_heading_pos:match.start()]
                if section_text.strip():
                    sections.append({
                        "text": section_text,
                        "page": page_num,
                        "heading": current_heading,
                        "level": current_level
                    })
            
            # Update current heading and position
            heading_marks = match.group(1)  # The # characters
            current_level = len(heading_marks)  # Number of # determines level
            current_heading = match.group(2).strip()  # The heading text
            last_heading_pos = match.start()
        
        # Add the final section in this page
        final_section = page_text[last_heading_pos:]
        if final_section.strip():
            sections.append({
                "text": final_section,
                "page": page_num,
                "heading": current_heading,
                "level": current_level
            })
    
    # Now chunk each section with sentence boundary detection
    chunks: List[TextChunk] = []
    
    # Use the appropriate chunking strategy
    if chunking_strategy == ChunkingStrategy.SIMPLE:
        # Simple character-based chunking without respecting structure
        for section in sections:
            section_text = section["text"]
            for i in range(0, len(section_text), max_chunk_size - overlap):
                chunk_text = section_text[i:i + max_chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "page": section["page"],
                        "heading": section["heading"],
                        "level": section["level"]
                    })
    elif chunking_strategy == ChunkingStrategy.PARAGRAPH:
        # Paragraph-aware chunking
        for section in sections:
            section_text = section["text"]
            paragraphs = section_text.split('\n\n')
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    # Store current chunk if not empty
                    if current_chunk:
                        chunks.append({
                            "content": current_chunk,
                            "page": section["page"],
                            "heading": section["heading"],
                            "level": section["level"]
                        })
                    
                    # Start a new chunk with overlap if the paragraph is too large
                    if len(paragraph) > max_chunk_size:
                        # Recursively chunk large paragraphs
                        for i in range(0, len(paragraph), max_chunk_size - overlap):
                            sub_chunk = paragraph[i:i + max_chunk_size]
                            if sub_chunk.strip():
                                chunks.append({
                                    "content": sub_chunk,
                                    "page": section["page"],
                                    "heading": section["heading"],
                                    "level": section["level"]
                                })
                        current_chunk = ""
                    else:
                        current_chunk = paragraph
            
            # Add the last chunk if not empty
            if current_chunk:
                chunks.append({
                    "content": current_chunk,
                    "page": section["page"],
                    "heading": section["heading"],
                    "level": section["level"]
                })
    else:
        # Default to section-based chunking with sentence awareness (SECTION strategy)
        for section in sections:
            section_text = section["text"]
            section_heading = section["heading"]
            section_page = section["page"]
            section_level = section["level"]
            
            # Skip heading line itself when chunking
            content_start = section_text.find('\n')
            if content_start > 0:
                content = section_text[content_start:].strip()
            else:
                content = section_text.strip()
            
            if not content:
                continue  # Skip empty sections
            
            # For very small sections, keep them as a single chunk
            if len(content) <= max_chunk_size:
                chunks.append({
                    "content": content,
                    "page": section_page,
                    "heading": section_heading,
                    "level": section_level
                })
                continue
            
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            current_chunk = ""
            current_sentences = []
            
            for paragraph in paragraphs:
                # Split paragraph into sentences using our language-agnostic approach
                sentences = split_into_sentences(paragraph)
                
                # Process each sentence
                for sentence in sentences:
                    # If adding this sentence would exceed max size and we already have content
                    if len(current_chunk) + len(sentence) + 2 > max_chunk_size and current_chunk:  # +2 for the newline
                        chunks.append({
                            "content": current_chunk.strip(),
                            "page": section_page,
                            "heading": section_heading,
                            "level": section_level
                        })
                        
                        # For overlap, include sentences from the previous chunk
                        overlap_text = ""
                        overlap_size = 0
                        
                        # Work backwards through sentences to create overlap
                        for prev_sentence in reversed(current_sentences):
                            if overlap_size + len(prev_sentence) + 1 <= overlap:  # +1 for space
                                overlap_text = prev_sentence + " " + overlap_text
                                overlap_size += len(prev_sentence) + 1
                            else:
                                break
                        
                        # Start a new chunk with the overlap plus current sentence
                        current_chunk = overlap_text + sentence
                        current_sentences = [sentence]
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                        current_sentences.append(sentence)
                
                # Add paragraph separator if this isn't the last paragraph
                if current_chunk and paragraph != paragraphs[-1]:
                    current_chunk += "\n\n"
                    current_sentences.append("\n\n")
            
            # Add the last chunk from this section
            if current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "page": section_page,
                    "heading": section_heading,
                    "level": section_level
                })
    
    return chunks

def detect_file_encoding(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a readable text file and return the correct encoding.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple[bool, str]: (is_valid, encoding or error_message)
    """
    # Expanded list of encodings to try
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check if it's actually a file
    if not os.path.isfile(file_path):
        return False, "Path exists but is not a file"
    
    # Check for zero-length files
    try:
        if os.path.getsize(file_path) == 0:
            return False, "File is empty"
    except OSError as e:
        return False, f"Error checking file size: {str(e)}"
    
    # Try to read with each encoding
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                # Try to read a sample to verify encoding works
                sample = file.read(100)
                return True, encoding
        except UnicodeDecodeError:
            continue
        except PermissionError as e:
            return False, f"Permission error reading file: {str(e)}"
        except OSError as e:
            return False, f"Error reading file: {str(e)}"
    
    return False, "Could not decode with any supported encoding"

async def connect_with_retry(
    weaviate_url: str,
    max_retries: int = config.MAX_RETRIES,
    retry_delay: int = config.RETRY_DELAY
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
        ConnectionError: If connection fails after max retries
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
    raise ConnectionError(error_message)

# ------------ Document Storage Class ------------

class DocumentStorage:
    """
    Storage class for managing document chunks in Weaviate.
    
    This class handles:
    - Setting up the Weaviate schema
    - Deleting existing chunks
    - Storing new chunks with metadata
    
    Attributes:
        client: Weaviate client connection
    """
    
    def __init__(self, weaviate_client: weaviate.Client):
        """
        Initialize storage with a Weaviate client connection.
        
        Args:
            weaviate_client: Connected Weaviate client instance
        """
        self.client = weaviate_client
        
    async def setup_schema(self) -> bool:
        """
        Set up the Weaviate schema for document chunks.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
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
            return True
        except Exception as e:
            logger.error(f"Error setting up Weaviate schema: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def delete_chunks(self, filename: str) -> int:
        """
        Delete all chunks associated with a specific filename.
        
        Args:
            filename: The filename to delete chunks for
            
        Returns:
            int: Number of chunks deleted
        """
        start_time = time.time()
        logger.info(f"Deleting chunks for file: {filename}")
        
        try:
            # Get the collection
            collection = self.client.collections.get("DocumentChunk")
            
            # Create a proper filter for Weaviate
            where_filter = Filter.by_property("filename").equal(filename)
            
            # Delete using the filter
            deletion_start = time.time()
            result = collection.data.delete_many(
                where=where_filter
            )
            deletion_time = time.time() - deletion_start
            
            # Log the result
            deleted_count = 0
            if hasattr(result, 'successful'):
                deleted_count = result.successful
                logger.info(f"Deleted {deleted_count} existing chunks for {filename} in {deletion_time:.2f}s")
            else:
                logger.info(f"No existing chunks found for {filename} ({deletion_time:.2f}s)")
                
            total_time = time.time() - start_time
            logger.debug(f"Total chunk deletion process took {total_time:.2f}s")
            
            return deleted_count
                
        except Exception as e:
            logger.error(f"Error deleting existing chunks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    async def store_chunk(
        self, 
        content: str, 
        filename: str, 
        chunk_id: int, 
        metadata: Optional[Dict[str, Any]] = None, 
        page: Optional[int] = None, 
        heading: Optional[str] = None, 
        level: Optional[int] = None
    ) -> bool:
        """
        Store a document chunk in Weaviate with metadata as a JSON string.
        
        Args:
            content: The text content of the chunk
            filename: Source document name
            chunk_id: Sequential ID of the chunk within the document
            metadata: Document metadata from the .metadata.json file
            page: Page number where this chunk appears
            heading: Section heading text for this chunk
            level: Heading level (1 for #, 2 for ##, etc.)
            
        Returns:
            bool: True if storage was successful, False otherwise
        """
        start_time = time.time()
        chunk_size = len(content) if isinstance(content, str) else 0
        logger.debug(f"Storing chunk {chunk_id} from {filename} (size: {chunk_size} chars)")
        
        try:
            # Input validation
            if not content or not content.strip():
                logger.warning(f"Empty content for chunk {chunk_id} from {filename}")
                return False
                
            if not filename:
                logger.warning(f"Missing filename for chunk {chunk_id}")
                return False
                
            if chunk_id < 0:
                logger.warning(f"Invalid chunk_id: {chunk_id}")
                return False
            
            properties = {
                "content": content,
                "filename": filename,
                "chunkId": chunk_id
            }

            # Add page number and heading if available
            chunk_metadata = {}
            
            if page is not None:
                chunk_metadata["page"] = page
                
            if heading is not None:
                chunk_metadata["heading"] = heading

            # Add heading level if available
            if level is not None:
                chunk_metadata["headingLevel"] = level            

            # Merge with existing metadata if provided
            if metadata and isinstance(metadata, dict):
                chunk_metadata.update(metadata)

            # Add metadata as a JSON string if we have any
            if chunk_metadata:
                properties["metadataJson"] = json.dumps(chunk_metadata)    
                logger.debug(f"Added metadata to chunk {chunk_id} from {filename}")            

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
            return True
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id} from {filename}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
                
    async def store_chunks_batch(
        self, 
        chunks: List[Dict[str, Any]], 
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, int]:
        """
        Store multiple chunks in a batch operation.
        
        Args:
            chunks: List of chunk data dictionaries
            filename: Source document name
            metadata: Optional document metadata
            
        Returns:
            Tuple[int, int]: (successful_count, failed_count)
        """
        if not chunks:
            logger.warning(f"No chunks provided for {filename}")
            return 0, 0
            
        success_count = 0
        fail_count = 0
        
        # Process in batches to avoid overwhelming the database
        batch_size = config.BATCH_SIZE
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for idx, chunk in enumerate(batch):
                chunk_id = i + idx
                success = await self.store_chunk(
                    content=chunk["content"],
                    filename=filename,
                    chunk_id=chunk_id,
                    metadata=metadata,
                    page=chunk.get("page"),
                    heading=chunk.get("heading"),
                    level=chunk.get("level")
                )
                
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    
            # Log progress for large batches
            if len(chunks) > batch_size and i % (batch_size * 5) == 0 and i > 0:
                logger.info(f"Stored {i}/{len(chunks)} chunks from {filename}")
        
        logger.info(f"Batch storage complete for {filename}: {success_count} succeeded, {fail_count} failed")
        return success_count, fail_count

    async def get_chunks(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve chunks relevant to a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of chunk objects
        """
        try:
            collection = self.client.collections.get("DocumentChunk")
            results = collection.query.near_text(
                query=query,
                limit=limit,
                return_properties=["content", "filename", "chunkId", "metadataJson"]
            )
            
            return [obj.properties for obj in results.objects]
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
            
    async def get_document_count(self) -> int:
        """
        Get the count of unique documents in the database.
        
        Returns:
            int: Number of unique documents
        """
        try:
            collection = self.client.collections.get("DocumentChunk")
            
            query_result = collection.query.fetch_objects(
                return_properties=["filename"],
                limit=10000  # Practical limit for most cases
            )
            
            unique_filenames = set()
            for obj in query_result.objects:
                unique_filenames.add(obj.properties["filename"])
                
            return len(unique_filenames)
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return -1

# ------------ Processing Tracker Class ------------

class ProcessingTracker:
    """
    Tracks files that have been processed to avoid redundant processing.
    
    Attributes:
        tracker_file_path: Path to the JSON file storing processing records
        processed_files: Dictionary of processed files with metadata
    """
    
    def __init__(self, tracker_file_path: str = ".processed_files.json"):
        """
        Initialize a tracker that keeps record of processed files.
        
        Args:
            tracker_file_path: Path to the JSON file storing processing records
        """
        logger.info(f"Initializing file processing tracker at {tracker_file_path}")
        self.tracker_file_path = tracker_file_path
        self.processed_files = self._load_tracker()
        logger.info(f"Tracker initialized with {len(self.processed_files)} previously processed files")

    def _load_tracker(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the tracker file or create it if it doesn't exist.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of processed files with metadata
        """
        if os.path.exists(self.tracker_file_path):
            try:
                logger.info(f"Loading existing tracker file from {self.tracker_file_path}")
                with open(self.tracker_file_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Successfully loaded tracker with {len(data)} records")
                    return data
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding tracker file JSON: {str(e)}")
                return {}
            except PermissionError as e:
                logger.error(f"Permission error reading tracker file: {str(e)}")
                return {}
            except Exception as e:
                logger.error(f"Error loading tracker file: {str(e)}")
                return {}
        logger.info("No existing tracker file found, starting with empty tracking")
        return {}

    def _save_tracker(self) -> bool:
        """
        Save the tracker data to file.
        
        Returns:
            bool: Whether the save was successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.tracker_file_path) or '.', exist_ok=True)
            
            with open(self.tracker_file_path, 'w') as f:
                json.dump(self.processed_files, f, indent=2)
                
            logger.debug(f"Saved processing tracker to {self.tracker_file_path}")
            return True
        except PermissionError as e:
            logger.error(f"Permission error saving tracker file: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error saving tracker file: {str(e)}")
            return False
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on modification time.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file is new or modified since last processing
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
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error checking file status: {str(e)}")
            # If in doubt, process the file
            return True

    def mark_as_processed(self, file_path: str) -> bool:
        """
        Mark a file as processed with current timestamps.
        
        Args:
            file_path: Path to the file to mark as processed
            
        Returns:
            bool: Whether marking was successful
        """
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
            return True
        except Exception as e:
            logger.error(f"Error marking file as processed: {str(e)}")
            return False
    
    def remove_file(self, filename: str) -> bool:
        """
        Remove a file from the tracking record.
        
        Args:
            filename: Name of the file to remove from tracking
            
        Returns:
            bool: Whether removal was successful
        """
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
    
    def get_all_tracked_files(self) -> List[str]:
        """
        Return a list of all tracked filenames.
        
        Returns:
            List[str]: List of tracked filenames
        """
        return list(self.processed_files.keys())
    
    def clear_tracking_data(self) -> bool:
        """
        Clear all tracking data.
        
        Returns:
            bool: Whether clearing was successful
        """
        try:
            self.processed_files = {}
            self._save_tracker()
            logger.info("Cleared all file tracking data")
            return True
        except Exception as e:
            logger.error(f"Error clearing tracking data: {str(e)}")
            return False
    
    async def get_tracking_stats(self) -> Dict[str, Any]:
        """
        Get statistics about tracked files.
        
        Returns:
            Dict[str, Any]: Statistics about tracked files
        """
        stats = {
            "total_files_tracked": len(self.processed_files),
            "recently_processed": 0,
            "oldest_file": None,
            "newest_file": None,
            "average_file_age_days": 0
        }
        
        if not self.processed_files:
            return stats
            
        now = time.time()
        last_24h = now - (24 * 60 * 60)
        
        # Calculate stats
        file_ages = []
        oldest_time = now
        oldest_file = None
        newest_time = 0
        newest_file = None
        
        for filename, data in self.processed_files.items():
            # Count recently processed files
            if data.get('last_processed', 0) > last_24h:
                stats["recently_processed"] += 1
                
            # Track file ages
            mod_time = data.get('last_modified', 0)
            file_ages.append(now - mod_time)
            
            # Track oldest file
            if mod_time < oldest_time:
                oldest_time = mod_time
                oldest_file = filename
                
            # Track newest file
            if mod_time > newest_time:
                newest_time = mod_time
                newest_file = filename
        
        # Set stats
        if oldest_file:
            stats["oldest_file"] = {
                "name": oldest_file,
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(oldest_time))
            }
            
        if newest_file:
            stats["newest_file"] = {
                "name": newest_file,
                "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(newest_time))
            }
            
        if file_ages:
            avg_age_seconds = sum(file_ages) / len(file_ages)
            stats["average_file_age_days"] = round(avg_age_seconds / (24 * 60 * 60), 2)
            
        return stats

# ------------ Document Processor Class ------------

class DocumentProcessor:
    """
    Processes text files into chunks and stores them in a vector database.
    
    Attributes:
        storage: DocumentStorage instance for storing document chunks
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Amount of overlap between consecutive chunks
        chunking_strategy: Strategy to use for chunking text
    """
    
    def __init__(
        self, 
        storage: DocumentStorage, 
        chunk_size: int = config.CHUNK_SIZE, 
        chunk_overlap: int = config.CHUNK_OVERLAP,
        chunking_strategy: ChunkingStrategy = config.CHUNKING_STRATEGY
    ):
        """
        Initialize a document processor that reads and chunks text files.
        
        Args:
            storage: DocumentStorage instance for storing document chunks
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Amount of overlap between consecutive chunks
            chunking_strategy: Strategy to use for chunking text
        """
        self.storage = storage
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.metrics = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "processing_time": 0
        }
    
    async def process_file(self, file_path: str) -> ProcessingResult:
        """
        Process a markdown file: read, chunk, and store in vector database.
        
        Handles Markdown files with structured headings and optional page markers.
        See chunk_text() function for details on supported Markdown conventions.
        
        Args:
            file_path: Path to the markdown file to process
                
        Returns:
            ProcessingResult: Result of the processing operation
        """
        start_time = time.time()
        file_size = 0
        chunks_processed = 0
        
        # Default result for failures
        failure_result = ProcessingResult(
            success=False,
            message="Processing failed",
            chunks_processed=0,
            file_path=file_path,
            metadata=None
        )
        
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
            failure_result["message"] = f"File validation failed: {result}"
            return failure_result
        
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
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing metadata JSON from {metadata_path}: {str(e)}")
                except Exception as e:
                    logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
            
            # Delete existing chunks for this file if any
            deletion_start = time.time()
            await self.storage.delete_chunks(filename)
            deletion_time = time.time() - deletion_start
            logger.info(f"Previous chunks deletion completed in {deletion_time:.2f}s")
            
            # Split the content into chunks
            chunk_start = time.time()
            chunks = chunk_text(
                content, 
                self.chunk_size, 
                self.chunk_overlap,
                self.chunking_strategy
            )
            chunk_time = time.time() - chunk_start
            
            # Calculate average chunk size and log
            avg_chunk_size = sum(len(chunk["content"]) for chunk in chunks) / max(len(chunks), 1)
            logger.info(f"Text chunking complete in {chunk_time:.2f}s. Created {len(chunks)} chunks with avg size of {avg_chunk_size:.1f} chars")
            
            # Store chunks in Weaviate
            storage_start = time.time()
            success_count, fail_count = await self.storage.store_chunks_batch(chunks, filename, metadata)
            storage_time = time.time() - storage_start
            
            # Update metrics
            self.metrics["chunks_created"] += len(chunks)
            self.metrics["chunks_stored"] += success_count
            chunks_processed = success_count
            
            if fail_count > 0:
                logger.warning(f"{fail_count} chunks failed to store from {filename}")
            
            total_time = time.time() - start_time
            self.metrics["processing_time"] += total_time
            self.metrics["files_processed"] += 1
            
            logger.info(f"File {filename} processed successfully in total time: {total_time:.2f}s")
            
            return ProcessingResult(
                success=True,
                message=f"Processing successful, stored {success_count} chunks",
                chunks_processed=chunks_processed,
                file_path=file_path,
                metadata=metadata
            )
                
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error processing file {file_path}: {str(e)}")
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Unicode decode error: {str(e)}"
            return failure_result
        except PermissionError as e:
            logger.error(f"Permission error processing file {file_path}: {str(e)}")
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Permission error: {str(e)}"
            return failure_result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Processing error: {str(e)}"
            return failure_result
                    
    async def process_directory(
        self, 
        directory_path: str, 
        tracker: Optional[ProcessingTracker] = None,
        file_extensions: Optional[List[str]] = None
    ) -> ProcessingStats:
        """
        Process all markdown files in a directory.
        
        Args:
            directory_path: Path to the directory containing text files
            tracker: Optional ProcessingTracker to track processed files
            file_extensions: Optional list of file extensions to process
            
        Returns:
            ProcessingStats: Statistics about the processing
        """
        start_time = time.time()
        logger.info(f"Scanning for files in {directory_path}")
        
        # Use default extensions if none provided
        if file_extensions is None:
            file_extensions = config.FILE_EXTENSIONS
            
        # Find all files with the specified extensions
        all_files = []
        for ext in file_extensions:
            all_files.extend(glob.glob(os.path.join(directory_path, f"*{ext}")))
            
        logger.info(f"Found {len(all_files)} files with extensions: {', '.join(file_extensions)}")
        
        stats: ProcessingStats = {
            "total": len(all_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "start_time": start_time,
            "duration": None
        }
        
        # Filter files if tracker is provided
        files_to_process = []
        for file_path in all_files:
            if tracker and not tracker.should_process_file(file_path):
                logger.info(f"Skipping already processed file: {file_path}")
                stats["skipped"] += 1
            else:
                files_to_process.append(file_path)
        
        # Process files concurrently
        if files_to_process:
            # Create tasks for each file
            tasks = [self.process_file(file_path) for file_path in files_to_process]
            
            # Process in batches to avoid overwhelming resources
            batch_size = config.BATCH_SIZE
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                # Process the batch of files concurrently
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                # Handle results
                for j, result in enumerate(batch_results):
                    file_path = files_to_process[i + j]
                    
                    if isinstance(result, Exception):
                        logger.error(f"Exception processing {file_path}: {str(result)}")
                        stats["failed"] += 1
                    elif result["success"]:
                        stats["processed"] += 1
                        if tracker:
                            tracker.mark_as_processed(file_path)
                    else:
                        stats["failed"] += 1
                        logger.error(f"Failed to process {file_path}: {result['message']}")
                
                # Log progress for large batches
                if len(files_to_process) > batch_size and i > 0:
                    logger.info(f"Processed {i + len(batch)}/{len(files_to_process)} files")
        
        process_time = time.time() - start_time
        stats["duration"] = process_time
        logger.info(f"Directory processing complete in {process_time:.2f}s. Stats: {stats}")
        return stats

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current processing metrics.
        
        Returns:
            Dict[str, Any]: Current processing metrics
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters to zero."""
        self.metrics = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "processing_time": 0
        }

# ------------ File Watcher Class ------------

class TextFileHandler(FileSystemEventHandler):
    """
    Watchdog event handler for monitoring file changes.
    
    Attributes:
        processor: DocumentProcessor instance to process files
        tracker: Optional ProcessingTracker to track processed files
        stats: Statistics about processing events
    """
    
    def __init__(
        self, 
        document_processor: DocumentProcessor, 
        tracker: Optional[ProcessingTracker] = None
    ):
        """
        Initialize a file system event handler for Markdown files.
        
        Args:
            document_processor: DocumentProcessor instance to process files
            tracker: Optional ProcessingTracker to track processed files
        """
        logger.info("Initializing TextFileHandler for watching Markdown files")
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
        """
        Handle file creation events.
        
        Args:
            event: The file system event
        """
        # Check if this is a file we should process
        if event.is_directory:
            return
            
        file_path = event.src_path
        _, file_ext = os.path.splitext(file_path)
        
        if file_ext not in config.FILE_EXTENSIONS:
            return
        
        event_time = time.time()
        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else "unknown"
            logger.info(f"New file detected: {file_path} (size: {file_size} bytes)")
            
            # Process the new file
            self.stats["created"] += 1
            
            # Use the event loop to run the async process_file method
            # Instead of asyncio.run which can't be nested
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                result = loop.run_until_complete(self.processor.process_file(file_path))
            else:
                # Create a Future to get the result
                future = asyncio.run_coroutine_threadsafe(
                    self.processor.process_file(file_path), 
                    loop
                )
                result = future.result()
            
            # Update tracker if processing was successful
            if result["success"]:
                self.stats["processed"] += 1
                if self.tracker:
                    self.tracker.mark_as_processed(file_path)
            else:
                self.stats["failed"] += 1
                
            process_time = time.time() - event_time
            logger.info(f"Creation event processed in {process_time:.2f}s (success: {result['success']})")
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Error handling file creation event: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Log overall statistics periodically
        self._log_stats()

    def on_modified(self, event):
        """
        Handle file modification events.
        
        Args:
            event: The file system event
        """
        # Check if this is a file we should process
        if event.is_directory:
            return
            
        file_path = event.src_path
        _, file_ext = os.path.splitext(file_path)
        
        if file_ext not in config.FILE_EXTENSIONS:
            return
        
        event_time = time.time()
        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else "unknown"
            logger.info(f"File modified: {file_path} (size: {file_size} bytes)")
            
            # Process the modified file
            self.stats["modified"] += 1
            
            # Use the event loop to run the async process_file method
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                result = loop.run_until_complete(self.processor.process_file(file_path))
            else:
                # Create a Future to get the result
                future = asyncio.run_coroutine_threadsafe(
                    self.processor.process_file(file_path), 
                    loop
                )
                result = future.result()
            
            # Update tracker if processing was successful
            if result["success"]:
                self.stats["processed"] += 1
                if self.tracker:
                    self.tracker.mark_as_processed(file_path)
            else:
                self.stats["failed"] += 1
                
            process_time = time.time() - event_time
            logger.info(f"Modification event processed in {process_time:.2f}s (success: {result['success']})")
        except Exception as e:
            self.stats["failed"] += 1
            logger.error(f"Error handling file modification event: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Log overall statistics periodically
        self._log_stats()

    def on_deleted(self, event):
        """
        Handle file deletion events.
        
        Args:
            event: The file system event
        """
        # Check if this is a file we should process
        if event.is_directory:
            return
            
        file_path = event.src_path
        _, file_ext = os.path.splitext(file_path)
        
        if file_ext not in config.FILE_EXTENSIONS:
            return
        
        event_time = time.time()
        try:
            logger.info(f"File deleted: {file_path}")
            
            # Get the filename without the path
            filename = os.path.basename(file_path)
            
            # Delete the chunks from storage
            self.stats["deleted"] += 1
            
            # Use the event loop to run the async delete_chunks method
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self.processor.storage.delete_chunks(filename))
            else:
                # Create a Future for the deletion
                future = asyncio.run_coroutine_threadsafe(
                    self.processor.storage.delete_chunks(filename), 
                    loop
                )
                future.result()
            
            # Update tracker if available
            tracker_updated = False
            if self.tracker:
                tracker_updated = self.tracker.remove_file(filename)
                
            process_time = time.time() - event_time
            logger.info(f"Deletion event processed in {process_time:.2f}s (tracker updated: {tracker_updated})")
        except Exception as e:
            logger.error(f"Error handling file deletion event: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Log overall statistics periodically
        self._log_stats()
    
    async def process_existing_files(self, data_folder: str) -> ProcessingStats:
        """
        Process existing text files in the specified folder.
        
        Args:
            data_folder: Path to the folder containing text files
            
        Returns:
            ProcessingStats: Processing statistics
        """
        logger.info(f"Processing existing files in {data_folder}")
        return await self.processor.process_directory(data_folder, self.tracker)
    
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

    def start_watching(
        self, 
        data_folder: str, 
        process_existing: bool = True
    ) -> Observer:
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
        
        # Process existing files if requested - but do it synchronously to avoid
        # event loop complications
        if process_existing:
            # Find Markdown files in the directory
            files = []
            for ext in config.FILE_EXTENSIONS:
                files.extend(glob.glob(os.path.join(data_folder, f"*{ext}")))
            
            processed = 0
            skipped = 0
            failed = 0
            
            logger.info(f"Found {len(files)} existing files to process")
            
            # Process each file one by one synchronously
            for file_path in files:
                if self.tracker and not self.tracker.should_process_file(file_path):
                    logger.info(f"Skipping already processed file: {file_path}")
                    skipped += 1
                    continue
                
                try:
                    # Use a separate event loop for each file to avoid nesting issues
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(self.processor.process_file(file_path))
                    loop.close()
                    
                    if result["success"]:
                        processed += 1
                        if self.tracker:
                            self.tracker.mark_as_processed(file_path)
                    else:
                        failed += 1
                        logger.error(f"Failed to process {file_path}: {result['message']}")
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    
            logger.info(f"Processed existing files summary: total={len(files)}, " +
                    f"processed={processed}, skipped={skipped}, failed={failed}")
        
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
    """
    Main function to run the document processor.
    """
    # Connect to Weaviate
    weaviate_url = config.WEAVIATE_URL
    data_folder = config.DATA_FOLDER
    
    try:
        # Set up a single event loop for the entire application
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Connect to Weaviate
        logger.info(f"Connecting to Weaviate at {weaviate_url}")
        client = loop.run_until_complete(connect_with_retry(weaviate_url))
        logger.info("Successfully connected to Weaviate")
        
        # Create storage, processor, and file handler instances
        storage = DocumentStorage(client)
        loop.run_until_complete(storage.setup_schema())
        
        tracker = ProcessingTracker(os.path.join(data_folder, ".processed_files.json"))
        processor = DocumentProcessor(
            storage,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            chunking_strategy=config.CHUNKING_STRATEGY
        )
        
        handler = TextFileHandler(processor, tracker)
        observer = handler.start_watching(data_folder, process_existing=True)
        
        # Log initial document count
        doc_count = loop.run_until_complete(storage.get_document_count())
        logger.info(f"Current document count in database: {doc_count}")
        
        # Keep the main thread running with signal handling
        try:
            import signal
            
            def signal_handler(sig, frame):
                logger.info("Stopping the file watcher (signal received)")
                observer.stop()
                raise KeyboardInterrupt
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            while observer.is_alive():
                observer.join(1)
                
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            observer.stop()
        finally:
            observer.join()
            # Close the client connection
            client.close()
            # Close the event loop
            loop.close()
            logger.info("Processor shutdown complete")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()