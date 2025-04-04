"""
Document processor for ingesting and processing document files.
"""
import os
import json
import time
from typing import Dict, Any, List, Set, Optional, Tuple

from config import settings
from utils.logging_config import get_logger
from utils.errors import (
    FileProcessingError, FileReadError, MetadataError, 
    WeaviateOperationError, TrackerError
)
from utils.file_utils import (
    detect_file_encoding, read_file_content, 
    get_all_document_files, get_metadata_for_file,
    get_relative_path
)
from storage.document_storage import DocumentStorage
from core.processor_tracker import ProcessingTracker
from core.chunking import chunk_text, TextChunk

logger = get_logger(__name__)

class ProcessingResult(Dict[str, Any]):
    """
    Type definition for processing result.
    
    Keys:
        success (bool): Whether processing was successful
        message (str): Result message
        chunks_processed (int): Number of chunks processed
        file_path (str): Path to the processed file
        metadata (Optional[Dict[str, Any]]): Document metadata
    """
    pass

class ProcessingStats(Dict[str, Any]):
    """
    Type definition for processing statistics.
    
    Keys:
        total (int): Total number of files
        processed (int): Number of processed files
        skipped (int): Number of skipped files
        failed (int): Number of failed files
        start_time (float): Start time in seconds since epoch
        duration (Optional[float]): Duration in seconds
    """
    pass

class DocumentProcessor:
    """
    Processes text files into chunks and stores them in a vector database.
    
    Attributes:
        storage: DocumentStorage instance for storing document chunks
        tracker: ProcessingTracker instance for tracking processed files
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Amount of overlap between consecutive chunks
        chunking_strategy: Strategy to use for chunking text
        process_subfolders: Whether to process files in subdirectories
    """
    
    def __init__(
        self, 
        storage: DocumentStorage, 
        tracker: ProcessingTracker = None,
        chunk_size: int = settings.CHUNK_SIZE, 
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        chunking_strategy: str = settings.CHUNKING_STRATEGY,
        process_subfolders: bool = settings.PROCESS_SUBFOLDERS
    ):
        """
        Initialize a document processor that reads and chunks text files.
        
        Args:
            storage: DocumentStorage instance for storing document chunks
            tracker: ProcessingTracker instance for tracking processed files
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Amount of overlap between consecutive chunks
            chunking_strategy: Strategy to use for chunking text
            process_subfolders: Whether to process files in subdirectories
        """
        self.storage = storage
        self.tracker = tracker or ProcessingTracker()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.process_subfolders = process_subfolders
        self.metrics = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "chunks_stored": 0,
            "processing_time": 0
        }
        
        logger.info(f"Document processor initialized with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, chunking_strategy={chunking_strategy}, "
                   f"process_subfolders={process_subfolders}")
    
    async def process_file(self, file_path: str) -> ProcessingResult:
        """
        Process a markdown file: read, chunk, and store in vector database.
        
        Handles Markdown files with structured headings and optional page markers.
        See chunk_text() function for details on supported Markdown conventions.
        
        Args:
            file_path: Path to the markdown file to process
                
        Returns:
            ProcessingResult: Result of the processing operation
            
        Raises:
            FileProcessingError: If file processing fails
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
        
        logger.info(f"Processing file: {file_path}")
        
        # Log file size for context
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size/1024:.1f} KB")
        except Exception as e:
            logger.warning(f"Could not determine file size: {str(e)}")
        
        try:
            # Detect file encoding
            encoding_start = time.time()
            _, encoding = detect_file_encoding(file_path)
            encoding_time = time.time() - encoding_start
            logger.info(f"File validated, using encoding: {encoding} (detection took {encoding_time:.2f}s)")
            
            # Read file content
            read_start = time.time()
            content = read_file_content(file_path, encoding)
            read_time = time.time() - read_start
            logger.info(f"File read complete in {read_time:.2f}s. Content length: {len(content)} characters")
            
            # Calculate relative path for filename in database
            data_folder = settings.DATA_FOLDER
            filename = get_relative_path(file_path, data_folder)
            logger.info(f"Using storage key: {filename}")

            # Load metadata if available
            try:
                metadata = get_metadata_for_file(file_path)
                if metadata:
                    logger.info(f"Loaded metadata for {filename}")
            except MetadataError as e:
                metadata = None
                logger.warning(f"Could not load metadata for {filename}: {str(e)}")
            
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
                
        except FileReadError as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"File read error: {str(e)}"
            raise FileProcessingError(str(e), file_path, {"original_error": str(e)})
        except WeaviateOperationError as e:
            logger.error(f"Weaviate error processing file {file_path}: {str(e)}")
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Storage error: {str(e)}"
            raise FileProcessingError(str(e), file_path, {"original_error": str(e)})
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.metrics["files_failed"] += 1
            failure_result["message"] = f"Processing error: {str(e)}"
            raise FileProcessingError(str(e), file_path, {"original_error": str(e)})
                    
    async def process_all_documents(self, data_folder: str = settings.DATA_FOLDER) -> Dict[str, Any]:
        """
        Process all documents in the data folder.
        Compare current files to tracked files and handle additions, modifications, and deletions.
        
        Args:
            data_folder: Path to the folder containing documents
            
        Returns:
            dict: Statistics about processing
            
        Raises:
            FileProcessingError: If processing fails
        """
        stats = {
            "total_files": 0,
            "new_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "unchanged_files": 0,
            "processed_success": 0,
            "processed_failed": 0
        }
        
        try:
            start_time = time.time()
            
            # Get all current files in the data folder, including in subdirectories if enabled
            current_files = get_all_document_files(data_folder, self.process_subfolders)
            
            stats["total_files"] = len(current_files)
            logger.info(f"Found {len(current_files)} files in {data_folder}" + 
                       (f" (including subdirectories)" if self.process_subfolders else ""))
            
            # Get previously tracked files
            tracked_files = self.tracker.get_all_tracked_files()
            
            logger.info(f"Found {len(tracked_files)} files in tracking data")
            
            # Process new or modified files
            for file_path in current_files:
                try:
                    if file_path not in tracked_files:
                        logger.info(f"New file detected: {file_path}")
                        stats["new_files"] += 1
                    elif self.tracker.should_process_file(file_path):
                        logger.info(f"Modified file detected: {file_path}")
                        stats["modified_files"] += 1
                    else:
                        logger.info(f"Unchanged file: {file_path}")
                        stats["unchanged_files"] += 1
                        continue
                    
                    # Process the file
                    result = await self.process_file(file_path)
                    
                    if result["success"]:
                        stats["processed_success"] += 1
                        self.tracker.mark_as_processed(file_path)
                        logger.info(f"Successfully processed: {file_path}")
                    else:
                        stats["processed_failed"] += 1
                        logger.error(f"Failed to process {file_path}: {result['message']}")
                except Exception as e:
                    stats["processed_failed"] += 1
                    logger.error(f"Error processing {file_path}: {str(e)}")
            
            # Handle deleted files
            for file_path in tracked_files:
                if file_path not in current_files:
                    logger.info(f"Deleted file detected: {file_path}")
                    stats["deleted_files"] += 1
                    
                    try:
                        # Get the relative path for storage deletion
                        file_key = self.tracker._get_file_key(file_path)
                        
                        # Delete chunks from Weaviate
                        await self.storage.delete_chunks(file_key)
                        
                        # Update tracker
                        self.tracker.remove_file(file_path)
                        logger.info(f"Successfully processed deletion: {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing deletion {file_path}: {str(e)}")
            
            # Add duration to stats
            stats["duration"] = time.time() - start_time
            logger.info(f"Document processing completed in {stats['duration']:.2f}s")
            
            return stats
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise FileProcessingError(str(e), data_folder, {"original_error": str(e)})
    
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