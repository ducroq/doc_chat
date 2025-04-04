"""
Document storage module for the document processor.
Handles storing and retrieving document chunks from Weaviate.
"""
import json
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional

import weaviate
from weaviate.classes.query import Filter

from config import settings
from utils.logging_config import get_logger
from utils.errors import WeaviateOperationError
from core.chunking import TextChunk

logger = get_logger(__name__)

class DocumentStorage:
    """
    Storage class for managing document chunks in Weaviate.
    
    This class handles:
    - Deleting existing chunks
    - Storing new chunks with metadata
    - Retrieving chunks for verification
    
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
        
    async def delete_chunks(self, filename: str) -> int:
        """
        Delete all chunks associated with a specific filename.
        
        Args:
            filename: The filename to delete chunks for (can be relative path)
            
        Returns:
            int: Number of chunks deleted
            
        Raises:
            WeaviateOperationError: If deletion fails
        """
        start_time = time.time()
        logger.info(f"Deleting chunks for file: {filename}")
        
        try:
            # Get the collection
            collection = self.client.collections.get("DocumentChunk")
            
            # Create a proper filter for Weaviate - use exact match on filename
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
            error_message = f"Error deleting existing chunks: {str(e)}"
            logger.error(error_message)
            import traceback
            logger.error(traceback.format_exc())
            raise WeaviateOperationError(error_message, {
                "filename": filename,
                "original_error": str(e)
            })

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
            
        Raises:
            WeaviateOperationError: If storing the chunk fails
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
            
            # First, try to delete the object if it exists to avoid conflicts
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
            error_message = f"Error storing chunk {chunk_id} from {filename}: {str(e)}"
            logger.error(error_message)
            import traceback
            logger.error(traceback.format_exc())
            raise WeaviateOperationError(error_message, {
                "filename": filename,
                "chunk_id": chunk_id,
                "original_error": str(e)
            })
                
    async def store_chunks_batch(
        self, 
        chunks: List[TextChunk], 
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
            
        Raises:
            WeaviateOperationError: If the batch operation fails
        """
        if not chunks:
            logger.warning(f"No chunks provided for {filename}")
            return 0, 0
            
        success_count = 0
        fail_count = 0
        
        # Process in batches to avoid overwhelming the database
        batch_size = settings.BATCH_SIZE
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for idx, chunk in enumerate(batch):
                chunk_id = i + idx
                try:
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
                except Exception as e:
                    logger.error(f"Error storing chunk {chunk_id} from {filename}: {str(e)}")
                    fail_count += 1
                    
            # Log progress for large batches
            if len(chunks) > batch_size and i % (batch_size * 5) == 0 and i > 0:
                logger.info(f"Stored {i}/{len(chunks)} chunks from {filename}")
        
        logger.info(f"Batch storage complete for {filename}: {success_count} succeeded, {fail_count} failed")
        return success_count, fail_count

    async def get_document_count(self) -> int:
        """
        Get the count of unique documents in the database.
        
        Returns:
            int: Number of unique documents
            
        Raises:
            WeaviateOperationError: If the count operation fails
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
            error_message = f"Error counting documents: {str(e)}"
            logger.error(error_message)
            raise WeaviateOperationError(error_message, {"original_error": str(e)})