"""
Custom error classes for the document processor.
"""
from typing import Optional, Dict, Any

class ProcessorError(Exception):
    """Base class for all processor errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class WeaviateConnectionError(ProcessorError):
    """Error connecting to Weaviate."""
    pass

class WeaviateOperationError(ProcessorError):
    """Error performing operation on Weaviate."""
    pass

class FileProcessingError(ProcessorError):
    """Error processing a file."""
    def __init__(self, message: str, file_path: str, details: Optional[Dict[str, Any]] = None):
        self.file_path = file_path
        super().__init__(message, details)

class FileReadError(FileProcessingError):
    """Error reading a file."""
    pass

class FileEncodingError(FileProcessingError):
    """Error detecting file encoding."""
    pass

class MetadataError(FileProcessingError):
    """Error processing metadata for a file."""
    pass

class ChunkingError(ProcessorError):
    """Error chunking text."""
    pass

class TrackerError(ProcessorError):
    """Error with the process tracker."""
    pass