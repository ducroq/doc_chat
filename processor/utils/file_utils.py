"""
File utilities for the document processor.
Handles file reading, encoding detection, and path operations.
"""
import os
import glob
from pathlib import Path
from typing import Tuple, List, Set, Dict, Any, Optional
import json

from utils.logging_config import get_logger
from utils.errors import FileReadError, FileEncodingError, MetadataError
from config import settings

logger = get_logger(__name__)

def detect_file_encoding(file_path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a readable text file and return the correct encoding.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple[bool, str]: (is_valid, encoding or error_message)
        
    Raises:
        FileEncodingError: If the file encoding cannot be detected
    """
    # Expanded list of encodings to try
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']
    
    # Check if file exists
    if not os.path.exists(file_path):
        error_msg = f"File does not exist: {file_path}"
        logger.error(error_msg)
        raise FileEncodingError(error_msg, file_path)
    
    # Check if it's actually a file
    if not os.path.isfile(file_path):
        error_msg = f"Path exists but is not a file: {file_path}"
        logger.error(error_msg)
        raise FileEncodingError(error_msg, file_path)
    
    # Check for zero-length files
    try:
        if os.path.getsize(file_path) == 0:
            error_msg = f"File is empty: {file_path}"
            logger.warning(error_msg)
            raise FileEncodingError(error_msg, file_path)
    except OSError as e:
        error_msg = f"Error checking file size: {str(e)}"
        logger.error(error_msg)
        raise FileEncodingError(error_msg, file_path, {"original_error": str(e)})
    
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
            error_msg = f"Permission error reading file: {str(e)}"
            logger.error(error_msg)
            raise FileEncodingError(error_msg, file_path, {"original_error": str(e)})
        except OSError as e:
            error_msg = f"Error reading file: {str(e)}"
            logger.error(error_msg)
            raise FileEncodingError(error_msg, file_path, {"original_error": str(e)})
    
    error_msg = f"Could not decode file with any supported encoding: {file_path}"
    logger.error(error_msg)
    raise FileEncodingError(error_msg, file_path)

def read_file_content(file_path: str, encoding: Optional[str] = None) -> str:
    """
    Read a file's content with the specified encoding.
    
    Args:
        file_path: Path to the file to read
        encoding: Encoding to use, if known
        
    Returns:
        str: File content
        
    Raises:
        FileReadError: If the file cannot be read
    """
    try:
        # If encoding is not provided, detect it
        if encoding is None:
            _, encoding = detect_file_encoding(file_path)
        
        # Open and read the file
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
            
        logger.debug(f"Read {len(content)} characters from {file_path} with encoding {encoding}")
        return content
    except FileEncodingError as e:
        # Re-raise the original encoding error
        raise e
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(error_msg)
        raise FileReadError(error_msg, file_path, {"original_error": str(e)})

def get_all_document_files(data_folder: str, recursive: bool = True) -> Set[str]:
    """
    Get all document files in the data folder.
    
    Args:
        data_folder: Folder to search in
        recursive: Whether to search in subdirectories
        
    Returns:
        Set[str]: Set of absolute file paths
    """
    all_files = set()
    
    for ext in settings.FILE_EXTENSIONS:
        if recursive:
            pattern = os.path.join(data_folder, f"**/*{ext}")
            all_files.update(os.path.abspath(f) for f in glob.glob(pattern, recursive=True))
        else:
            pattern = os.path.join(data_folder, f"*{ext}")
            all_files.update(os.path.abspath(f) for f in glob.glob(pattern))
    
    logger.info(f"Found {len(all_files)} document files in {data_folder}")
    return all_files

def get_metadata_for_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a file from its corresponding .metadata.json file.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Optional[Dict[str, Any]]: Metadata dictionary or None if no metadata file exists
        
    Raises:
        MetadataError: If metadata file exists but cannot be parsed
    """
    base_name, ext = os.path.splitext(file_path)
    metadata_path = f"{base_name}.metadata.json"
    
    if not os.path.exists(metadata_path):
        logger.debug(f"No metadata file found for {file_path}")
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as meta_file:
            metadata = json.load(meta_file)
            logger.info(f"Loaded metadata from {metadata_path}")
            return metadata
    except json.JSONDecodeError as e:
        error_msg = f"Error parsing metadata JSON from {metadata_path}: {str(e)}"
        logger.error(error_msg)
        raise MetadataError(error_msg, file_path, {"metadata_path": metadata_path, "original_error": str(e)})
    except Exception as e:
        error_msg = f"Error loading metadata from {metadata_path}: {str(e)}"
        logger.error(error_msg)
        raise MetadataError(error_msg, file_path, {"metadata_path": metadata_path, "original_error": str(e)})

def get_relative_path(file_path: str, base_folder: str) -> str:
    """
    Get the relative path of a file from a base folder.
    
    Args:
        file_path: Absolute path to the file
        base_folder: Base folder to compute the relative path from
        
    Returns:
        str: Relative path
    """
    abs_base_folder = os.path.abspath(base_folder)
    abs_file_path = os.path.abspath(file_path)
    
    # Make sure file is within base folder
    if abs_file_path.startswith(abs_base_folder):
        rel_path = os.path.relpath(abs_file_path, abs_base_folder)
        return rel_path
    else:
        # Fallback to basename if the file is not within base folder
        logger.warning(f"File {file_path} is not within base folder {base_folder}")
        return os.path.basename(file_path)