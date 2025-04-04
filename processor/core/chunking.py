"""
Text chunking module for document processor.
Implements various strategies for splitting text into manageable chunks.
"""
import re
from typing import List, Dict, Any, Optional, Tuple

from config import settings
from utils.logging_config import get_logger
from utils.errors import ChunkingError

logger = get_logger(__name__)

class TextChunk(Dict[str, Any]):
    """
    Type definition for a text chunk with content and metadata.
    
    Keys:
        content (str): The text content of the chunk
        page (Optional[int]): Page number
        heading (Optional[str]): Section heading
        level (Optional[int]): Heading level
    """
    pass

def chunk_text(
    text: str,
    max_chunk_size: int = settings.CHUNK_SIZE,
    overlap: int = settings.CHUNK_OVERLAP,
    chunking_strategy: str = settings.CHUNKING_STRATEGY
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
        
    Raises:
        ChunkingError: If chunking fails
    """
    try:
        # Input validation
        if not text:
            logger.warning("Empty text provided to chunk_text")
            return []
        
        if max_chunk_size <= 0:
            logger.warning(f"Invalid max_chunk_size: {max_chunk_size}, using default")
            max_chunk_size = settings.CHUNK_SIZE
            
        if overlap < 0 or overlap >= max_chunk_size:
            logger.warning(f"Invalid overlap: {overlap}, using default")
            overlap = settings.CHUNK_OVERLAP
        
        # Extract page markers
        pages = _extract_pages(text)
        
        # Process sections based on headings
        sections = _extract_sections(pages)
        
        # Apply chunking strategy
        chunking_func = _get_chunking_strategy(chunking_strategy)
        chunks = chunking_func(sections, max_chunk_size, overlap)
        
        logger.info(f"Generated {len(chunks)} chunks using {chunking_strategy} strategy")
        return chunks
        
    except Exception as e:
        error_msg = f"Error chunking text: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        raise ChunkingError(error_msg, {"original_error": str(e)})

def _extract_pages(text: str) -> List[Tuple[str, int]]:
    """
    Extract pages from text based on HTML comments.
    
    Args:
        text: The text to process
        
    Returns:
        List[Tuple[str, int]]: List of (text_content, page_number) tuples
    """
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
    
    return pages

def _extract_sections(pages: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
    """
    Extract sections based on Markdown headings.
    
    Args:
        pages: List of (text_content, page_number) tuples
        
    Returns:
        List[Dict[str, Any]]: List of section dictionaries
    """
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
    
    return sections

def _get_chunking_strategy(strategy_name: str):
    """
    Get chunking strategy function based on name.
    
    Args:
        strategy_name: Name of the chunking strategy
        
    Returns:
        Callable: Chunking strategy function
    """
    strategies = {
        "simple": _simple_chunking,
        "paragraph": _paragraph_chunking,
        "section": _section_chunking,
        "semantic": _semantic_chunking  # Placeholder for future implementation
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Unknown chunking strategy: {strategy_name}, using section strategy")
        return strategies["section"]
    
    return strategies[strategy_name]

def _simple_chunking(sections: List[Dict[str, Any]], max_chunk_size: int, overlap: int) -> List[TextChunk]:
    """
    Simple character-based chunking without respecting structure.
    
    Args:
        sections: List of section dictionaries
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[TextChunk]: List of chunks with metadata
    """
    chunks = []
    
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
                
    return chunks

def _paragraph_chunking(sections: List[Dict[str, Any]], max_chunk_size: int, overlap: int) -> List[TextChunk]:
    """
    Paragraph-aware chunking that respects paragraph boundaries.
    
    Args:
        sections: List of section dictionaries
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[TextChunk]: List of chunks with metadata
    """
    chunks = []
    
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
                
    return chunks

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling multiple languages.
    
    Args:
        text: Text to split into sentences
        
    Returns:
        List[str]: List of sentences
    """
    # This pattern works for many European languages
    # It looks for periods, question marks, or exclamation points
    # followed by spaces and capital letters
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    return sentences

def _section_chunking(sections: List[Dict[str, Any]], max_chunk_size: int, overlap: int) -> List[TextChunk]:
    """
    Section-based chunking with sentence awareness, preserving section context.
    
    Args:
        sections: List of section dictionaries
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[TextChunk]: List of chunks with metadata
    """
    chunks = []
    
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
            sentences = _split_into_sentences(paragraph)
            
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

def _semantic_chunking(sections: List[Dict[str, Any]], max_chunk_size: int, overlap: int) -> List[TextChunk]:
    """
    Placeholder for semantic chunking using AI.
    Currently falls back to section chunking.
    
    Args:
        sections: List of section dictionaries
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[TextChunk]: List of chunks with metadata
    """
    logger.warning("Semantic chunking not yet implemented, falling back to section chunking")
    return _section_chunking(sections, max_chunk_size, overlap)