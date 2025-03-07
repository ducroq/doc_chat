import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from dotenv import load_dotenv
from mistralai import Mistral
import uuid
import time
from functools import lru_cache
import hashlib

# Simple cache for Mistral responses
@lru_cache(maxsize=100)  # Adjust size as needed
def get_cached_response(query_hash, model):
    # This function will be automatically cached
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="EU-Compliant RAG API")

# Initialize Weaviate client
weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
mistral_model = os.getenv("MISTRAL_MODEL", "mistral-tiny")

# Initialize Mistral client
mistral_client = None
if mistral_api_key:
    try:
        mistral_client = Mistral(api_key=mistral_api_key)
        logger.info("Mistral client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Mistral client: {str(e)}")

# Create Weaviate client
try:
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
    
    # Connect to Weaviate using the same method as the processor
    client = weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=use_https,
        grpc_host=host,
        grpc_port=50051,  # Default gRPC port
        grpc_secure=use_https,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=60, query=30, insert=30)
        )
    )
    logger.info(f"Connected to Weaviate at {weaviate_url}")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {str(e)}")
    client = None

class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "EU-Compliant RAG API is running"}

@app.get("/status")
async def status():
    """Check the status of the API and its connections."""
    weaviate_status = "connected" if client and client.is_ready() else "disconnected"
    
    return {
        "api": "running",
        "weaviate": weaviate_status,
        "mistral_api": "configured" if mistral_client else "not configured"
    }

@app.post("/search")
async def search_documents(query: Query):
    """Search for relevant document chunks without LLM generation."""
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Search Weaviate for relevant chunks
        collection = client.collections.get("DocumentChunk")
        
        search_result = collection.query.near_text(
            query=query.question,
            limit=5,
            return_properties=["content", "filename", "chunkId"]
        )
        
        # Format results
        results = []
        for obj in search_result.objects:
            results.append(obj.properties)
        
        return {
            "query": query.question,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.post("/chat")
async def chat(query: Query):
    """RAG-based chat endpoint that queries documents and generates a response."""
    request_id = str(uuid.uuid4())[:8]  # Generate a short request ID for tracing
    
    logger.info(f"[{request_id}] Chat request received: '{query.question[:50]}...' if len(query.question) > 50 else query.question")

    # Create a hash of the query and context to use as cache key
    query_text = query.question.strip().lower()
    context_hash = hashlib.md5(context.encode()).hexdigest()
    cache_key = f"{query_text}_{context_hash}"
    
    # Check cache first
    cached_result = get_cached_response(cache_key, mistral_model)
    if cached_result:
        logger.info(f"[{request_id}] Cache hit! Returning cached response")
        return cached_result   
    
    if not client:
        logger.error(f"[{request_id}] Weaviate connection not available")
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    if not mistral_client:
        logger.error(f"[{request_id}] Mistral API client not configured")
        raise HTTPException(status_code=503, detail="Mistral API client not configured")
    
    try:
        # Get the collection
        collection = client.collections.get("DocumentChunk")
        
        # Search Weaviate for relevant chunks using v4 API
        search_result = collection.query.near_text(
            query=query.question,
            limit=3,
            return_properties=["content", "filename", "chunkId"]
        )
        
        # Check if we got any results
        if len(search_result.objects) == 0:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
        # Log search results
        logger.info(f"[{request_id}] Retrieved {len(search_result.objects)} relevant chunks")        
        
        # Format context from chunks
        context = "\n\n".join([obj.properties["content"] for obj in search_result.objects])
        logger.info(f"[{request_id}] Context size: {len(context)} characters")

        # Log generation attempt
        logger.info(f"[{request_id}] Sending request to Mistral API using model: {mistral_model}")
        
        start_time = time.time()        

        # Format sources for citation
        sources = [{"filename": obj.properties["filename"], "chunkId": obj.properties["chunkId"]} 
                   for obj in search_result.objects]
        
        # Use Mistral client to generate response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context. Stick to the information in the context. If you don't know the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query.question}"}
        ]
        
        chat_response = mistral_client.chat.complete(
            model=mistral_model,
            messages=messages,
            temperature=0.7,
        )
        
        answer = chat_response.choices[0].message.content

        generation_time = time.time() - start_time
        
        # Log success and timing
        logger.info(f"[{request_id}] Mistral response received in {generation_time:.2f}s")
        logger.info(f"[{request_id}] Answer length: {len(answer)} characters")        

        # Cache the result before returning
        result = {"answer": answer, "sources": sources}
        get_cached_response.cache_clear()  # Clear one entry if needed
        get_cached_response(cache_key, mistral_model)  # Store in cache        
            
        return result
            
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/documents/count")
async def count_documents():
    """Count the number of unique documents indexed in the system."""
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Get the collection
        collection = client.collections.get("DocumentChunk")
        
        # Get all unique filenames
        # Note: In Weaviate v4, we need to use the GroupBy feature
        query_result = collection.query.fetch_objects(
            return_properties=["filename"],
            limit=10000  # Use a reasonably high limit
        )
        
        # Count unique filenames
        unique_filenames = set()
        for obj in query_result.objects:
            unique_filenames.add(obj.properties["filename"])
        
        return {
            "count": len(unique_filenames),
            "documents": list(unique_filenames)
        }
        
    except Exception as e:
        logger.error(f"Error counting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/statistics")
async def get_document_statistics():
    """
    Get comprehensive statistics about documents in the system.
    Returns counts, document metadata, and processing information.
    """
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Get the DocumentChunk collection
        collection = client.collections.get("DocumentChunk")
        
        # 1. Get all objects to gather statistics
        # Limited to 10,000 for practicality - adjust if needed
        query_result = collection.query.fetch_objects(
            return_properties=["filename", "chunkId", "content"],
            limit=10000
        )
        
        if not query_result.objects:
            return {
                "document_count": 0,
                "chunk_count": 0,
                "message": "No documents found in the system"
            }
        
        # 2. Calculate basic statistics
        document_chunks = {}
        total_content_length = 0
        
        for obj in query_result.objects:
            filename = obj.properties["filename"]
            chunk_id = obj.properties["chunkId"]
            content = obj.properties["content"]
            
            # Track chunks per document
            if filename not in document_chunks:
                document_chunks[filename] = []
            document_chunks[filename].append(chunk_id)
            
            # Track total content length
            total_content_length += len(content)
        
        # 3. Prepare document details
        documents = []
        for filename, chunks in document_chunks.items():
            documents.append({
                "filename": filename,
                "chunk_count": len(chunks),
                "first_chunk": min(chunks),
                "last_chunk": max(chunks)
            })
        
        # Sort documents by filename
        documents.sort(key=lambda x: x["filename"])
        
        # 4. Calculate summary statistics
        document_count = len(document_chunks)
        chunk_count = len(query_result.objects)
        avg_chunks_per_doc = chunk_count / max(document_count, 1)
        avg_chunk_length = total_content_length / max(chunk_count, 1)
        
        # 5. Compile and return the statistics
        return {
            "summary": {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "avg_chunks_per_document": round(avg_chunks_per_doc, 2),
                "avg_chunk_length": round(avg_chunk_length, 2),
                "total_content_length": total_content_length,
            },
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Error retrieving document statistics: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
