import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from dotenv import load_dotenv
from mistralai import Mistral

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
        response = collection.query.near_text(
            query=query.question,
            limit=5,
            return_properties=["content", "filename", "chunkId"]
        ).do()
        
        # Format results
        results = []
        for obj in response.objects:
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
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    if not mistral_client:
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
        
        # Format context from chunks
        context = "\n\n".join([obj.properties["content"] for obj in search_result.objects])
        
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
            
        return {
            "answer": answer,
            "sources": sources
        }
            
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
