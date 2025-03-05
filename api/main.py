import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import httpx
from dotenv import load_dotenv

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

# Create Weaviate client
try:
    client = weaviate.Client(weaviate_url)
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
        "mistral_api": "configured" if mistral_api_key else "not configured"
    }

@app.post("/search")
async def search_documents(query: Query):
    """Search for relevant document chunks without LLM generation."""
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    try:
        # Search Weaviate for relevant chunks
        result = client.query.get(
            "DocumentChunk", ["content", "filename", "chunkId"]
        ).with_near_text({
            "concepts": [query.question]
        }).with_limit(5).do()
        
        if "errors" in result:
            logger.error(f"Weaviate query error: {result['errors']}")
            raise HTTPException(status_code=500, detail="Error querying the vector database")
        
        chunks = result.get("data", {}).get("Get", {}).get("DocumentChunk", [])
        
        return {
            "query": query.question,
            "results": chunks
        }
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat")
async def chat(query: Query):
    """RAG-based chat endpoint that queries documents and generates a response."""
    if not client:
        raise HTTPException(status_code=503, detail="Weaviate connection not available")
    
    if not mistral_api_key:
        raise HTTPException(status_code=503, detail="Mistral API key not configured")
    
    try:
        # Search Weaviate for relevant chunks
        search_result = client.query.get(
            "DocumentChunk", ["content", "filename", "chunkId"]
        ).with_near_text({
            "concepts": [query.question]
        }).with_limit(3).do()
        
        chunks = search_result.get("data", {}).get("Get", {}).get("DocumentChunk", [])
        
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            }
        
        # Format context from chunks
        context = "\n\n".join([chunk["content"] for chunk in chunks])
        
        # Format sources for citation
        sources = [{"filename": chunk["filename"], "chunkId": chunk["chunkId"]} 
                   for chunk in chunks]
        
        # Send prompt to Mistral AI
        async with httpx.AsyncClient() as http_client:
            mistral_response = await http_client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {mistral_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistral-tiny",
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that answers questions based on the provided document context. "
                                      "Stick to the information in the context. If you don't know the answer, say so."
                        },
                        {
                            "role": "user", 
                            "content": f"Context:\n{context}\n\nQuestion: {query.question}"
                        }
                    ],
                    "temperature": 0.7
                },
                timeout=30.0
            )
            
            if mistral_response.status_code != 200:
                logger.error(f"Mistral API error: {mistral_response.text}")
                raise HTTPException(status_code=mistral_response.status_code, 
                                    detail="Error communicating with LLM service")
            
            response_data = mistral_response.json()
            answer = response_data["choices"][0]["message"]["content"]
            
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