# Document Chat System Setup Guide

## Introduction

This guide outlines the setup for a document-based chat system with focus on:
- European data privacy compliance
- Easy deployment and testing
- Handling large document collections
- Simple prototype-to-production path

After evaluating multiple options across aspects like privacy, maintenance, scalability, and costs, we selected:
- Weaviate (Dutch) for vector database
- Mistral AI (French) for LLM services
- FastAPI for backend
- Streamlit for prototype frontend (Production on Hetzner)

Key advantages:
- Full EU data sovereignty
- Self-hostable components
- Scalable to hundreds of users
- Simple Python-based deployment
- Docker containerization for easy testing

The setup below provides a foundation for local development with path to EU-compliant cloud deployment.

## Project Structure
```
doc-chat/
├── docker-compose.yml
├── .env
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── routes/
│   └── data/
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```

## Technology Stack
- FastAPI: Backend API
- Streamlit: Frontend (temporary for prototype)
- Weaviate: Vector Database
- Mistral API: LLM Provider
- Docker: Containerization

## Setup Steps

1. **Environment Setup**
```bash
# Create project structure
mkdir doc-chat && cd doc-chat
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. **Install Dependencies**
```bash
# Backend requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
python-dotenv==1.0.0
weaviate-client==3.24.1
httpx==0.25.1

# Frontend requirements.txt
streamlit==1.28.1
```

3. **Environment Variables (.env)**
```
MISTRAL_API_KEY=your_key_here
WEAVIATE_URL=http://weaviate:8080
```

4. **Docker Compose**
```yaml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:1.21.2
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
    volumes:
      - weaviate_data:/var/lib/weaviate

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - weaviate

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    env_file:
      - .env
    depends_on:
      - backend

volumes:
  weaviate_data:
```

## Initial Code

### Backend (main.py)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import os
import httpx

app = FastAPI()

# Initialize Weaviate client
client = weaviate.Client(
    url=os.getenv("WEAVIATE_URL")
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(query: Query):
    try:
        # 1. Search relevant documents
        result = client.query.get(
            "Document", ["content"]
        ).with_near_text({
            "concepts": [query.question]
        }).do()
        
        # 2. Get context from search results
        context = result['data']['Get']['Document']
        
        # 3. Query Mistral
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"},
                json={
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system", "content": "Answer based on the context provided."},
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query.question}"}
                    ]
                }
            )
        
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, str(e))
```

### Frontend (app.py)
```python
import streamlit as st
import httpx

st.title("Document Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://backend:8000/chat",
                json={"question": prompt}
            )
            response_data = response.json()
            st.markdown(response_data["choices"][0]["message"]["content"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["choices"][0]["message"]["content"]
            })
```

## Running the Project

1. Start services:
```bash
docker-compose up --build
```

2. Access:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- Weaviate Console: http://localhost:8080

## Next Steps
1. Add document upload functionality
2. Implement authentication
3. Add response caching
4. Improve error handling
5. Add conversation history
6. Deploy to EU-based VPS
