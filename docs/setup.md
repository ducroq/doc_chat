# EU-Compliant RAG System Setup Guide

## Introduction

This guide outlines the setup for an EU-compliant document-based Retrieval-Augmented Generation (RAG) system designed for academic environments. The system enables users to query document collections through a chat interface while maintaining full EU data sovereignty.

### Key Design Decisions

After evaluating multiple options across aspects like privacy, maintenance, scalability, and costs, we selected:

- **Weaviate** (Dutch) for vector database - providing EU-based, self-hostable vector search
- **Mistral AI** (French) for LLM services - offering EU-compliant language model capabilities
- **FastAPI** for backend - enabling efficient API development with Python
- **Streamlit** for prototype frontend - allowing rapid UI development (production on Hetzner)
- **Docker** for containerization - ensuring consistent deployment across environments

Key advantages:
- Full EU data sovereignty
- GDPR compliance by design
- Open source components where possible
- Scalable to hundreds of users
- Simple Python-based deployment
- Text-only document processing for simplicity

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vector Database** | Weaviate | Stores and searches document embeddings |
| **Text Embeddings** | text2vec-transformers | Converts text to vector embeddings |
| **LLM Provider** | Mistral AI | Generates responses based on retrieved context |
| **Backend API** | FastAPI | Handles requests, orchestrates RAG process |
| **Document Processor** | Python/watchdog | Monitors and processes new text documents |
| **Prototype Frontend** | Streamlit | Provides user chat interface |
| **Production Frontend** | HTML/JS + Nginx | Lightweight production web interface |
| **Containerization** | Docker | Packages all components for deployment |
| **Production Hosting** | Hetzner | EU-based cloud provider for production |

## Project Structure

```
rag-system/
├── docker-compose.yml
├── .env
├── docs/                  # Watched folder for text documents
├── processor/             # Document processor service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── processor.py
├── api/                   # API service
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── web-prototype/         # Streamlit prototype
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
└── web-production/        # Production web interface
    ├── Dockerfile
    ├── static/
    ├── index.html
    └── nginx.conf
```

## Setup Steps

### 1. Prerequisites

- Docker and Docker Compose installed
- Mistral AI API key (register at https://mistral.ai/)
- Git for version control
- Basic understanding of Python and Docker

### 2. Install Dependencies

Create a new project and set up the environment:

```bash
# Create project directories
mkdir -p rag-system/{processor,api,web-prototype,web-production,docs,web-production/static}
cd rag-system

# Initialize git repository
git init
echo "docs/" >> .gitignore
echo ".env" >> .gitignore
```

### 3. Environment Variables

Create a `.env` file in the project root:

```
# Mistral AI Configuration
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-medium  # or another appropriate model

# Weaviate Configuration
WEAVIATE_URL=http://weaviate:8080

# Document Processor Configuration
DOCS_DIR=/docs
```

### 4. Docker Compose Configuration

Create a `docker-compose.yml` file in the project root with the following services:

- **weaviate**: Vector database
- **t2v-transformers**: Text embedding service
- **processor**: Document processor service
- **api**: FastAPI backend
- **web**: Web interface (Streamlit for prototype, Nginx for production)

The volumes should include:
- `./docs:/docs`: For document monitoring
- `weaviate_data`: Persistent storage for vector database

### 5. Document Processor Setup

The document processor should:
- Watch the `/docs` folder for new or modified text files
- Process text files by splitting them into chunks
- Add chunks to Weaviate with appropriate metadata
- Set up a proper schema in Weaviate

### 6. API Service Setup

The API service should:
- Provide endpoints for querying documents
- Search Weaviate for relevant content
- Format retrieved content for Mistral AI
- Send properly formatted prompts to Mistral AI
- Return responses with source references

### 7. Web Interface Setup

For the prototype:
- Build a simple Streamlit chat interface
- Display conversation history
- Show source references for answers

For production:
- Create a lightweight HTML/JS interface
- Set up Nginx to serve static files and proxy API requests

### 8. Running the System

Start the system with:

```bash
docker-compose up --build
```

Access:
- Web interface: http://localhost:80
- API documentation: http://localhost:8000/docs
- Weaviate console: http://localhost:8080

## Next Steps

1. **Document Processing Enhancements**:
   - Add support for other text formats
   - Improve chunking strategies for better retrieval
   - Add metadata extraction

2. **User Experience Improvements**:
   - Enhance source citation format
   - Add filtering options for documents
   - Implement conversation memory

3. **Production Deployment**:
   - Deploy to Hetzner or another EU-based cloud
   - Set up monitoring and logging
   - Implement backup strategy for Weaviate data

4. **Performance Optimization**:
   - Add response caching to reduce Mistral API costs
   - Optimize chunk size and retrieval parameters
   - Implement batching for document processing

5. **Additional Features**:
   - Add document metadata search
   - Implement simple analytics on usage
   - Create admin dashboard for system monitoring

## Conclusion

This setup provides a foundation for an EU-compliant RAG system suitable for academic environments. The system respects data sovereignty requirements while providing a user-friendly interface for document queries.

By using Docker and standardized components, the system can be easily deployed to different environments while maintaining consistency and compliance.