# Developer Guide

This guide provides technical information for developers who want to understand, modify, or extend the EU-Compliant Document Chat system.

## Architecture Overview

The system follows a modular architecture with several key components:

1. **Document Processor**: Monitors a folder for text files, chunks text, and indexes in Weaviate
2. **Vector Database**: Weaviate stores document chunks with vector embeddings
3. **API Service**: FastAPI-based service that handles queries and orchestrates the RAG workflow
4. **Web Interface**: Streamlit prototype and production web frontend
5. **LLM Integration**: Mistral AI provides language model capabilities

For a visual representation, refer to the architecture diagrams in the `docs/diagrams` directory.

## Development Environment Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git
- Mistral AI API key

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ducroq/doc-chat.git
   cd doc-chat
   ```

2. **Set up environment**:
   ```bash
   # Create a virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies for local development
   pip install -r api/requirements.txt
   pip install -r processor/requirements.txt
   pip install -r web-prototype/requirements.txt
   ```

3. **Configure environment variables**:
   Create a `.env` file with:
   ```
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_MODEL=mistral-tiny
   MISTRAL_DAILY_TOKEN_BUDGET=10000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   ENABLE_CHAT_LOGGING=false
   ANONYMIZE_CHAT_LOGS=true
   LOG_RETENTION_DAYS=30
   CHAT_LOG_DIR=chat_data
   ```

4. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

### Secure Secrets Management with Docker Secrets

For improved security, use Docker Secrets instead of environment variables for sensitive information:

1. **Create a secrets directory and files**:
   ```bash
   mkdir -p ./secrets
   echo "your_mistral_api_key_here" > ./secrets/mistral_api_key.txt
   chmod 600 ./secrets/mistral_api_key.txt   


### Direct Component Development

If you want to develop or debug individual components:

#### API Service
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Document Processor
```bash
cd processor
python processor.py
```

#### Web Prototype
```bash
cd web-prototype
streamlit run app.py
```

## Key Components and Code Structure

### Document Processor (`processor.py`)

The processor is responsible for:
- Watching a folder for new/modified/deleted text files
- Chunking text into manageable segments
- Creating vector embeddings via Weaviate
- Tracking processed files to avoid redundant processing

Key classes:
- `DocumentStorage`: Handles interaction with Weaviate
- `ProcessingTracker`: Tracks processed files and their timestamps
- `DocumentProcessor`: Processes text files into chunks
- `TextFileHandler`: Watches folder and triggers processing

### API Service (`main.py`)

The API service provides:
- RESTful endpoints for queries and search
- RAG implementation using Weaviate and Mistral AI
- Rate limiting and token budget management
- Response caching for performance
- Chat logging for research purposes

Key endpoints:
- `/status`: Check system status
- `/search`: Search documents without LLM generation
- `/chat`: Full RAG endpoint with LLM-generated responses
- `/privacy`: Serves the privacy notice
- `/documents/count` and `/statistics`: System information

### Vector Database Schema

The system uses a single Weaviate collection:

```
DocumentChunk
├─ content: text
├─ filename: text
├─ chunkId: int
└─ metadataJson: text
```

### Chat Logger (`chat_logger.py`)

The chat logger provides privacy-compliant logging for research:
- Anonymization of user identifiers
- Automatic log rotation
- GDPR-compliant retention policies
- Transparent data handling

## Workflows and Processes

### Document Processing Workflow

1. Admin adds `.txt` file to watched folder
2. Processor detects file change
3. Text is chunked into segments
4. Chunks are stored in Weaviate with metadata
5. Vector embeddings are generated automatically by Weaviate

See `docs/workflows/document-processing.md` for detailed sequence diagram.

### Query Processing Workflow

1. User submits question through interface
2. API converts query to vector embedding
3. Weaviate performs similarity search
4. Relevant chunks are retrieved
5. Context and query are sent to Mistral AI
6. Response is generated and returned with sources

See `docs/workflows/query-processing.md` for detailed sequence diagram.

## Converting Documents to Markdown

When converting PDFs or other documents to text files for processing:

1. Use Markdown formatting to preserve document structure:
   ```markdown
   # Document Title
   ## Section 1
   This is the content of section 1...
   ### Subsection 1.1
   More detailed content...
   ## Section 2
   ```

2. Content from the second main section...
   Add page numbers using HTML comments:
   ```markdown
   <!-- page: 1 -->
   # Introduction
   Content from page 1...
   <!-- page: 2 -->
   ## Background
   Content from page 2...
   ```
3. Save the file with a .md extension in the data/ directory

4. Create corresponding metadata files as needed

## Security Features

### Authentication and Authorization

The system includes authentication for the web interface:
- Password-based authentication using bcrypt for secure password hashing
- Login session management using Streamlit session state
- API key-based authorization for API endpoints

### Request Validation

Comprehensive request validation is implemented:
```python
class Query(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    
    @field_validator('question')
    @classmethod
    def validate_question_content(cls, v: str) -> str:
        # Check for script injection patterns
        dangerous_patterns = [
            '<script>', 'javascript:', 'onload=', 'onerror=', 'onclick='
            # ... more patterns
        ]
        # Check for SQL injection patterns
        # Check for command injection patterns
        # Check for excessive special characters
        # ... validation logic
        return v
```

### Secrets Management

The system uses Docker Secrets for managing sensitive credentials:

```bash
# Create the secrets directory
mkdir -p ./secrets

# Add your API key
echo "your_mistral_api_key_here" > ./secrets/mistral_api_key.txt

# Secure the file
chmod 600 ./secrets/mistral_api_key.txt
```

In docker-compose.yml:
```yaml
secrets:
  mistral_api_key:
    file: ./secrets/mistral_api_key.txt
  internal_api_key:
    file: ./secrets/internal_api_key.txt

services:
  api:
    secrets:
      - mistral_api_key
      # ...
```

### API Key Rotation

The system checks secret age to prompt rotation:
```python
def check_secret_age(secret_path, max_age_days=90):
    """Check if a secret file is older than max_age_days"""
    if not os.path.exists(secret_path):
        return False
    
    file_timestamp = os.path.getmtime(secret_path)
    file_age_days = (time.time() - file_timestamp) / (60 * 60 * 24)
    
    if file_age_days > max_age_days:
        logger.warning(f"Secret at {secret_path} is {file_age_days:.1f} days old and should be rotated")
        return False
        
    return True
```

### Reverse Proxy and Security Headers

Nginx is configured with security headers:
```
add_header X-Frame-Options "SAMEORIGIN";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
```

The API also adds security headers:
```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'..."
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response
```

### Rate Limiting and Abuse Prevention

Multiple rate limiting layers protect the system:
```python
@app.middleware("http")
async def rate_limit_by_ip(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    
    # Clean old timestamps
    now = time.time()
    ip_request_counters[client_ip] = [timestamp for timestamp in ip_request_counters[client_ip] 
                                     if now - timestamp < 60]
    
    # Check limits
    if len(ip_request_counters[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Add current timestamp
    ip_request_counters[client_ip].append(now)
    
    # Process request
    return await call_next(request)
```

### Docker Security

Container security is enhanced with:
```yaml
user: "1000:1000"  # Use non-root user
security_opt:
  - no-new-privileges:true
```

### Network Isolation

Services are isolated into frontend and backend networks:
```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

services:
  weaviate:
    networks:
      - backend
  api:
    networks:
      - frontend
      - backend
  web-prototype:
    networks:
      - frontend
```

## Adding New Features

### Adding New Document Types

To add support for new document types (PDF, DOCX, etc.):

1. Create a new processor in the `processor.py` file:
   ```python
   def process_pdf(file_path):
       # PDF processing code
       # Return text content
   ```

2. Update the file event handler to detect new file types:
   ```python
   def on_created(self, event):
       if event.src_path.endswith('.pdf'):
           # Process PDF
   ```

### Modifying the Web Interface

The web interface uses Streamlit for prototyping:

1. Edit `web-prototype/app.py` to modify the prototype interface
2. For production changes, update the files in `web-production/`

### Extending API Capabilities

To add new API endpoints:

1. Add new route to `main.py`:
   ```python
   @app.get("/new-endpoint")
   async def new_endpoint():
       # Implementation
       return {"result": "data"}
   ```

2. Update documentation to reflect new capabilities

## Testing

### Testing Document Processing

1. Add a test file to the `data/` directory
2. Check processor logs:
   ```bash
   docker-compose logs -f processor
   ```
3. Verify document count via API:
   ```bash
   curl http://localhost:8000/documents/count
   ```

### Testing Search Functionality

Use the direct search endpoint to test vector search:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question":"What is GDPR?"}'
```

### Testing RAG Capabilities

Use the chat endpoint to test full RAG functionality:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain how the document processor works"}'
```

## Chat Logger Component

The system includes a privacy-focused chat logging component for research:

### Key Files
- `api/chat_logger.py`: Core logging implementation
- `api/main.py`: Integration with API service
- `privacy_notice.html`: User-facing privacy information

### Configuration

Logging is controlled via environment variables:
- `ENABLE_CHAT_LOGGING`: Master switch (default: false)
- `ANONYMIZE_CHAT_LOGS`: Controls anonymization (default: true)
- `LOG_RETENTION_DAYS`: Automatic deletion period (default: 30)
- `CHAT_LOG_DIR`: Storage location (default: chat_data)

### Log Format

Logs are stored as JSONL files with daily rotation:
```json
{
  "timestamp": "2025-03-09T08:53:44.295",
  "request_id": "abc12345",
  "user_id": "anon_123456789abc",
  "query": "What is GDPR?",
  "response": {
    "answer": "GDPR is...",
    "sources": [...]
  }
}
```

## Deployment and CI/CD

### Docker Build

Build all components:
```bash
docker-compose build
```

Build specific component:
```bash
docker-compose build api
```

### Production Deployment

See `docs/deployment-guide.md` for complete production deployment instructions.

## Troubleshooting

### Common Issues

1. **Weaviate connection issues**:
   - Check if Weaviate container is running
   - Verify network connectivity between containers
   - Ensure schema was created successfully

2. **Document processing failures**:
   - Check file encodings (UTF-8 is recommended)
   - Verify file permissions are correct
   - Look for specific errors in processor logs

3. **API errors**:
   - Verify Mistral API key is valid
   - Check token budget and rate limits
   - Monitor API logs for specific error messages

### Debugging Tools

1. **Container logs**:
   ```bash
   docker-compose logs -f [service_name]
   ```

2. **API documentation**:
   Access Swagger UI at `http://localhost:8000/docs`

3. **Weaviate console**:
   Access at `http://localhost:8080`

4. **Test scripts**:
   Use the scripts in the `tests/` directory to verify system functionality

## Contributing

When contributing to this project:

1. Ensure all code follows the established patterns
2. Document new features and changes
3. Update diagrams when modifying the architecture
4. Add tests for new functionality
5. Maintain GDPR compliance and data privacy standards

For detailed contribution guidelines, see the CONTRIBUTING.md file (if available).