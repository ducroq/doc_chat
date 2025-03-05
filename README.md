# RAG System with Folder Watching

## Introduction

EU-compliant document-based Retrieval-Augmented Generation (RAG) system designed for academic environments:
- Admins add text (.txt) files to a watched folder
- Documents are automatically processed and indexed
- End-users can query the documents through a web interface

By using Docker and standardized components, the system can be easily deployed to different environments while maintaining consistency and compliance.

### Key design decisions

After evaluating multiple options across aspects like privacy, maintenance, scalability, and costs, we selected:

- **Weaviate** (Dutch) for vector database - providing EU-based, self-hostable vector search
- **Mistral AI** (French) for LLM services - offering EU-compliant language model capabilities
- **FastAPI** for backend - enabling efficient API development with Python
- **Streamlit** for prototype frontend - allowing rapid UI development (production on Hetzner)
- **Docker** for containerization - ensuring consistent deployment across environments

### Key advantages

- Full EU data sovereignty
- GDPR compliance by design
- Open source components where possible
- Scalable to hundreds of users
- Simple Python-based deployment
- Text-only document processing for simplicity

## Technology stack

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

## Project structure

```
doc-chat/
├── docker-compose.yml
├── .env
├── data/                  # Watched folder for text documents
├── docs/                  # Project documentation
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

## Architecture

![Simplified RAG System with Folder-Based Document Ingestion](docs/architecture-diagram.svg)

### Class diagram

```mermaid
classDiagram
    class DocumentProcessor {
        +process_documents()
        +watch_folder()
        +handle_file_change(event)
        +chunk_text(text, max_size, overlap)
        +add_document_to_weaviate(file_path)
    }
    
    class FileWatcher {
        +start()
        +stop()
        +on_created(event)
        +on_modified(event)
    }
    
    class TextChunker {
        +chunk_text(text, max_size, overlap)
        +create_chunks_from_paragraphs()
    }
    
    class WeaviateClient {
        +setup_schema()
        +add_chunk(chunk_data, uuid)
        +search_similar(query, limit)
        +get_document_by_id(id)
    }
    
    class APIService {
        +chat_endpoint(query)
        +search_documents(query)
        +generate_response(context, query)
    }
    
    class WebServer {
        +serve_static_files()
        +proxy_api_requests()
    }
    
    class OllamaClient {
        +generate_completion(prompt, model)
        +get_models()
    }
    
    class Document {
        +String filename
        +String content
        +DateTime added_date
    }
    
    class DocumentChunk {
        +String content
        +String filename
        +Int chunk_id
        +Vector embedding
    }
    
    class UserQuery {
        +String question
        +process()
    }
    
    class SystemResponse {
        +String answer
        +Array sources
    }
    
    DocumentProcessor --> FileWatcher : uses
    DocumentProcessor --> TextChunker : uses
    DocumentProcessor --> WeaviateClient : uses
    DocumentProcessor --> Document : processes
    Document --> DocumentChunk : split into
    
    APIService --> WeaviateClient : searches
    APIService --> OllamaClient : prompts
    APIService --> UserQuery : handles
    APIService --> SystemResponse : produces
    
    WebServer --> APIService : forwards requests to
```

### Document processing flow

```mermaid
sequenceDiagram
    actor Admin
    participant Folder as Watched Folder
    participant Processor as Document Processor
    participant Chunker as Text Chunker
    participant Weaviate as Vector Database
    
    Admin->>Folder: Add .txt file
    Folder->>Processor: File change event
    Processor->>Processor: Read file
    Processor->>Chunker: Send text content
    Chunker->>Chunker: Split into chunks
    loop For each chunk
        Processor->>Weaviate: Store chunk with metadata
        Weaviate->>Weaviate: Generate embedding
        Weaviate->>Weaviate: Index chunk
        Weaviate-->>Processor: Confirm storage
    end
    Processor-->>Folder: Continue watching
    
    Note over Weaviate: Document is now searchable
```

### Query processing flow

```mermaid
sequenceDiagram
    actor User
    participant WebUI as Web Interface
    participant API as API Service
    participant Weaviate as Vector Database
    participant Ollama as LLM Service
    
    User->>WebUI: Ask question
    WebUI->>API: POST /chat with question
    API->>Weaviate: Search for relevant chunks
    Weaviate->>Weaviate: Perform vector similarity search
    Weaviate-->>API: Return top matching chunks
    
    API->>API: Format chunks as context
    API->>Ollama: Send prompt with context
    Ollama->>Ollama: Generate response
    Ollama-->>API: Return generated text
    
    API->>API: Format final response with sources
    API-->>WebUI: Return JSON response
    WebUI-->>User: Display answer and sources
```

### RAG system component diagram

```mermaid
flowchart TD
    subgraph Host["Host Machine"]
        subgraph DocProc["Document Processor"]
            FileWatcher["File Watcher\n(watchdog)"]
            TextReader["Text Reader"]
            TextChunker["Text Chunker"]
            WeaviateClient1["Weaviate Client"]
        end
        
        subgraph Docker["Docker Environment"]
            subgraph VectorDB["Vector Database"]
                Weaviate["Weaviate"]
                TextVectorizer["Text2Vec Transformer"]
            end
            
            subgraph APIService["API Service"]
                FastAPI["FastAPI"]
                Endpoints["Query Endpoints"]
                WeaviateClient2["Weaviate Client"]
                LLMClient["LLM Client"]
            end
            
            subgraph WebInterface["Web Interface"]
                Nginx["Nginx Server"]
                StaticFiles["Static Files"]
                ChatUI["Chat UI"]
                APIProxy["API Proxy"]
            end
        end
        
        subgraph Ollama["LLM Service (Ollama)"]
            LlamaModel["Llama Model"]
        end
        
        DocsFolder["/docs Folder"]
    end
    
    User["End User"]
    Admin["Admin"]
    
    %% Connections
    Admin -->|"adds .txt files"| DocsFolder
    FileWatcher -->|"monitors"| DocsFolder
    FileWatcher -->|"file event"| TextReader
    TextReader -->|"file content"| TextChunker
    TextChunker -->|"text chunks"| WeaviateClient1
    WeaviateClient1 -->|"store chunks"| Weaviate
    Weaviate <-->|"generates embeddings"| TextVectorizer
    
    User -->|"interacts with"| ChatUI
    ChatUI -->|"submits query"| APIProxy
    APIProxy -->|"forwards request"| Endpoints
    Endpoints -->|"searches"| WeaviateClient2
    WeaviateClient2 -->|"vector search"| Weaviate
    Endpoints -->|"prompts"| LLMClient
    LLMClient -->|"completion request"| LlamaModel
    LlamaModel -->|"generated text"| LLMClient
    LLMClient -->|"response"| Endpoints
    Endpoints -->|"JSON response"| APIProxy
    APIProxy -->|"display answer"| ChatUI
    
    %% Styling
    classDef process fill:#d1e7dd,stroke:#198754,stroke-width:1px
    classDef storage fill:#e2e3e5,stroke:#212529,stroke-width:1px
    classDef service fill:#cfe2ff,stroke:#0d6efd,stroke-width:1px
    classDef ui fill:#f8d7da,stroke:#dc3545,stroke-width:1px
    classDef user fill:#fff3cd,stroke:#664d03,stroke-width:1px
    classDef llm fill:#e0cffc,stroke:#6f42c1,stroke-width:1px
    
    class DocProc,TextReader,TextChunker,FileWatcher,WeaviateClient1,WeaviateClient2,LLMClient,Endpoints process
    class Weaviate,TextVectorizer,DocsFolder storage
    class APIService,FastAPI service
    class WebInterface,Nginx,StaticFiles,ChatUI,APIProxy ui
    class User,Admin user
    class Ollama,LlamaModel llm
```

### RAG system deployment diagram

```mermaid
flowchart TD
    subgraph Host["Host Machine"]
        subgraph DockerEnv["Docker Environment"]
            subgraph ProcessorContainer["processor container"]
                DocProcessor["Document Processor\n(Python)"]
            end
            
            subgraph WeaviateContainer["weaviate container"]
                Weaviate["Weaviate\n(Vector Database)"]
            end
            
            subgraph TransformerContainer["t2v-transformers container"]
                TextVectorizer["Text2Vec Transformer\n(NLP Model)"]
            end
            
            subgraph APIContainer["api container"]
                FastAPI["FastAPI Service\n(Python)"]
            end
            
            subgraph WebContainer["web container"]
                Nginx["Nginx\n(Web Server)"]
                HTML["Static HTML/JS/CSS"]
            end
        end
        
        OllamaService["Ollama\n(Running on Host)"]
        DocsVolume["./docs\n(Mounted Volume)"]
        WeaviateVolume["weaviate_data\n(Docker Volume)"]
    end
    
    Browser["User's Browser"]
    
    %% Connections
    DocProcessor -->|"watches"| DocsVolume
    DocProcessor -->|"stores embeddings"| Weaviate
    Weaviate -->|"computes embeddings"| TextVectorizer
    Weaviate <-->|"persists data"| WeaviateVolume
    
    FastAPI -->|"searches"| Weaviate
    FastAPI -->|"prompt completion"| OllamaService
    
    Browser -->|"HTTP/HTTPS"| Nginx
    Nginx -->|"serves"| HTML
    Nginx -->|"proxies API calls"| FastAPI
    
    %% Styling
    classDef container fill:#f5f5f5,stroke:#333,stroke-width:1px
    classDef volume fill:#fff3cd,stroke:#664d03,stroke-width:1px
    classDef service fill:#cfe2ff,stroke:#0d6efd,stroke-width:1px
    classDef external fill:#f8d7da,stroke:#dc3545,stroke-width:1px
    
    class DockerEnv,ProcessorContainer,WeaviateContainer,TransformerContainer,APIContainer,WebContainer container
    class DocsVolume,WeaviateVolume volume
    class DocProcessor,Weaviate,TextVectorizer,FastAPI,Nginx,HTML,OllamaService service
    class Browser external
```

## Docker compose

Docker Compose is a tool that helps you define and manage multi-container Docker applications. In the doc chat system, the docker-compose.yml file defines three services:

- A Weaviate vector database container
- A FastAPI backend container
- A Streamlit frontend container

The file also specifies how these containers connect to each other, what ports they expose, and what environment variables they use. For instance, your backend depends on Weaviate, and your frontend depends on the backend.

### Running the system

On Windows, start the system by runnning the Power Shell script:

```bash
.\start.ps1
```
which builds and starts all three containers together. 

Access:
- Web interface: http://localhost:8501
- API documentation: http://localhost:8000/docs
- Weaviate console: http://localhost:8080

