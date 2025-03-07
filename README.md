# EU-Compliant Document Chat System

A GDPR-compliant Retrieval-Augmented Generation (RAG) system designed for academic environments to securely query document collections.

![Simplified Architecture](docs/diagrams/architecture-diagram.svg)

## Key Features

- **EU Data Sovereignty**: All components comply with EU data protection regulations
- **Simple Document Management**: Add text files to a watched folder for automatic processing
- **Natural Language Querying**: Ask questions about your documents in natural language
- **Source Citations**: All answers include references to source documents
- **GDPR Compliance**: Built with privacy by design principles

## Technology Stack

- **Vector Database**: Weaviate (Netherlands-based)
- **LLM Provider**: Mistral AI (France-based)
- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit (prototype) and Nginx/HTML/JS (production)
- **Deployment**: Docker containers on Hetzner (German cloud provider)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB of available RAM
- Mistral AI API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/doc-chat.git
   cd doc-chat
   ```

2. Create a `.env` file with your Mistral AI credentials:
   ```
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL=mistral-tiny
   MISTRAL_DAILY_TOKEN_BUDGET=10000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   ```

3. Start the system:
   ```bash
   # On Windows
   .\start.ps1
   
   # On Linux/macOS
   docker-compose up -d
   ```

4. Access the interfaces:
   - Web interface: http://localhost:8501
   - API documentation: http://localhost:8000/docs
   - Weaviate console: http://localhost:8080

### Adding Documents

Simply place .txt files in the `data/` directory. The system will automatically process and index them.

## Documentation

For more detailed information about the system, check the following documentation:

- [Architecture Overview](docs/architecture.md)
- [Deployment Guide](docs/deployment-guide.md)
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


