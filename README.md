# EU-Compliant Document Chat System

A GDPR-compliant Retrieval-Augmented Generation (RAG) system designed for academic environments to securely query document collections.

![Simplified Architecture](docs/diagrams/architecture-diagram.svg)

## Key Features

- **EU Data Sovereignty**: All components comply with EU data protection regulations
- **Simple Document Management**: Add text files to a watched folder for automatic processing
- **Metadata Support**: Include bibliographic data for academic publications and other documents
- **Natural Language Querying**: Ask questions about your documents in natural language
- **Source Citations**: All answers include references to source documents
- **GDPR Compliance**: Built with privacy by design principles
- **Enhanced Security**:
  - Authentication for web interface
  - Request validation and sanitization
  - API key rotation mechanisms
  - Docker Secrets for credential management
  - Network isolation between components
  - Security headers via reverse proxy
  - Rate limiting and abuse prevention
  
## Technology Stack

- **Vector Database**: Weaviate (Netherlands-based)
- **LLM Provider**: Mistral AI (France-based)
- **Backend**: FastAPI (Python)
- **Frontend** Vue.js + Nginx
- **Deployment**: Docker containers on Hetzner (German cloud provider)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 4GB of available RAM
- Mistral AI API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ducroq/doc-chat.git
   cd doc-chat
   ```

2. Set up your Mistral AI API key securely using Docker Secrets:
   ```bash
   mkdir -p ./secrets
   echo "your_api_key_here" > ./secrets/mistral_api_key.txt
   chmod 600 ./secrets/mistral_api_key.txt
   ```

3. Start the system:

   - On Windows
   ```bash
   .\start.ps1
   ```

   - On Linux
   ```bash
   chmod +x start.sh stop.sh
   ./start.sh
   ```
   
4. Access the interfaces:
   - Web interface: http://localhost:8081 (served by Nginx)
   - API documentation: http://localhost:8000/docs
   - Weaviate console: http://localhost:8080

### Adding Documents with Metadata

Simply place files in the `data/` directory. The system will automatically process and index them.

1. Place your `.txt` files in the `data/` directory
2. For each text file, create a corresponding metadata file with the same base name:
   ```
   data/
   example.txt
   example.metadata.json
   ```
3. Format the metadata file using a Zotero-inspired schema:
   ```json
   {
   "itemType": "journalArticle",
   "title": "Example Paper Title",
   "creators": [
      {"firstName": "John", "lastName": "Smith", "creatorType": "author"}
   ],
   "date": "2023",
   "publicationTitle": "Journal Name",
   "tags": ["tag1", "tag2"]
   }
   ```
   
The system will automatically associate metadata with documents and display it when providing answers.

### Authentication System

The system includes a secure authentication system:

- JWT-based authentication for API and web interfaces
- User management via command-line tool
- Bcrypt password hashing
- Role-based access control

To set up initial authentication after installation:

```bash
# Create a JWT secret key
openssl rand -hex 32 > ./secrets/jwt_secret_key.txt
chmod 600 ./secrets/jwt_secret_key.txt

# Create an admin user 
python manage_users.py create admin --generate-password --admin
```

For detailed information on authentication, see the [Authentication System Documentation](docs/security.md#authentication-system).

## Research & Analytics Features

- **Chat Logging**: Optional logging of interactions for research purposes
- **Privacy-First Design**: GDPR-compliant with anonymization and automatic data retention policies
- **Transparent Processing**: Clear user notifications when logging is enabled

## Documentation

For more detailed information about the system, check the following documentation:

- [Architecture Overview](docs/architecture.md)
- [Authentication](docs/authentication.md)
- [Deployment Guide](docs/deployment-guide.md)
- [User Guide](docs/user-guide.md)
- [Developer Guide](docs/developer-guide.md)
- [Security](docs/security.md)
- [Privacy Notice](docs/privacy-notice.md)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


