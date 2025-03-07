# Deployment Guide

This guide provides instructions for deploying the EU-Compliant Document Chat system in various environments.

## Local Deployment

### Prerequisites

- **Windows users**: Update Windows Subsystem for Linux
  ```bash
  wsl --update
- Docker and Docker Compose installed (See [Get Docker](https://docs.docker.com/get-started/get-docker/))
- At least 4GB of available RAM
- Mistral AI API key ([Sign up here](https://console.mistral.ai/))

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ducroq/doc-chat.git
   cd doc-chat
   ```

2. **Configure environment variables**:
   Create a `.env` file in the root directory with:
   ```
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL=mistral-tiny
   MISTRAL_DAILY_TOKEN_BUDGET=10000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   ```

3. **Start the system**:
   
   On Windows:
   ```powershell
   .\start.ps1
   ```
   
   On Linux/macOS:
   ```bash
   docker-compose up -d
   ```

4. **Access the interfaces**:
   - Web interface: http://localhost:8501
   - API documentation: http://localhost:8000/docs
   - Weaviate console: http://localhost:8080

5. **Add documents**:
   Place .txt files in the `data/` directory.

## Production Deployment (Hetzner)

### Prerequisites

- Hetzner Cloud account
- Domain name (optional, for HTTPS)
- SSH key registered with Hetzner

### Server Sizing

Recommended server configuration:
- CX31 (4 vCPU, 8GB RAM)
- Ubuntu 22.04
- 40GB SSD (minimum)

### Setup Steps

1. **Create server**:
   Create a new server in Hetzner Cloud with the recommended configuration.

2. **Install Docker**:
   ```bash
   ssh root@your-server-ip
   apt update && apt upgrade -y
   apt install -y docker.io docker-compose
   ```

3. **Clone repository**:
   ```bash
   git clone https://github.com/ducroq/doc-chat.git
   cd doc-chat
   ```

4. **Configure environment variables**:
   ```bash
   nano .env
   ```
   Add the following content:
   ```
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL=mistral-small
   MISTRAL_DAILY_TOKEN_BUDGET=50000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   ```

5. **Adjust docker-compose.yml** (optional):
   For production, replace the prototype web interface with the production one:
   ```bash
   nano docker-compose.yml
   ```
   Comment out the `web-prototype` service and uncomment the `web-production` service.

6. **Start the system**:
   ```bash
   docker-compose up -d
   ```

7. **Set up a domain** (optional):
   To use HTTPS, set up DNS to point to your server's IP address, then install Certbot for SSL:
   ```bash
   apt install -y certbot python3-certbot-nginx
   certbot --nginx -d yourdomain.com
   ```

8. **Create a data volume** (recommended):
   For data persistence:
   ```bash
   docker volume create doc_chat_data
   ```
   Update your docker-compose.yml to use this volume for the data directory.

## Backup and Maintenance

### Backup Weaviate Data

```bash
# Stop the containers
docker-compose down

# Backup the Weaviate data volume
docker run --rm -v weaviate_data:/data -v $(pwd)/backups:/backups ubuntu tar -czvf /backups/weaviate_backup_$(date +%Y%m%d).tar.gz /data

# Restart the containers
docker-compose up -d
```

### Update the System

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

## Monitoring

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
```

### Health Checks

```bash
# Check API status
curl http://localhost:8000/status

# Check document count
curl http://localhost:8000/documents/count

# Detailed statistics
curl http://localhost:8000/statistics
```

## Troubleshooting

### Common Issues

1. **Weaviate not starting**:
   - Check RAM availability
   - Verify port 8080 is not in use
   - Check logs: `docker-compose logs weaviate`

2. **Document processor not detecting files**:
   - Ensure files are .txt format
   - Check permissions on data directory
   - Verify processor logs: `docker-compose logs processor`

3. **API connection errors**:
   - Confirm Mistral API key is valid
   - Check network connectivity
   - Verify token budget hasn't been exhausted

### Restarting Services

```bash
# Restart a specific service
docker-compose restart api

# Reset Weaviate data (caution: deletes all indexed documents)
docker-compose down
docker volume rm doc_chat_weaviate_data
docker-compose up -d
```

## Security Considerations

- Keep your Mistral API key secure
- Limit access to the server using firewall rules
- Consider implementing authentication for production
- Regularly update the system and dependencies
- Monitor logs for unusual activity