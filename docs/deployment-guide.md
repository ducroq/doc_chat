# Deployment Guide

This guide provides instructions for deploying the EU-Compliant Document Chat system in various environments.

## Web Architecture

The system uses a modern web architecture:

1. **Frontend**: Vue.js Single Page Application (SPA)
   - Built with Vue 3 and Pinia for state management
   - Compiled to static assets during Docker build

2. **Web Server**: Nginx
   - Serves the compiled Vue.js application
   - Acts as a reverse proxy for API requests
   - Adds security headers
   - Configured automatically via the entrypoint.sh script
   
3. **API Service**: FastAPI backend
   - All actual data processing happens here
   - Communicates with Weaviate and Mistral AI

This architecture provides excellent performance, security, and scalability while maintaining a clean separation of concerns.

## Local Deployment

### Prerequisites

- **Windows users**: Update Windows Subsystem for Linux
  ```bash
  wsl --update
  ```
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
   # Core Configuration
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL=mistral-tiny
   MISTRAL_DAILY_TOKEN_BUDGET=10000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   
   # Chat Logging Configuration (for research purposes)
   ENABLE_CHAT_LOGGING=false        # Set to 'true' to enable logging
   ANONYMIZE_CHAT_LOGS=true         # Controls anonymization of identifiers
   LOG_RETENTION_DAYS=30            # Days to keep logs before deletion
   ```

3. **Create chat_data directory** (if you plan to use logging):
   ```bash
   mkdir -p chat_data
   ```

4. **Start the system**:
   
   On Windows:
   ```powershell
   .\start.ps1
   ```
   
   On Linux/macOS:
   ```bash
   docker-compose up -d
   ```

5. **Access the interfaces**:
   - Web interface: http://localhost
   - API documentation: http://localhost:8000/docs
   - Weaviate console: http://localhost:8080
   - Privacy notice: http://localhost:8000/privacy

6. **Add documents**:
   Place .txt files in the `data/` directory.


## Linux Server Deployment

### Prerequisites

- Linux server (Ubuntu 22.04 LTS recommended)
- Docker and Docker Compose installed
- At least 4GB of RAM
- Mistral AI API key

### Docker Setup

When deploying on Linux, you may need to configure Docker permissions:

```bash
# Add your user to the docker group (recommended)
sudo usermod -aG docker $USER
# Log out and log back in, or run:
newgrp docker
```

### Deployment Steps

1. Clone the repository:

```bash
# If using SSH (recommended)
git clone git@github.com:ducroq/doc-chat.git
# Or using HTTPS with a personal access token
git clone https://TOKEN@github.com/ducroq/doc-chat.git
```

2. Use the start and stop scripts:

```bash
# Make scripts executable
chmod +x start.sh stop.sh

# Start the system
./start.sh

# Stop the system when needed
./stop.sh
```

3. Troubleshooting common issues:
- If Docker permission errors occur, ensure your user is in the docker group
- If build errors mention COPY commands, ensure destination paths end with '/'
- For networking issues, check if required ports are open and not blocked by firewalls

## Production Deployment (E.g. Hetzner)

### Prerequisites

- Domain name (optional, for HTTPS)
- Hetzner Cloud account
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
   # Core Configuration
   WEAVIATE_URL=http://weaviate:8080
   MISTRAL_API_KEY=your_api_key_here
   MISTRAL_MODEL=mistral-small
   MISTRAL_DAILY_TOKEN_BUDGET=50000
   MISTRAL_MAX_REQUESTS_PER_MINUTE=10
   
   # Chat Logging Configuration (for research purposes)
   ENABLE_CHAT_LOGGING=false        # Set to 'true' when needed for research
   ANONYMIZE_CHAT_LOGS=true         # Keep enabled for GDPR compliance
   LOG_RETENTION_DAYS=30            # Adjust based on your data policy
   ```

5. **Create directories for persistence**:
   ```bash
   mkdir -p data chat_data
   chmod 777 data chat_data  # Ensure Docker can write to these directories
   ```

6. **Adjust docker-compose.yml** (optional):
   For production, replace the prototype web interface with the production one:
   ```bash
   nano docker-compose.yml
   ```
   Comment out the `web-prototype` service and uncomment the `web-production` service.
   
   Make sure the volume mounts include chat_data:
   ```yaml
   api:
     # ... other configuration ...
     volumes:
       - ./chat_data:/app/chat_data
   ```

7. **Start the system**:
   ```bash
   docker-compose up -d
   ```

8. **Set up a domain** (optional):
   To use HTTPS, set up DNS to point to your server's IP address, then install Certbot for SSL:
   ```bash
   apt install -y certbot python3-certbot-nginx
   certbot --nginx -d yourdomain.com
   ```

9. **Create a data volume** (recommended):
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

# Backup chat logs if enabled
tar -czvf backups/chat_logs_backup_$(date +%Y%m%d).tar.gz chat_data/

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

### Managing Chat Logs

```bash
# Enable chat logging (edit .env file)
sed -i 's/ENABLE_CHAT_LOGGING=false/ENABLE_CHAT_LOGGING=true/' .env
docker-compose restart api vue-frontend

# Disable chat logging
sed -i 's/ENABLE_CHAT_LOGGING=true/ENABLE_CHAT_LOGGING=false/' .env
docker-compose restart api vue-frontend

# Delete all chat logs (if needed for GDPR compliance)
rm -rf chat_data/chat_log_*.jsonl
```

## Monitoring

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Check chat logs directory
ls -la chat_data/
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

## Research and Analytics

### Working with Chat Logs

When `ENABLE_CHAT_LOGGING` is set to `true`, the system will log chat interactions to JSONL files in the `chat_data` directory:

```
chat_data/
  chat_log_20250309.jsonl
  chat_log_20250310.jsonl
```

These logs can be processed for research purposes:

```bash
# Simple analysis example using jq (install with: apt install jq)
cat chat_data/chat_log_20250309.jsonl | jq '.query' | sort | uniq -c | sort -nr

# Count interactions by day
find chat_data/ -name "chat_log_*.jsonl" | xargs wc -l

# Extract all questions to a CSV file
echo "date,question" > questions.csv
for f in chat_data/chat_log_*.jsonl; do
  date=$(basename $f | sed 's/chat_log_\(.*\)\.jsonl/\1/')
  cat $f | jq -r --arg date "$date" '.query | $date + "," + .'  >> questions.csv
done
```

Remember that any research using these logs must comply with GDPR and your organization's data policies.

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

4. **Chat logging issues**:
   - Check if the chat_data directory exists and is writable
   - Verify environment variables are set correctly
   - Look for errors in API logs: `docker-compose logs api | grep "chat_logger"`
   - Make sure comments in .env file are on separate lines from values

### Restarting Services

```bash
# Restart a specific service
docker-compose restart api

# Reset Weaviate data (caution: deletes all indexed documents)
docker-compose down
docker volume rm doc_chat_weaviate_data
docker-compose up -d
```

## Security and Privacy Considerations

- Keep your Mistral API key secure using Docker Secrets
- Rotate API keys regularly (system warns when keys are older than 90 days)
- Limit access to the server using firewall rules
- Web authentication is enabled by default - protect your credentials
- Store all sensitive information in the `./secrets/` directory, not environment variables
- Use the configured security headers to protect against common web vulnerabilities
- Regularly update the system and dependencies
- Monitor logs for unusual activity or failed login attempts
- Only enable chat logging when necessary for research
- Ensure users are informed when logging is active (UI warning)
- Process log data in compliance with GDPR
- Delete logs after they are no longer needed
- Maintain the privacy notice at `/privacy` endpoint
- Verify Docker container images before deployment when possible

### Setting Up API Keys Securely

For better security, set up API keys using Docker Secrets:

```bash
# Create the secrets directory
mkdir -p ./secrets

# Set up the Mistral API key
echo "your_mistral_api_key_here" > ./secrets/mistral_api_key.txt
chmod 600 ./secrets/mistral_api_key.txt

# Set up the internal API key
openssl rand -hex 32 > ./secrets/internal_api_key.txt
chmod 600 ./secrets/internal_api_key.txt
```

### API Key Rotation

To rotate your API keys:

```bash
# Generate a new internal API key
openssl rand -hex 32 > ./secrets/internal_api_key.txt
chmod 600 ./secrets/internal_api_key.txt

# Restart services to apply the new key
docker-compose restart api vue-frontend
```

For Mistral API key, update the key in their portal, then:

```bash
# Update the Mistral API key
echo "your_new_mistral_api_key_here" > ./secrets/mistral_api_key.txt
chmod 600 ./secrets/mistral_api_key.txt

# Restart services to apply the new key
docker-compose restart api
```

### Authentication Setup

#### Initial User Setup

Before deploying the system to production, set up initial user accounts:

```bash
# Generate a JWT secret key
openssl rand -hex 32 > ./secrets/jwt_secret_key.txt
chmod 600 ./secrets/jwt_secret_key.txt

# Create an admin user
python manage_users.py create admin --generate-password --admin --full-name "System Administrator" --email "admin@yourdomain.com"
```

The generated password will be displayed in the console. Make note of it as it won't be shown again.

#### User Management in Production

To manage users in a production environment:

```bash
# Connect to the production server
ssh user@production-server

# Navigate to the application directory
cd /path/to/doc-chat

# List existing users
python manage_users.py list

# Add a new user
python manage_users.py create username --generate-password --full-name "User Name" --email "user@example.com"

# Disable a user (e.g., when they leave the organization)
python manage_users.py disable username

# Reset a user's password
python manage_users.py reset-password username --generate
```

After making changes to user accounts, restart the API service to ensure the changes take effect:

```bash
docker-compose restart api
```

#### Securing JWT Secret

The JWT secret key is used to sign authentication tokens and should be treated as sensitive:

1. Store the secret key in the `./secrets/jwt_secret_key.txt` file
2. Ensure this file has restricted permissions (chmod 600)
3. Backup this file securely - if lost, all users will need to log in again
4. Rotate this secret periodically (e.g., every 90 days) for enhanced security

When rotating the JWT secret:

```bash
# Generate a new secret
openssl rand -hex 32 > ./secrets/jwt_secret_key.txt.new
chmod 600 ./secrets/jwt_secret_key.txt.new

# Replace the old secret
mv ./secrets/jwt_secret_key.txt.new ./secrets/jwt_secret_key.txt

# Restart the API to use the new secret
docker-compose restart api
```

Note that changing the JWT secret will invalidate all existing sessions, requiring users to log in again.
