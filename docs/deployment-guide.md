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
   - Web interface: http://localhost:8081
   - API documentation: http://localhost:8000/docs
   - Weaviate console: http://localhost:8080
   - Privacy notice: http://localhost:8000/privacy

6. **Add documents**:
   Place .md files in the `data/` directory.


## Linux Server Deployment

### Prerequisites

- Linux server, e.g. Hetzner Cloud account
- SSH key registered with Hetzner
- At least 4GB of RAM
- Mistral AI API key
- Domain name (optional, for HTTPS)

### Server setup

Create a new server in Hetzner Cloud with the recommended configuration.

**1. Server selection**

First, you'll need to choose an appropriate Hetzner Cloud server:

Recommended: CX31 (4 vCPU, 8GB RAM) or higher
Operating System: Ubuntu 22.04 LTS
Storage: At least 40GB SSD (your documents will be stored here)

**2. Initial server setup**

After creating your server, connect via SSH:
```bash
ssh root@your-server-ip
```

Update system
```bash
apt update && apt upgrade -y
```

Install Docker and Docker Compose
```bash
apt install -y docker.io docker-compose
```

Create a non-root user (optional but recommended)
```bash
useradd -m -s /bin/bash docadmin
usermod -aG docker docadmin
passwd docadmin
# Set a strong password when prompted
```

<!-- **2. Docker Setup**

When deploying on Linux, you may need to configure Docker permissions:

```bash
# Add your user to the docker group (recommended)
sudo usermod -aG docker $USER
# Log out and log back in, or run:
newgrp docker
``` -->

**3. Clone the repository**

```bash
# If using SSH (recommended)
git clone git@github.com:ducroq/doc-chat.git

# Or using HTTPS with a personal access token
git clone https://TOKEN@github.com/ducroq/doc-chat.git
```

**4. Create directories for persistence**

```bash
mkdir -p data chat_data secrets
```

Set ownership to match the user in the container (1000:1000)
```bash
chown -R 1000:1000 chat_data data secrets
```

Set appropriate permissions
```bash
chmod -R 755 chat_data data secrets
```

<!-- or overly permissive  -->
<!-- chmod 777 data chat_data  # Ensure Docker can write to these directories -->

Same for users database, set the appropriate permissions on the host
```bash
sudo chown 1000:1000 users.json
sudo chmod 755 users.json 
```

**5. Adding documents**

You can add documents to your system by uploading them to the data directory:

For example, using SCP from your local machine:
```bash
scp your-document.md root@your-server-ip:/root/doc_chat/data/
```

If you need to add metadata:
```bash
scp your-document.metadata.json root@your-server-ip:/root/doc_chat/data/
```

The document processor should automatically detect and process these files on application startup.


**6. Configure API keys and secrets**

Create Mistral API key file
```bash
echo "your_mistral_api_key_here" > ./secrets/mistral_api_key.txt
```

Create internal API key
```bash
openssl rand -hex 32 > ./secrets/internal_api_key.txt
```

Create JWT secret for authentication
```bash
openssl rand -hex 32 > ./secrets/jwt_secret_key.txt
```

**7. Create environment configuration**

Create a .env file with appropriate settings:

```bash
cat > .env << EOL
# Mistral AI Configuration
MISTRAL_MODEL=mistral-large-latest
# mistral-small-latest, mistral-large-latest, mistral-medium, mistral-tiny (for cost control)
MISTRAL_DAILY_TOKEN_BUDGET=100000
MISTRAL_MAX_TOKENS_PER_REQUEST=5000
MISTRAL_MAX_REQUESTS_PER_MINUTE=30

# Weaviate Configuration
WEAVIATE_URL=http://weaviate:8080

# Document Processor Configuration
DOCS_DIR=/data

# Chat Logging Controls - All default to privacy-preserving settings
# Set to 'true' to enable logging
ENABLE_CHAT_LOGGING=true
# Set to 'false' to disable anonymization (not recommended)
ANONYMIZE_CHAT_LOGS=true
# Number of days to keep logs before automatic deletion
LOG_RETENTION_DAYS=30
EOL
```

**8. Setup user authentication**

Create at least one admin user:

```bash
python3 manage_users.py create admin --generate-password --admin --full-name "System Administrator" --email "your@email.com"
```

**9. Build and launch**

If needed, delete old stuff:
```bash
docker-compose down --volumes --remove-orphans
```

```bash
docker-compose build --no-cache
```

Enable Docker to start on boot
```bash
sudo systemctl enable docker
```

Use the start and stop scripts

```bash
# Make scripts executable
chmod +x start.sh stop.sh

# Start the system
./start.sh

# Stop the system when needed
./stop.sh
```

**10. Domain setup**

If you have a domain name, set up DNS to point to your Hetzner server's IP.

Create or Modify DNS Records by setting an A Record (for IPv4):

- An "A" record maps a domain name (or subdomain) to an IPv4 address.   
- Create a new "A" record (or modify an existing one).
- Host/Name:
   - To point your main domain (e.g., yourdomain.com) to the IP, enter @ or leave it blank.
   - To point a subdomain (e.g., www.yourdomain.com), enter www.
   - Value/Points to/Destination: Enter your Hetzner server's IPv4 address.

**11. Test the Deployment**

Visit http://your-server-ip (or https://yourdomain.com if you set up SSL)
Log in with the admin credentials you created
Test querying your documents


**12. HTTPS setup**

Stop the application using the script
```bash
./stop.sh
```

Change the port mapping for the vue-frontend service to port 8081 instead:

Find this section in docker-compose.yml:
```bash
vue-frontend:
  build: 
    context: ./vue-frontend
    dockerfile: Dockerfile
  ports:
    - "80:80"
```

Configure Nginx to proxy requests to Docker, but first get the internal API key:
```bash
cat ./secrets/internal_api_key.txt
```

Set the right apikey in the API location section of the configuration below 
```
proxy_set_header X-API-Key "";
```

Now create the Nginx configuration
```bash
cat > /etc/nginx/sites-available/docchat << 'EOL'
server {
    listen 80;
    server_name doc-chat-demo.jeroenveen.nl;
    # Simple redirect to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name doc-chat-demo.jeroenveen.nl;
    ssl_certificate /etc/letsencrypt/live/doc-chat-demo.jeroenveen.nl/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/doc-chat-demo.jeroenveen.nl/privkey.pem;
    
    # Frontend location
    location / {
        proxy_pass http://localhost:8081;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_pass_request_headers on;
    }
    
    # API location - THIS WAS MISSING FROM YOUR HTTPS SERVER BLOCK
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 90;
        # Add the API key header
        proxy_set_header X-API-Key "";
    }
}
EOL
```

Double-check that /etc/letsencrypt/live/your-domain.nl/fullchain.pem and /etc/letsencrypt/live/yor-domain.nl/privkey.pem are the correct paths to your Let's Encrypt certificate files.

Enable the site
```bash
ln -s /etc/nginx/sites-available/dochat /etc/nginx/sites-enabled/
```

Sometimes, previous failed Nginx attempts can leave lingering processes. If you see multiple Nginx processes, try killing them all:
```bash
sudo killall nginx
```

Start Nginx
```bash
sudo systemctl start nginx
```

Or reload Nginx
```bash
sudo systemctl reload nginx
```

Test Nginx configuration
```bash
nginx -t
```

Install Certbot
```bash
apt install -y certbot python3-certbot-nginx
```

Configure Nginx (assuming the frontend is already running on port 80)
```bash
certbot --nginx -d yourdomain.com
```
This will automatically obtain and configure SSL certificates

Start your application using the script
```bash
./start.sh
```

**13. Configure Firewall**

Install UFW if not already installed
```bash
apt install -y ufw
```

Allow SSH (important to do this first so you don't lock yourself out)
```bash
ufw allow ssh
```

Allow HTTP/HTTPS
```bash
ufw allow 80/tcp
ufw allow 443/tcp
```

Optionally allow direct access to the API (if needed)
```bash
ufw allow 8000/tcp
```

Enable the firewall
```bash
ufw enable
```

**14. Test the Deployment**
- Visit https://yourdomain.com 
- Log in with the admin credentials you created
- Test querying your documents


## Monitoring and maintenance

### Update the system

```bash
# Pull latest changes
git pull

# Rebuild and restart
./stop.sh
docker-compose build --no-cache
./start.sh
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Check chat logs directory
ls -la chat_data/
```

### Health checks

```bash
# Check API status
curl http://localhost:8000/status

# Check document count
curl http://localhost:8000/documents/count

# Detailed statistics
curl http://localhost:8000/statistics
```

### Setup monitoring scripts

**1. Basic monitoring**
```bash

cat > /root/monitor.sh << 'EOL'
#!/bin/bash
cd /root/doc-chat
docker-compose ps
echo "API Status:"
curl -s http://localhost:8000/status
echo -e "\n\nDocument Count:"
curl -s http://localhost:8000/documents/count
EOL

chmod +x /root/monitor.sh
```

**2. Enhanced monitoring**

```bash
cat > /root/enhanced-monitor.sh << 'EOL'
#!/bin/bash
LOG_FILE="/var/log/doc-chat-health.log"
EMAIL="your-email@example.com"

echo "========== $(date) ==========" >> $LOG_FILE

# Check each container status
echo "Container Status:" >> $LOG_FILE
docker-compose -f /root/doc-chat/docker-compose.yml ps >> $LOG_FILE

# Check API health
API_HEALTH=$(curl -s http://localhost:8000/status)
echo "API Health: $API_HEALTH" >> $LOG_FILE

# Check document count
DOC_COUNT=$(curl -s http://localhost:8000/documents/count)
echo "Document Count: $DOC_COUNT" >> $LOG_FILE

# Check disk space
echo "Disk Space:" >> $LOG_FILE
df -h / >> $LOG_FILE

# Check memory usage
echo "Memory Usage:" >> $LOG_FILE
free -h >> $LOG_FILE

# Check for containers that need to be restarted
UNHEALTHY=$(docker ps --filter "status=exited" --filter "name=doc_chat" --format "{{.Names}}")
if [ ! -z "$UNHEALTHY" ]; then
  echo "Unhealthy containers found: $UNHEALTHY" >> $LOG_FILE
  echo "Attempting to restart..." >> $LOG_FILE
  cd /root/doc-chat && docker-compose up -d
  
  # Optionally send an email alert
  # echo "Containers restarted on $(hostname): $UNHEALTHY" | mail -s "DocChat Alert" $EMAIL
fi

echo "========== End of Check ==========" >> $LOG_FILE
EOL

chmod +x /root/enhanced-monitor.sh
```

**3. Automated updates**

Set up periodic Docker image updates:

Create update script
```bash
cat > /root/update.sh << 'EOL'
#!/bin/bash
cd /root/doc-chat
git pull
docker-compose build
docker-compose up -d
EOL

chmod +x /root/update.sh
```

Set up daily checks via cron:

Edit the crontab
```bash
crontab -e
```
paste:
```
## Run monitoring every 6 hours
#0 */6 * * * /root/monitor.sh > /root/system_status.log 2>&1

# Run enhanced monitoring every hour
0 * * * * /root/enhanced-monitor.sh

# Run update check weekly on Sunday at 3 AM
0 3 * * 0 /root/update.sh > /root/update.log 2>&1
```

**4. Resource monitoring**

Install basic monitoring tools:

```bash
apt install -y htop iotop
```

**5. Log Rotation**

Ensure Docker logs don't fill up your disk:

```bash
# Configure Docker log rotation
cat > /etc/docker/daemon.json << 'EOL'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOL

# Restart Docker to apply changes
systemctl restart docker
```


**6. TODO: Setup ntfy**

ntfy (pronounced notify) is a simple HTTP-based pub-sub notification service. It allows you to send notifications to your phone or desktop via scripts from any computer, and/or using a REST API.

 ntfy.sh:

We need a mail client to send monitoring info

```bash
apt-get install -y ssmtp mailutils
```

Quick configuration

<!-- 
# Then use the mail command as is
echo "DocChat system on $(hostname) was updated on $(date)" | mail -s "DocChat Update" recipient@example.com -->







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
   - Ensure files are .md format
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
