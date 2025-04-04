#!/bin/sh
set -e

API_URL='/api'

# Read API key from file
if [ -f "$INTERNAL_API_KEY_FILE" ]; then
  API_KEY=$(cat $INTERNAL_API_KEY_FILE)
  echo "API key found"
else
  echo "Warning: No API key file found at $INTERNAL_API_KEY_FILE"
  API_KEY=""
fi

# Parse ENABLE_CHAT_LOGGING to ensure it's a proper boolean value for JavaScript
LOGGING_ENABLED="false"
if [ "${ENABLE_CHAT_LOGGING}" = "true" ]; then
  LOGGING_ENABLED="true"
fi

# Make sure we escape any special characters in the API key for JavaScript
ESCAPED_API_KEY=$(echo "$API_KEY" | sed 's/[\&/]/\\&/g')

# Create a custom Nginx configuration file with the API key
cat > /etc/nginx/conf.d/default.conf << EOF
server {
    listen 80;
    server_name localhost;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    root /usr/share/nginx/html;
    index index.html;
    
    location / {
        try_files \$uri \$uri/ /index.html;
    }
    
    # Proxy API requests and add API key
    location /api/ {
        proxy_pass http://api:8000/api/v1/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        # Set the API key directly (from secrets)
        proxy_set_header X-API-Key "${API_KEY}";
    }
}
EOF

# Properly escape special characters for the JS code
CONFIG_SCRIPT="<script>window.APP_CONFIG = { apiUrl: '${API_URL}', apiKey: '${ESCAPED_API_KEY}', enableChatLogging: ${LOGGING_ENABLED} };</script>"
sed -i "s|</head>|${CONFIG_SCRIPT}</head>|" /usr/share/nginx/html/index.html

echo "Generated custom Nginx configuration with API key"

# Start nginx
exec nginx -g "daemon off;"