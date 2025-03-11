#!/bin/sh

# Generate config file from environment variables
cat > /usr/share/nginx/html/config.js << EOF
window.config = {
  apiUrl: '/api',
  chatLoggingEnabled: ${ENABLE_CHAT_LOGGING:-false}
};
EOF

# Start nginx
exec nginx -g 'daemon off;'