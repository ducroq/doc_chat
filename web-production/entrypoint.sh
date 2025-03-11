#!/bin/sh

# If using Streamlit proxy, the config has already been mounted
# Otherwise, use the default nginx config
if [ "${USE_STREAMLIT_PROXY}" != "true" ]; then
    echo "Using default HTML/JS frontend"
    cp /etc/nginx/conf.d/default.conf.original /etc/nginx/conf.d/default.conf
fi

# Generate config file from environment variables
cat > /usr/share/nginx/html/config.js << EOF
window.config = {
  apiUrl: '/api',
  chatLoggingEnabled: ${ENABLE_CHAT_LOGGING:-false}
};
EOF

# Start nginx
exec nginx -g 'daemon off;'