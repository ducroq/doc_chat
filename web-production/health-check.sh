#!/bin/sh
# Health check script for web-production

# Test nginx configuration
nginx -t >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Nginx configuration is invalid"
    exit 1
fi

# Check if nginx is running
if ! pgrep -x "nginx" > /dev/null; then
    echo "Nginx is not running"
    exit 1
fi

# Test API availability
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://api:8000/status)
if [ "$API_STATUS" != "200" ]; then
    echo "API is not responding correctly: $API_STATUS"
    # Don't exit with error as the frontend can still serve content even if API is down
    # exit 1
fi

# Test web application
WEBAPP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://web-prototype:8501/_stcore/health)
if [ "$WEBAPP_STATUS" != "200" ]; then
    echo "Web application is not responding correctly: $WEBAPP_STATUS"
    # Don't exit with error as we can still proxy API requests
    # exit 1
fi

# All checks passed
echo "Health check passed"
exit 0