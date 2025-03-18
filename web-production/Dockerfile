FROM nginx:stable-alpine AS builder

# Create a stage for generating any static content
WORKDIR /build
COPY html/ ./html/

# Main stage
FROM nginx:stable-alpine

# Install required packages
RUN apk --no-cache add curl

# Set up non-root user (nginx user already exists in the nginx image)
RUN touch /var/run/nginx.pid \
    && chown -R nginx:nginx /var/run/nginx.pid

# Copy static files from builder stage 
COPY --from=builder /build/html /usr/share/nginx/html

# Copy nginx configuration files
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf
COPY nginx/mime.types /etc/nginx/mime.types

# Create health check script
COPY health-check.sh /usr/local/bin/health-check.sh
RUN chmod +x /usr/local/bin/health-check.sh

# Create /app directory to store dynamic content
RUN mkdir -p /app && \
    chown -R nginx:nginx /app /var/cache/nginx /var/log/nginx /etc/nginx/conf.d

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD /usr/local/bin/health-check.sh || exit 1

EXPOSE 80

# Switch to non-root user
USER nginx

# Start nginx
CMD ["nginx", "-g", "daemon off;"]