FROM nginx:stable-alpine

# Copy configuration
COPY nginx/nginx.conf /etc/nginx/nginx.conf
# COPY nginx/streamlit-proxy.conf /etc/nginx/streamlit-proxy.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf.original
COPY nginx/mime.types /etc/nginx/mime.types
COPY nginx/.htpasswd /etc/nginx/.htpasswd

# Copy static files
COPY html /usr/share/nginx/html
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
RUN chmod -R 755 /usr/share/nginx/html

EXPOSE 80

CMD ["/entrypoint.sh"]