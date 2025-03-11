FROM nginx:stable-alpine

COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# No need for an entrypoint script
CMD ["nginx", "-g", "daemon off;"]