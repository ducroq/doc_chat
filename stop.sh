#!/bin/bash

echo "Stopping EU-compliant RAG system..."

# Gracefully stop all services
docker-compose down

echo "All services stopped successfully."