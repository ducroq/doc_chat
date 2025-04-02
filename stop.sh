#!/bin/bash

echo "Starting graceful shutdown of EU-compliant RAG system..."

# Make a call to the API to flush logs
echo "Requesting API to flush logs..."
api_key_path="./secrets/internal_api_key.txt"
if [ -f "$api_key_path" ]; then
    api_key=$(cat "$api_key_path" | tr -d '\n\r')
    
    # Make the request with the API key header
    response=$(curl -s -X POST -H "X-API-Key: $api_key" -H "Content-Type: application/json" http://localhost:8000/api/v1/admin/flush-logs)
    
    if [ $? -eq 0 ]; then
        echo "Log flush completed: $response"
    else
        echo "Failed to flush logs: $response"
    fi
    
    # Small delay to allow the API time to process the request
    sleep 2
else
    echo "API key file not found at $api_key_path - skipping log flush"
fi

# Step 1: Stop the frontend and API first (they depend on other services)
echo "Stopping frontend and API..."
docker-compose stop vue-frontend api
echo "Frontend and API stopped."

# Step 2: Stop the processor
echo "Stopping document processor..."
docker-compose stop processor
echo "Document processor stopped."

# Step 3: Stop Weaviate and text vectorizer
echo "Stopping Weaviate and text vectorizer..."
docker-compose stop weaviate t2v-transformers
echo "Weaviate and text vectorizer stopped."

echo "All services have been gracefully stopped."

# Fully remove everything at the end
echo "Removing all containers..."
docker-compose down
echo "All containers removed."