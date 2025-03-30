#!/bin/bash

echo "Starting EU-compliant RAG system..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
else
    echo "Docker is running."
fi

# Start Weaviate and text vectorizer
echo "Starting Weaviate and text vectorizer..."
docker-compose up -d weaviate t2v-transformers

# Wait for the text vectorizer to be ready
echo "Waiting for text vectorizer to be ready..."
t2v_ready=false
attempts=0
max_attempts=20

while [ "$t2v_ready" = false ] && [ $attempts -lt $max_attempts ]; do
    ((attempts++))
    echo "Checking text vectorizer... ($attempts/$max_attempts)"
    
    if docker run --rm --network doc_chat_backend curlimages/curl -s --fail http://t2v-transformers:8080/.well-known/ready > /dev/null 2>&1; then
        t2v_ready=true
        echo "Text vectorizer is ready!"
    else
        echo "  Text vectorizer not ready yet, waiting..."
        sleep 3
    fi
done

if [ "$t2v_ready" = false ]; then
    echo "Warning: Text vectorizer did not become ready within the timeout period."
    echo "Continuing anyway, but there might be initialization issues..."
fi

# Check if Weaviate is ready
echo "Waiting for Weaviate to be ready..."
weaviate_ready=false
attempts=0
max_attempts=30

while [ "$weaviate_ready" = false ] && [ $attempts -lt $max_attempts ]; do
    ((attempts++))
    echo "Checking Weaviate... ($attempts/$max_attempts)"
    
    if docker run --rm --network doc_chat_backend curlimages/curl -s --fail http://weaviate:8080/v1/.well-known/ready > /dev/null 2>&1; then
        weaviate_ready=true
        echo "Weaviate is ready!"
    else
        echo "  Weaviate not ready yet, waiting..."
        sleep 3
    fi
done

if [ "$weaviate_ready" = false ]; then
    echo "Error: Weaviate did not become ready within the timeout period."
    exit 1
fi

# Allow additional time for Weaviate to fully initialize
echo "Giving Weaviate extra time to fully initialize..."
sleep 5

# Start the processor
echo "Starting document processor..."
docker-compose up -d processor

# Wait for processor to initialize
echo "Waiting for processor to initialize..."
sleep 10

# Start the API and frontend
echo "Starting API and Vue.js frontend..."
docker-compose up -d api vue-frontend
frontend_url="http://localhost:8081"

# Wait for API to initialize and verify connection to Weaviate
echo "Waiting for API to connect to Weaviate..."
api_ready=false
attempts=0
max_attempts=20

while [ "$api_ready" = false ] && [ $attempts -lt $max_attempts ]; do
    ((attempts++))
    echo "Checking API status... ($attempts/$max_attempts)"
    
    if response=$(curl -s http://localhost:8000/status 2>/dev/null); then
        if echo "$response" | grep -q '"weaviate":"connected"'; then
            api_ready=true
            echo "API successfully connected to Weaviate!"
        else
            echo "  API not fully connected yet, waiting..."
            sleep 3
        fi
    else
        echo "  API not ready yet, waiting..."
        sleep 3
    fi
done

if [ "$api_ready" = false ]; then
    echo "Warning: API did not connect to Weaviate properly within the timeout period."
    echo "You may need to restart the API container: docker-compose restart api"
else
    echo "All services started and connected successfully!"
fi

# Display access information
echo "Vue.js frontend: $frontend_url"
echo "API documentation: http://localhost:8000/docs"
echo "System statistics: http://localhost:8000/statistics"
echo "Weaviate console: http://localhost:8080"
echo "Processor logs: docker-compose logs -f processor"
echo "Vue.js logs: docker-compose logs -f vue-frontend"
echo ""
echo "Use ./stop.sh to stop all services."