#!/bin/bash

echo "Starting EU-compliant RAG system..."

# Start Weaviate and text vectorizer
echo "Starting Weaviate and text vectorizer..."
docker-compose up -d weaviate t2v-transformers

# Wait for Weaviate to be ready
echo "Waiting for Weaviate to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8080/.well-known/ready | grep -q "true"; then
    echo "Weaviate is ready!"
    break
  fi
  echo "Waiting for Weaviate... ($i/30)"
  sleep 5
done

# Start the processor
echo "Starting document processor..."
docker-compose up -d processor

# Wait a bit for processor to initialize
sleep 5

# Start the API and web interface
echo "Starting API and web interface..."
docker-compose up -d api web-prototype

echo "All services started!"
echo "Web interface: http://localhost:8501"
echo "API documentation: http://localhost:8000/docs"
echo "Weaviate console: http://localhost:8080"