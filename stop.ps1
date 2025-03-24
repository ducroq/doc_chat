param ()

Write-Host "Starting graceful shutdown of EU-compliant RAG system..." -ForegroundColor Cyan

# Make a call to the API to flush logs (optional step)
try {
    Write-Host "Requesting API to flush logs..." -ForegroundColor Yellow
    
    # If you have an API endpoint for flushing, you could use it like this:
    Invoke-WebRequest -Uri "http://localhost:8000/admin/flush-logs" -Method POST -UseBasicParsing
    
    # Small delay to allow the API time to process the request
    Start-Sleep -Seconds 2
} catch {
    Write-Host "Failed to request log flush: $_" -ForegroundColor Yellow
}

# Step 1: Stop the frontend and API first (they depend on other services)
Write-Host "Stopping frontend and API..." -ForegroundColor Yellow
docker-compose stop vue-frontend api
Write-Host "Frontend and API stopped." -ForegroundColor Green

# Step 2: Stop the processor 
Write-Host "Stopping document processor..." -ForegroundColor Yellow
docker-compose stop processor
Write-Host "Document processor stopped." -ForegroundColor Green

# Step 3: Stop Weaviate and text vectorizer 
Write-Host "Stopping Weaviate and text vectorizer..." -ForegroundColor Yellow
docker-compose stop weaviate t2v-transformers
Write-Host "Weaviate and text vectorizer stopped." -ForegroundColor Green

Write-Host "All services have been gracefully stopped." -ForegroundColor Green

# Fully remove everything at the end
Write-Host "Removing all containers..." -ForegroundColor Yellow
docker-compose down
Write-Host "All containers removed." -ForegroundColor Green