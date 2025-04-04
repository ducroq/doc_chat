param ()

Write-Host "Starting graceful shutdown of EU-compliant RAG system..." -ForegroundColor Cyan

# Make a call to the API to flush logs
try {
    Write-Host "Requesting API to flush logs..." -ForegroundColor Yellow
    
    # Get the API key from the internal_api_key file
    $apiKeyPath = "./secrets/internal_api_key.txt"
    if (Test-Path $apiKeyPath) {
        $apiKey = Get-Content $apiKeyPath -Raw
        
        # Make the request with the API key header
        $headers = @{
            "X-API-Key" = $apiKey.Trim()
            "Content-Type" = "application/json"
        }
        
        $response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/admin/flush-logs" -Method POST -Headers $headers
        Write-Host "Log flush result: $($response.status) - $($response.message)" -ForegroundColor Green
    } else {
        Write-Host "API key file not found at $apiKeyPath - skipping log flush" -ForegroundColor Yellow
    }
    
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