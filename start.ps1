Write-Host "Starting EU-compliant RAG system..." -ForegroundColor Cyan

# Start Weaviate and text vectorizer
Write-Host "Starting Weaviate and text vectorizer..." -ForegroundColor Yellow
docker-compose up -d weaviate t2v-transformers

# Wait for Weaviate to be ready
Write-Host "Waiting for Weaviate to be ready..." -ForegroundColor Yellow
$ready = $false
$attempts = 0
$maxAttempts = 30

while (-not $ready -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Waiting for Weaviate... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/v1/.well-known/ready" -UseBasicParsing
        # If we get a 200 response, Weaviate is ready (even with empty content)
        if ($response.StatusCode -eq 200) {
            $ready = $true
            Write-Host "Weaviate is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Not ready yet: $_" -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
}

if (-not $ready) {
    Write-Host "Weaviate did not become ready within the timeout period." -ForegroundColor Red
    exit 1
}

# Start the processor
Write-Host "Starting document processor..." -ForegroundColor Yellow
docker-compose up -d processor

# Wait a bit for processor to initialize
Write-Host "Waiting for processor to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Start the API and web interface
Write-Host "Starting API and web interface..." -ForegroundColor Yellow
docker-compose up -d api web-prototype

Write-Host "All services started!" -ForegroundColor Green
Write-Host "Web interface: http://localhost:8501" -ForegroundColor Cyan
Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Weaviate console: http://localhost:8080" -ForegroundColor Cyan