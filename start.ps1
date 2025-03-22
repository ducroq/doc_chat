param ()

Write-Host "Starting EU-compliant RAG system..." -ForegroundColor Cyan

# Check if Docker is running first
try {
    $dockerStatus = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
        exit 1
    } else {
        Write-Host "Docker is running." -ForegroundColor Green
    }
} catch {
    Write-Host "Error checking Docker status: $_" -ForegroundColor Red
    Write-Host "Please ensure Docker is installed and running, then try again." -ForegroundColor Red
    exit 1
}

# Start Weaviate and text vectorizer
Write-Host "Starting Weaviate and text vectorizer..." -ForegroundColor Yellow
docker-compose up -d weaviate t2v-transformers

# Wait for the text vectorizer to be ready first
Write-Host "Waiting for text vectorizer to be ready..." -ForegroundColor Yellow
$t2vReady = $false
$attempts = 0
$maxAttempts = 20

while (-not $t2vReady -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Checking text vectorizer... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        # Start a temporary container that has curl to check readiness
        $output = docker run --rm --network doc_chat_backend curlimages/curl -s http://t2v-transformers:8080/.well-known/ready
        if ($output -ne $null -or $LASTEXITCODE -eq 0) {
            $t2vReady = $true
            Write-Host "Text vectorizer is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Text vectorizer not ready yet: $_" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if (-not $t2vReady) {
    Write-Host "Text vectorizer did not become ready within the timeout period." -ForegroundColor Yellow
    Write-Host "Continuing anyway, but there might be initialization issues..." -ForegroundColor Yellow
}

# Now check if Weaviate is ready
Write-Host "Waiting for Weaviate to be ready..." -ForegroundColor Yellow
$weaviateReady = $false
$attempts = 0
$maxAttempts = 30

while (-not $weaviateReady -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Checking Weaviate... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        # Use a temporary container with curl to check readiness
        $output = docker run --rm --network doc_chat_backend curlimages/curl -s http://weaviate:8080/v1/.well-known/ready
        if ($output -ne $null -or $LASTEXITCODE -eq 0) {
            $weaviateReady = $true
            Write-Host "Weaviate is ready!" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Weaviate not ready yet: $_" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if (-not $weaviateReady) {
    Write-Host "Weaviate did not become ready within the timeout period." -ForegroundColor Red
    exit 1
}

# Allow additional time for Weaviate to fully initialize after reporting ready
Write-Host "Giving Weaviate extra time to fully initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Start the processor
Write-Host "Starting document processor..." -ForegroundColor Yellow
docker-compose up -d processor

# Wait for processor to initialize
Write-Host "Waiting for processor to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Start the API and frontend
Write-Host "Starting API and Vue.js frontend..." -ForegroundColor Yellow
docker-compose up -d api vue-frontend
$frontendUrl = "http://localhost:80"

# Wait for API to initialize and verify connection to Weaviate
Write-Host "Waiting for API to connect to Weaviate..." -ForegroundColor Yellow
$apiReady = $false
$attempts = 0
$maxAttempts = 20

while (-not $apiReady -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Checking API status... ($attempts/$maxAttempts)" -ForegroundColor Gray
    
    try {
        # Direct HTTP request to status endpoint
        $response = Invoke-WebRequest -Uri "http://localhost:8000/status" -UseBasicParsing
        $status = $response.Content | ConvertFrom-Json
        
        if ($status.weaviate -eq "connected") {
            $apiReady = $true
            Write-Host "API successfully connected to Weaviate!" -ForegroundColor Green
        } else {
            Write-Host "  API not fully connected yet, waiting..." -ForegroundColor Gray
            Start-Sleep -Seconds 3
        }
    } catch {
        Write-Host "  API not ready yet: $_" -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
}

if (-not $apiReady) {
    Write-Host "API did not connect to Weaviate properly within the timeout period." -ForegroundColor Yellow
    Write-Host "You may need to restart the API container: docker-compose restart api" -ForegroundColor Yellow
} else {
    Write-Host "All services started and connected successfully!" -ForegroundColor Green
}

# Display access information
Write-Host "Vue.js frontend: $frontendUrl" -ForegroundColor Cyan
Write-Host "API documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "System statistics: http://localhost:8000/statistics" -ForegroundColor Cyan
Write-Host "Weaviate console: http://localhost:8080" -ForegroundColor Cyan
Write-Host "Processor logs: docker-compose logs -f processor" -ForegroundColor Cyan
Write-Host "Vue.js logs: docker-compose logs -f vue-frontend" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Cyan