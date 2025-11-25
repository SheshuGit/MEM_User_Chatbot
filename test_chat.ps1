# PowerShell script to test the chatbot API

Write-Host "Testing Wedding Chatbot API..." -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "1. Testing Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method GET
    Write-Host "✓ Health Check: " -ForegroundColor Green -NoNewline
    Write-Host ($health | ConvertTo-Json)
} catch {
    Write-Host "✗ Health Check Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 2: Chat - Photographers
Write-Host "2. Testing Chat: Show me photographers..." -ForegroundColor Yellow
try {
    $body = @{
        question = "Show me photographers in Bengaluru"
        session_id = "test_session_123"
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/chat" -Method POST -ContentType "application/json" -Body $body
    Write-Host "✓ Chat Response:" -ForegroundColor Green
    Write-Host $response.answer
} catch {
    Write-Host "✗ Chat Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 3: Chat - Makeup Artists
Write-Host "3. Testing Chat: I need makeup artists..." -ForegroundColor Yellow
try {
    $body = @{
        question = "I need makeup artists"
        session_id = "test_session_123"
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "http://localhost:5000/api/chat" -Method POST -ContentType "application/json" -Body $body
    Write-Host "✓ Chat Response:" -ForegroundColor Green
    Write-Host $response.answer
} catch {
    Write-Host "✗ Chat Failed: $_" -ForegroundColor Red
}

Write-Host ""

# Test 4: Categories
Write-Host "4. Testing Categories..." -ForegroundColor Yellow
try {
    $categories = Invoke-RestMethod -Uri "http://localhost:5000/api/categories" -Method GET
    Write-Host "✓ Categories: " -ForegroundColor Green -NoNewline
    Write-Host ($categories.categories -join ", ")
} catch {
    Write-Host "✗ Categories Failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Testing Complete!" -ForegroundColor Cyan
