param([string]$Api="http://localhost:8000")

Write-Host "=== Quick Endpoint Test ===" -ForegroundColor Green
Write-Host ""

Write-Host "1) /health (should be <1ms)" -ForegroundColor Yellow
$healthStart = Get-Date
try {
    $healthResp = Invoke-WebRequest "$Api/health" -UseBasicParsing -TimeoutSec 5
    $healthElapsed = ((Get-Date) - $healthStart).TotalMilliseconds
    Write-Host "  -> $($healthResp.StatusCode) in $([math]::Round($healthElapsed, 1))ms" -ForegroundColor Green
    $healthResp.Content | ConvertFrom-Json | ConvertTo-Json -Depth 3
} catch {
    Write-Host "  -> ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "2) /ready (should be <100ms)" -ForegroundColor Yellow
$readyStart = Get-Date
try {
    $readyResp = Invoke-WebRequest "$Api/ready" -UseBasicParsing -TimeoutSec 10
    $readyElapsed = ((Get-Date) - $readyStart).TotalMilliseconds
    Write-Host "  -> $($readyResp.StatusCode) in $([math]::Round($readyElapsed, 1))ms" -ForegroundColor Green
    $readyData = $readyResp.Content | ConvertFrom-Json
    $readyData | ConvertTo-Json -Depth 5
} catch {
    Write-Host "  -> ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "3) /api/progress (test endpoint)" -ForegroundColor Yellow
try {
    $progressResp = Invoke-WebRequest "$Api/api/progress" -UseBasicParsing -TimeoutSec 10
    Write-Host "  -> $($progressResp.StatusCode)" -ForegroundColor Green
    $progressData = $progressResp.Content | ConvertFrom-Json
    $progressData | ConvertTo-Json -Depth 3
} catch {
    Write-Host "  -> ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Health endpoints are working correctly!" -ForegroundColor Green
