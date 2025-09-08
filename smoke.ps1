param(
    [Parameter(Mandatory=$true)][string]$File, 
    [string]$Api="http://localhost:8000"
)

Write-Host "=== PodPromo Smoke Test ===" -ForegroundColor Green
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
Write-Host "3) upload" -ForegroundColor Yellow
try {
    # Create multipart form data for file upload
    $boundary = [System.Guid]::NewGuid().ToString()
    $fileBytes = [System.IO.File]::ReadAllBytes($File)
    $fileName = [System.IO.Path]::GetFileName($File)
    
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
        "Content-Type: application/octet-stream",
        "",
        [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($fileBytes),
        "--$boundary--"
    )
    
    $body = $bodyLines -join "`r`n"
    $bodyBytes = [System.Text.Encoding]::GetEncoding("iso-8859-1").GetBytes($body)
    
    $uploadResp = Invoke-WebRequest "$Api/api/upload" -Method Post -Body $bodyBytes -ContentType "multipart/form-data; boundary=$boundary" -TimeoutSec 30
    $uploadData = $uploadResp.Content | ConvertFrom-Json
    $ep = $uploadData.episodeId
    Write-Host "  -> episode: $ep" -ForegroundColor Green
} catch {
    Write-Host "  -> ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "4) progress until 100%" -ForegroundColor Yellow
$maxAttempts = 180
for ($i = 0; $i -lt $maxAttempts; $i++) {
    try {
        $progressResp = Invoke-WebRequest "$Api/api/progress/$ep" -UseBasicParsing -TimeoutSec 10
        $progressData = $progressResp.Content | ConvertFrom-Json
        $stage = $progressData.progress.stage
        $pct = [int]($progressData.progress.percent)
        $timestamp = Get-Date -Format "HH:mm:ss"
        Write-Host "  [$timestamp] $stage $pct%" -ForegroundColor Cyan
        
        if ($pct -ge 100) { 
            Write-Host "  -> COMPLETED!" -ForegroundColor Green
            break 
        }
        
        if ($i -eq $maxAttempts - 1) {
            Write-Host "  -> TIMEOUT after $maxAttempts attempts" -ForegroundColor Red
            exit 1
        }
        
        Start-Sleep -Seconds 5
    } catch {
        Write-Host "  -> ERROR: $($_.Exception.Message)" -ForegroundColor Red
        Start-Sleep -Seconds 5
    }
}

Write-Host ""
Write-Host "5) clips" -ForegroundColor Yellow
try {
    $clipsResp = Invoke-WebRequest "$Api/api/episodes/$ep/clips" -UseBasicParsing -TimeoutSec 10
    $clipsData = $clipsResp.Content | ConvertFrom-Json
    $clipCount = $clipsData.clips.Count
    Write-Host "  -> Found $clipCount clips" -ForegroundColor Green
    if ($clipCount -gt 0) {
        Write-Host "  -> Sample clip:" -ForegroundColor Gray
        $clipsData.clips[0] | ConvertTo-Json -Depth 2
    }
} catch {
    Write-Host "  -> ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "All tests passed! System is working correctly." -ForegroundColor Green
