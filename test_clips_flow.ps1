# Test script for the "See my viral clips" flow
param([string]$Api="http://localhost:8000", [string]$Frontend="http://localhost:3000")

Write-Host "=== Testing 'See my viral clips' Flow ===" -ForegroundColor Green

# Test 1: Health check
Write-Host "`n1) Testing health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest "$Api/health" -UseBasicParsing
    Write-Host "   ✓ Health: $($health.StatusCode) in $($health.Headers.'X-Response-Time')" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Health failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Test 2: Upload a test file (if available)
Write-Host "`n2) Testing upload..." -ForegroundColor Yellow
$testFile = "..\backend\uploads\5ecf9042-7864-4671-95d2-cbdbcdeced47.mp3"
if (Test-Path $testFile) {
    try {
        $uploadResp = Invoke-WebRequest "$Api/api/upload" -Method Post -Form @{ file = Get-Item $testFile } -UseBasicParsing
        $uploadData = $uploadResp.Content | ConvertFrom-Json
        $episodeId = $uploadData.episodeId
        Write-Host "   ✓ Upload successful: $episodeId" -ForegroundColor Green
    } catch {
        Write-Host "   ✗ Upload failed: $($_.Exception.Message)" -ForegroundColor Red
        # Use existing episode for testing
        $episodeId = "74203114-9c20-497b-8b36-6faaa0771e70"
        Write-Host "   → Using existing episode: $episodeId" -ForegroundColor Yellow
    }
} else {
    Write-Host "   → No test file found, using existing episode" -ForegroundColor Yellow
    $episodeId = "74203114-9c20-497b-8b36-6faaa0771e70"
}

# Test 3: Check progress
Write-Host "`n3) Testing progress endpoint..." -ForegroundColor Yellow
try {
    $progressResp = Invoke-WebRequest "$Api/api/progress/$episodeId" -UseBasicParsing
    $progressData = $progressResp.Content | ConvertFrom-Json
    $stage = $progressData.progress.stage
    $percent = $progressData.progress.percent
    Write-Host "   ✓ Progress: $stage ($($percent)%)" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Progress failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Test HEAD endpoint for readiness
Write-Host "`n4) Testing HEAD endpoint for readiness..." -ForegroundColor Yellow
try {
    $headResp = Invoke-WebRequest "$Api/api/episodes/$episodeId/clips" -Method HEAD -UseBasicParsing
    $status = $headResp.StatusCode
    $retryAfter = $headResp.Headers.'Retry-After'
    if ($status -eq 202) {
        Write-Host "   ✓ HEAD: 202 (not ready, retry after $retryAfter seconds)" -ForegroundColor Yellow
    } elseif ($status -eq 204) {
        Write-Host "   ✓ HEAD: 204 (ready)" -ForegroundColor Green
    } else {
        Write-Host "   ? HEAD: $status" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ✗ HEAD failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 5: Test GET endpoint for clips
Write-Host "`n5) Testing GET endpoint for clips..." -ForegroundColor Yellow
try {
    $clipsResp = Invoke-WebRequest "$Api/api/episodes/$episodeId/clips" -UseBasicParsing
    $status = $clipsResp.StatusCode
    $clipsData = $clipsResp.Content | ConvertFrom-Json
    
    if ($status -eq 202) {
        Write-Host "   ✓ GET: 202 (not ready) - $($clipsData.message)" -ForegroundColor Yellow
        Write-Host "   → Retry-After: $($clipsResp.Headers.'Retry-After') seconds" -ForegroundColor Cyan
    } elseif ($status -eq 200) {
        $clipsCount = $clipsData.clips.Count
        Write-Host "   ✓ GET: 200 (ready) - $clipsCount clips available" -ForegroundColor Green
        if ($clipsCount -gt 0) {
            $firstClip = $clipsData.clips[0]
            Write-Host "   → First clip: $($firstClip.title.Substring(0, [Math]::Min(50, $firstClip.title.Length)))..." -ForegroundColor Cyan
        }
    } else {
        Write-Host "   ? GET: $status" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ✗ GET failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 6: Test frontend accessibility
Write-Host "`n6) Testing frontend accessibility..." -ForegroundColor Yellow
try {
    $frontendResp = Invoke-WebRequest $Frontend -UseBasicParsing
    Write-Host "   ✓ Frontend: $($frontendResp.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "   ✗ Frontend: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Green
Write-Host "`nTo test the complete flow:" -ForegroundColor Cyan
Write-Host "1. Open $Frontend in your browser" -ForegroundColor White
Write-Host "2. Upload a file and wait for it to reach 'scoring' stage" -ForegroundColor White
Write-Host "3. Click the 'See my viral clips' button" -ForegroundColor White
Write-Host "4. The button should show 'Checking...' and then either:" -ForegroundColor White
Write-Host "   - Show clips if ready (200 response)" -ForegroundColor White
Write-Host "   - Show 'Still scoring...' with auto-retry if not ready (202 response)" -ForegroundColor White
