@echo off
REM PodPromo AI - Smoke Tests (Windows Batch Version)
REM Comprehensive manual testing script for all critical endpoints

setlocal enabledelayedexpansion

REM Configuration
set BASE_URL=http://localhost:8000
set TEST_FILE=%1
set EPISODE_ID=

echo [INFO] Starting PodPromo AI Smoke Tests
echo [INFO] Base URL: %BASE_URL%

REM Check if curl is available
curl --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] curl is not installed or not in PATH
    exit /b 1
)

REM Check if jq is available
jq --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] jq is not installed or not in PATH
    exit /b 1
)

REM Test 1: Health Check
echo [INFO] Test 1: Health Check
curl -sS "%BASE_URL%/health" | jq -e ".status == \"healthy\"" >nul
if errorlevel 1 (
    echo [ERROR] Health check failed
    exit /b 1
) else (
    echo [SUCCESS] Health check passed
)

REM Test 2: Progress endpoint never 500s
echo [INFO] Test 2: Progress endpoint resilience
echo [INFO] Testing nonexistent episode (should return 404 JSON, not 500)

for /f "tokens=2" %%i in ('curl -sS -i "%BASE_URL%/api/progress/does-not-exist" ^| findstr "HTTP"') do set HTTP_CODE=%%i
curl -sS -i "%BASE_URL%/api/progress/does-not-exist" | findstr "content-type" >nul
if errorlevel 1 (
    set CONTENT_TYPE=unknown
) else (
    set CONTENT_TYPE=application/json
)

if "%HTTP_CODE%"=="404" if "%CONTENT_TYPE%"=="application/json" (
    echo [SUCCESS] Progress endpoint correctly returns 404 JSON for nonexistent episodes
) else (
    echo [ERROR] Progress endpoint failed: HTTP %HTTP_CODE%, Content-Type: %CONTENT_TYPE%
)

REM Test 3: Upload flow (if test file provided)
if not "%TEST_FILE%"=="" (
    if exist "%TEST_FILE%" (
        echo [INFO] Test 3: Upload → Progress → Completion flow
        echo [INFO] Using test file: %TEST_FILE%
        
        REM Upload file
        echo [INFO] Uploading file...
        for /f "tokens=*" %%i in ('curl -sS -F "file=@%TEST_FILE%" "%BASE_URL%/api/upload" ^| jq -r ".episodeId"') do set EPISODE_ID=%%i
        
        if "%EPISODE_ID%"=="null" if "%EPISODE_ID%"=="" (
            echo [ERROR] Upload failed - no episode ID returned
            exit /b 1
        )
        
        echo [SUCCESS] File uploaded successfully. Episode ID: %EPISODE_ID%
        
        REM Poll for completion
        echo [INFO] Polling for completion (up to 60 attempts, 2s intervals)...
        set COMPLETED=false
        
        for /L %%i in (1,1,60) do (
            for /f "tokens=*" %%j in ('curl -sS "%BASE_URL%/api/progress/%EPISODE_ID%" 2^>nul ^| jq -r ".progress.stage // \"unknown\""') do set STAGE=%%j
            for /f "tokens=*" %%k in ('curl -sS "%BASE_URL%/api/progress/%EPISODE_ID%" 2^>nul ^| jq -r ".progress.percentage // 0"') do set PERCENTAGE=%%k
            
            echo [INFO] Poll %%i: Stage=!STAGE!, Progress=!PERCENTAGE!%%
            
            if "!STAGE!"=="completed" (
                set COMPLETED=true
                echo [SUCCESS] Processing completed!
                goto :clips_test
            )
            
            timeout /t 2 /nobreak >nul
        )
        
        :clips_test
        if "!COMPLETED!"=="false" (
            echo [WARNING] Processing did not complete within 2 minutes
        )
        
        REM Test clips endpoint
        echo [INFO] Testing clips endpoint...
        for /f "tokens=*" %%i in ('curl -sS "%BASE_URL%/api/episodes/%EPISODE_ID%/clips" ^| jq -r ".clips | length"') do set CLIPS_COUNT=%%i
        
        if !CLIPS_COUNT! gtr 0 (
            echo [SUCCESS] Clips endpoint returned !CLIPS_COUNT! clips
        ) else (
            echo [WARNING] No clips returned
        )
    ) else (
        echo [ERROR] Test file not found: %TEST_FILE%
    )
) else (
    echo [WARNING] No test file provided. Skipping upload tests.
    echo [INFO] Usage: %0 ^<path-to-test-audio-file^>
)

REM Test 4: Static file serving
echo [INFO] Test 4: Static file serving headers
echo [INFO] Testing MIME types and cache headers...

REM Test .m4a files
echo [INFO] Testing .m4a files...
for /f "tokens=2" %%i in ('curl -sS -I "%BASE_URL%/clips/previews/test.m4a" 2^>nul ^| findstr "HTTP"') do set HTTP_CODE=%%i
if "%HTTP_CODE%"=="200" (
    echo [SUCCESS] .m4a files accessible
) else (
    echo [INFO] .m4a: HTTP %HTTP_CODE% (expected for test file)
)

REM Test 5: Error handling
echo [INFO] Test 5: Error handling
echo [INFO] Testing various error conditions...

REM Test invalid endpoints
for /f "tokens=2" %%i in ('curl -sS -i "%BASE_URL%/api/invalid-endpoint" 2^>nul ^| findstr "HTTP"') do set INVALID_HTTP_CODE=%%i
if "%INVALID_HTTP_CODE%"=="404" (
    echo [SUCCESS] Invalid endpoints return 404
) else (
    echo [WARNING] Invalid endpoint returned HTTP %INVALID_HTTP_CODE%
)

REM Summary
echo [INFO] Smoke tests completed!
echo [INFO] Check the output above for any errors or warnings.

if not "%EPISODE_ID%"=="" (
    echo [INFO] Test episode ID: %EPISODE_ID%
    echo [INFO] You can manually test the frontend at: http://localhost:3000?episodeId=%EPISODE_ID%
)

echo [INFO] For automated testing, run: pytest tests/test_progress.py -v
