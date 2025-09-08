#!/bin/bash

# PodPromo AI - Smoke Tests
# Comprehensive manual testing script for all critical endpoints

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_URL="http://localhost:8000"
TEST_FILE=""
EPISODE_ID=""

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed or not in PATH"
        exit 1
    fi
}

# Check required commands
check_command curl
check_command jq

log_info "Starting PodPromo AI Smoke Tests"
log_info "Base URL: $BASE_URL"

# Test 1: Health Check
log_info "Test 1: Health Check"
if curl -sS "$BASE_URL/health" | jq -e '.status == "healthy"' > /dev/null; then
    log_success "Health check passed"
else
    log_error "Health check failed"
    exit 1
fi

# Test 2: Progress endpoint never 500s
log_info "Test 2: Progress endpoint resilience"
log_info "Testing nonexistent episode (should return 404 JSON, not 500)"

RESPONSE=$(curl -sS -i "$BASE_URL/api/progress/does-not-exist")
HTTP_CODE=$(echo "$RESPONSE" | head -n 1 | cut -d' ' -f2)
CONTENT_TYPE=$(echo "$RESPONSE" | grep -i "content-type" | head -n 1)

if [[ "$HTTP_CODE" == "404" ]] && [[ "$CONTENT_TYPE" == *"application/json"* ]]; then
    log_success "Progress endpoint correctly returns 404 JSON for nonexistent episodes"
else
    log_error "Progress endpoint failed: HTTP $HTTP_CODE, Content-Type: $CONTENT_TYPE"
    echo "Response:"
    echo "$RESPONSE" | head -n 20
fi

# Test 3: Upload flow (if test file provided)
if [[ -n "$1" && -f "$1" ]]; then
    TEST_FILE="$1"
    log_info "Test 3: Upload → Progress → Completion flow"
    log_info "Using test file: $TEST_FILE"
    
    # Upload file
    log_info "Uploading file..."
    UPLOAD_RESPONSE=$(curl -sS -F "file=@$TEST_FILE" "$BASE_URL/api/upload")
    EPISODE_ID=$(echo "$UPLOAD_RESPONSE" | jq -r '.episodeId')
    
    if [[ "$EPISODE_ID" == "null" || -z "$EPISODE_ID" ]]; then
        log_error "Upload failed - no episode ID returned"
        echo "Upload response: $UPLOAD_RESPONSE"
        exit 1
    fi
    
    log_success "File uploaded successfully. Episode ID: $EPISODE_ID"
    
    # Poll for completion
    log_info "Polling for completion (up to 60 attempts, 2s intervals)..."
    COMPLETED=false
    
    for i in {1..60}; do
        PROGRESS_RESPONSE=$(curl -sS "$BASE_URL/api/progress/$EPISODE_ID" || echo '{}')
        STAGE=$(echo "$PROGRESS_RESPONSE" | jq -r '.progress.stage // "unknown"')
        PERCENTAGE=$(echo "$PROGRESS_RESPONSE" | jq -r '.progress.percentage // 0')
        
        log_info "Poll $i: Stage=$STAGE, Progress=$PERCENTAGE%"
        
        if [[ "$STAGE" == "completed" ]]; then
            COMPLETED=true
            log_success "Processing completed!"
            break
        fi
        
        sleep 2
    done
    
    if [[ "$COMPLETED" == false ]]; then
        log_warning "Processing did not complete within 2 minutes"
    fi
    
    # Test clips endpoint
    log_info "Testing clips endpoint..."
    CLIPS_RESPONSE=$(curl -sS "$BASE_URL/api/episodes/$EPISODE_ID/clips")
    CLIPS_COUNT=$(echo "$CLIPS_RESPONSE" | jq -r '.clips | length')
    
    if [[ "$CLIPS_COUNT" -gt 0 ]]; then
        log_success "Clips endpoint returned $CLIPS_COUNT clips"
        
        # Test preview URLs
        log_info "Testing preview URLs..."
        PREVIEW_URL=$(echo "$CLIPS_RESPONSE" | jq -r '.clips[0].previewUrl // empty')
        if [[ -n "$PREVIEW_URL" ]]; then
            log_info "Testing preview URL: $PREVIEW_URL"
            PREVIEW_RESPONSE=$(curl -sS -I "$BASE_URL$PREVIEW_URL")
            PREVIEW_HTTP_CODE=$(echo "$PREVIEW_RESPONSE" | head -n 1 | cut -d' ' -f2)
            PREVIEW_CONTENT_TYPE=$(echo "$PREVIEW_RESPONSE" | grep -i "content-type" | head -n 1)
            
            if [[ "$PREVIEW_HTTP_CODE" == "200" ]] && [[ "$PREVIEW_CONTENT_TYPE" == *"audio/mp4"* ]]; then
                log_success "Preview URL working correctly (HTTP 200, audio/mp4)"
            else
                log_warning "Preview URL issues: HTTP $PREVIEW_HTTP_CODE, Content-Type: $PREVIEW_CONTENT_TYPE"
            fi
            
            # Test range request
            log_info "Testing range request for preview..."
            RANGE_RESPONSE=$(curl -sS -I -H "Range: bytes=0-1023" "$BASE_URL$PREVIEW_URL")
            RANGE_HTTP_CODE=$(echo "$RANGE_RESPONSE" | head -n 1 | cut -d' ' -f2)
            
            if [[ "$RANGE_HTTP_CODE" == "206" ]]; then
                log_success "Range request working (HTTP 206 Partial Content)"
            else
                log_warning "Range request not supported (HTTP $RANGE_HTTP_CODE)"
            fi
        else
            log_warning "No preview URL found in clips"
        fi
    else
        log_warning "No clips returned"
    fi
    
else
    log_warning "No test file provided. Skipping upload tests."
    log_info "Usage: $0 <path-to-test-audio-file>"
fi

# Test 4: Static file serving
log_info "Test 4: Static file serving headers"
log_info "Testing MIME types and cache headers..."

# Test a few common file types (these might not exist, but we test the headers)
for ext in ".m4a" ".mp4" ".vtt" ".srt"; do
    TEST_PATH="/clips/previews/test$ext"
    log_info "Testing $ext files..."
    
    RESPONSE=$(curl -sS -I "$BASE_URL$TEST_PATH" || true)
    HTTP_CODE=$(echo "$RESPONSE" | head -n 1 | cut -d' ' -f2)
    
    if [[ "$HTTP_CODE" == "404" ]]; then
        log_info "$ext: 404 (expected for test file)"
    elif [[ "$HTTP_CODE" == "200" ]]; then
        CACHE_HEADER=$(echo "$RESPONSE" | grep -i "cache-control" | head -n 1)
        CONTENT_TYPE=$(echo "$RESPONSE" | grep -i "content-type" | head -n 1)
        
        if [[ "$CACHE_HEADER" == *"max-age"* ]]; then
            log_success "$ext: Cache headers present"
        else
            log_warning "$ext: Missing cache headers"
        fi
        
        if [[ "$ext" == ".m4a" && "$CONTENT_TYPE" == *"audio/mp4"* ]]; then
            log_success "$ext: Correct MIME type (audio/mp4)"
        elif [[ "$ext" == ".vtt" && "$CONTENT_TYPE" == *"text/vtt"* ]]; then
            log_success "$ext: Correct MIME type (text/vtt)"
        else
            log_info "$ext: Content-Type: $CONTENT_TYPE"
        fi
    else
        log_warning "$ext: Unexpected HTTP code $HTTP_CODE"
    fi
done

# Test 5: Error handling
log_info "Test 5: Error handling"
log_info "Testing various error conditions..."

# Test invalid endpoints
INVALID_RESPONSE=$(curl -sS -i "$BASE_URL/api/invalid-endpoint" || true)
INVALID_HTTP_CODE=$(echo "$INVALID_RESPONSE" | head -n 1 | cut -d' ' -f2)

if [[ "$INVALID_HTTP_CODE" == "404" ]]; then
    log_success "Invalid endpoints return 404"
else
    log_warning "Invalid endpoint returned HTTP $INVALID_HTTP_CODE"
fi

# Test malformed requests
log_info "Testing malformed requests..."
MALFORMED_RESPONSE=$(curl -sS -X POST "$BASE_URL/api/upload" -H "Content-Type: application/json" -d '{"invalid": "data"}' || true)
MALFORMED_HTTP_CODE=$(echo "$MALFORMED_RESPONSE" | head -n 1 | cut -d' ' -f2)

if [[ "$MALFORMED_HTTP_CODE" == "422" || "$MALFORMED_HTTP_CODE" == "400" ]]; then
    log_success "Malformed requests handled gracefully (HTTP $MALFORMED_HTTP_CODE)"
else
    log_warning "Malformed request returned HTTP $MALFORMED_HTTP_CODE"
fi

# Summary
log_info "Smoke tests completed!"
log_info "Check the output above for any errors or warnings."

if [[ -n "$EPISODE_ID" ]]; then
    log_info "Test episode ID: $EPISODE_ID"
    log_info "You can manually test the frontend at: http://localhost:3000?episodeId=$EPISODE_ID"
fi

log_info "For automated testing, run: pytest tests/test_progress.py -v"
