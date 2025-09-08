# PodPromo AI - Testing Guide

This document outlines the comprehensive testing setup for PodPromo AI, including automated tests and manual smoke tests.

## Automated Tests

### Progress Service Tests

Run the comprehensive progress service tests:

```bash
# Run all progress tests
pytest tests/test_progress.py -v

# Run with coverage
pytest tests/test_progress.py --cov=backend.services.progress_service --cov-report=html
```

### Test Coverage

The progress tests cover:

- **Nonexistent episodes**: Returns 404 JSON, never 500
- **Valid progress files**: Proper JSON structure with all required fields
- **Corrupt files**: Graceful degradation, never crashes
- **Disk inference**: Handles missing progress files by inferring from disk state
- **Atomic writes**: Prevents corruption during concurrent access
- **Unicode handling**: Proper encoding of international characters
- **Large files**: Handles large progress data without issues
- **Concurrent access**: Multiple simultaneous requests work correctly

## Manual Smoke Tests

### Quick Smoke Test (Bash/Linux/Mac)

```bash
# Basic health check
./smoke_tests.sh

# Full test with audio file
./smoke_tests.sh /path/to/test-audio.mp3
```

### Windows Smoke Test

```cmd
REM Basic health check
smoke_tests.bat

REM Full test with audio file
smoke_tests.bat C:\path\to\test-audio.mp3
```

### Manual Testing Checklist

#### 1. Progress Endpoint Resilience
- [ ] Nonexistent episode returns 404 JSON (not HTML, not 500)
- [ ] Corrupt progress files don't cause 500 errors
- [ ] Missing directories handled gracefully

#### 2. Upload → Progress → Completion Flow
- [ ] Upload small test file
- [ ] Immediately polling allows 404 as "not ready"
- [ ] Eventually produces "completed" status
- [ ] Clips endpoint returns JSON with clips array

#### 3. Restart Resilience
- [ ] Kill server while job is "processing"
- [ ] Restart server
- [ ] Keep polling - should still return JSON, not 500
- [ ] Progress should recover

#### 4. Previews + Captions
- [ ] Preview URLs return proper MIME types
- [ ] Cache headers present for static files
- [ ] Range requests work (206 Partial Content)
- [ ] Caption files (.vtt, .srt) have correct content types

#### 5. Frontend Checks
- [ ] Upload → queued → uploading → processing → completed
- [ ] No red error toasts during normal flow
- [ ] Network interruption during processing resumes correctly
- [ ] Preview audio plays
- [ ] Captions render correctly
- [ ] Resume banner appears on refresh
- [ ] "Start new upload" works immediately

## Test Data

### Sample Audio Files

For testing, use:
- Short audio files (30-60 seconds) for quick tests
- Longer files (5-10 minutes) for stress testing
- Various formats: MP3, WAV, M4A

### Expected Processing Times

- **Short files (1-2 min)**: 30-60 seconds
- **Medium files (5-10 min)**: 2-5 minutes  
- **Long files (30+ min)**: 5-15 minutes

## Debugging

### Common Issues

1. **Progress stuck at 0%**: Check backend logs for transcription errors
2. **No clips generated**: Verify audio quality and content
3. **Preview not loading**: Check file permissions and MIME types
4. **Frontend not updating**: Check WebSocket/polling connections

### Log Locations

- **Backend logs**: `backend/logs/` or console output
- **Frontend logs**: Browser developer console
- **Test logs**: pytest output with `-v` flag

### Health Check Endpoints

```bash
# Backend health
curl http://localhost:8000/health

# Progress for specific episode
curl http://localhost:8000/api/progress/{episode_id}

# Clips for episode
curl http://localhost:8000/api/episodes/{episode_id}/clips
```

## Performance Testing

### Load Testing

```bash
# Test concurrent uploads
for i in {1..5}; do
  curl -F "file=@test.mp3" http://localhost:8000/api/upload &
done
wait

# Test concurrent progress polling
for i in {1..10}; do
  curl http://localhost:8000/api/progress/{episode_id} &
done
wait
```

### Memory Testing

Monitor memory usage during:
- Large file uploads
- Long processing jobs
- Multiple concurrent requests

## Continuous Integration

The tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Progress Tests
  run: pytest tests/test_progress.py -v

- name: Run Smoke Tests
  run: ./smoke_tests.sh
```

## Troubleshooting

### Test Failures

1. **Check service status**: Ensure backend is running on port 8000
2. **Verify dependencies**: curl, jq, pytest installed
3. **Check file permissions**: Test files readable, output directories writable
4. **Review logs**: Look for specific error messages

### Performance Issues

1. **Database connections**: Check connection pooling
2. **File I/O**: Monitor disk space and permissions
3. **Memory usage**: Watch for memory leaks
4. **Network timeouts**: Adjust timeout settings if needed

## Contributing

When adding new features:

1. **Add tests**: Include both unit and integration tests
2. **Update smoke tests**: Add new endpoints to manual tests
3. **Document changes**: Update this guide with new test procedures
4. **Test edge cases**: Include error conditions and boundary cases
