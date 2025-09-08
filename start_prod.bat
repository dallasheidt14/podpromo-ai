@echo off
echo Starting PodPromo Backend in Production Mode...
echo.

REM Set production environment
set ENVIRONMENT=production
set CORS_ORIGINS=http://localhost:3000,https://highlightly.ai

REM Start with production optimizations
echo Starting with 2 workers, no access logs, no reload...
python -m uvicorn main:app --workers 2 --host 0.0.0.0 --port 8000 --no-access-log

echo.
echo Backend started in production mode!
echo Health: http://localhost:8000/health
echo Ready:  http://localhost:8000/ready
echo API:    http://localhost:8000/api/upload
