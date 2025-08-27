@echo off
echo 🚀 Starting PodPromo AI Development Environment...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker and try again.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

REM Start the services
echo 🐳 Starting Docker services...
docker-compose up --build -d

echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

echo 🏥 Checking service health...

REM Check backend
curl -f http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend is running at http://localhost:8000
) else (
    echo ❌ Backend is not responding
)

REM Check frontend
curl -f http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend is running at http://localhost:3000
) else (
    echo ❌ Frontend is not responding
)

echo.
echo 🎉 PodPromo AI is starting up!
echo.
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo 📁 Uploads directory: ./uploads
echo 📁 Outputs directory: ./outputs
echo.
echo To stop the services, run: docker-compose down
echo To view logs, run: docker-compose logs -f
echo.
pause
