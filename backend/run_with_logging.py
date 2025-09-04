#!/usr/bin/env python3
"""
Backend server wrapper with better error handling and logging
"""
import sys
import traceback
import logging
import uvicorn

# Import here to avoid circular imports
try:
    from main import app
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the backend directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend.log')
    ]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Starting backend server with enhanced logging...")
        logger.info("Server will be available at http://localhost:8000")
        logger.info("Health check: http://localhost:8000/health")
        
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000, 
            log_level="info",
            access_log=True,
            reload=False  # Disable reload to prevent interruptions during scoring
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)
