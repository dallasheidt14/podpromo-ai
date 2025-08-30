"""
File Management Service - Handles file cleanup, validation, and storage management
"""

import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles

logger = logging.getLogger(__name__)

class FileManager:
    """Manages file storage, cleanup, and validation"""
    
    def __init__(self, upload_dir: str = "./uploads", output_dir: str = "./outputs"):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.max_file_size = 500 * 1024 * 1024  # 500MB
        self.max_storage_gb = 10  # 10GB total storage limit
        self.cleanup_interval_hours = 24
        
        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start background cleanup task
        asyncio.create_task(self._background_cleanup())
    
    async def validate_upload(self, file_path: str, file_size: int) -> Dict[str, any]:
        """Validate uploaded file before processing"""
        try:
            # Check file size
            if file_size > self.max_file_size:
                return {
                    "valid": False,
                    "error": f"File too large: {file_size / (1024*1024):.1f}MB exceeds {self.max_file_size / (1024*1024):.1f}MB limit"
                }
            
            # Check file extension
            allowed_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi'}
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in allowed_extensions:
                return {
                    "valid": False,
                    "error": f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
                }
            
            # Check available storage
            if not await self._check_storage_available(file_size):
                return {
                    "valid": False,
                    "error": "Insufficient storage space available"
                }
            
            return {"valid": True, "message": "File validation passed"}
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _check_storage_available(self, required_bytes: int) -> bool:
        """Check if sufficient storage is available"""
        try:
            # Get current storage usage
            current_usage = await self._get_storage_usage()
            available_gb = self.max_storage_gb - current_usage
            
            # Convert required bytes to GB
            required_gb = required_bytes / (1024 * 1024 * 1024)
            
            return available_gb >= required_gb
            
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return False  # Fail safe
    
    async def _get_storage_usage(self) -> float:
        """Get current storage usage in GB"""
        try:
            total_size = 0
            
            # Calculate upload directory size
            for file_path in self.upload_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            # Calculate output directory size
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024 * 1024)  # Convert to GB
            
        except Exception as e:
            logger.error(f"Storage usage calculation failed: {e}")
            return 0.0
    
    async def cleanup_old_files(self, max_age_hours: int = None):
        """Clean up old files to free storage"""
        if max_age_hours is None:
            max_age_hours = self.cleanup_interval_hours
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_files = []
            freed_space = 0
            
            # Clean uploads directory
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                        freed_space += file_size
                        logger.info(f"Cleaned up old upload: {file_path}")
            
            # Clean outputs directory (keep recent clips)
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_files.append(str(file_path))
                        freed_space += file_size
                        logger.info(f"Cleaned up old output: {file_path}")
            
            if cleaned_files:
                freed_gb = freed_space / (1024 * 1024 * 1024)
                logger.info(f"Cleanup completed: {len(cleaned_files)} files removed, {freed_gb:.2f}GB freed")
            
            return {
                "files_removed": len(cleaned_files),
                "space_freed_gb": freed_space / (1024 * 1024 * 1024),
                "files": cleaned_files
            }
            
        except Exception as e:
            logger.error(f"File cleanup failed: {e}")
            return {"error": str(e)}
    
    async def _background_cleanup(self):
        """Background task for periodic file cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_files()
            except Exception as e:
                logger.error(f"Background cleanup failed: {e}")
    
    async def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics"""
        try:
            upload_size = sum(f.stat().st_size for f in self.upload_dir.rglob('*') if f.is_file())
            output_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())
            total_size = upload_size + output_size
            
            return {
                "upload_size_gb": upload_size / (1024 * 1024 * 1024),
                "output_size_gb": output_size / (1024 * 1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "max_storage_gb": self.max_storage_gb,
                "available_gb": self.max_storage_gb - (total_size / (1024 * 1024 * 1024)),
                "usage_percent": (total_size / (1024 * 1024 * 1024)) / self.max_storage_gb * 100
            }
        except Exception as e:
            logger.error(f"Storage stats failed: {e}")
            return {"error": str(e)}
    
    async def move_to_archive(self, file_path: str, archive_dir: str = "./archive") -> bool:
        """Move file to archive instead of deleting"""
        try:
            archive_path = Path(archive_dir)
            archive_path.mkdir(parents=True, exist_ok=True)
            
            source_path = Path(file_path)
            if not source_path.exists():
                return False
            
            # Create archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"{timestamp}_{source_path.name}"
            archive_file_path = archive_path / archive_filename
            
            # Move file
            shutil.move(str(source_path), str(archive_file_path))
            logger.info(f"Archived file: {file_path} -> {archive_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Archive move failed: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, any]]:
        """Get detailed file information"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            return {
                "name": path.name,
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "extension": path.suffix.lower(),
                "is_file": path.is_file(),
                "is_dir": path.is_dir()
            }
        except Exception as e:
            logger.error(f"File info failed: {e}")
            return None
