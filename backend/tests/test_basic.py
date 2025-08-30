"""
Basic test suite for the podcast clip generation system
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

# Import the services we want to test
from services.file_manager import FileManager
from services.queue_manager import QueueManager, JobPriority, JobStatus
from services.monitoring import MonitoringService, MetricType

class TestFileManager:
    """Test FileManager service"""
    
    @pytest.fixture
    def file_manager(self):
        """Create a FileManager instance for testing"""
        return FileManager(upload_dir="./test_uploads", output_dir="./test_outputs")
    
    @pytest.mark.asyncio
    async def test_file_validation_valid(self, file_manager):
        """Test file validation with valid file"""
        result = await file_manager.validate_upload("test.mp3", 1024 * 1024)  # 1MB
        assert result["valid"] is True
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_file_validation_too_large(self, file_manager):
        """Test file validation with file too large"""
        result = await file_manager.validate_upload("test.mp3", 600 * 1024 * 1024)  # 600MB
        assert result["valid"] is False
        assert "too large" in result["error"]
    
    @pytest.mark.asyncio
    async def test_file_validation_invalid_extension(self, file_manager):
        """Test file validation with invalid extension"""
        result = await file_manager.validate_upload("test.txt", 1024 * 1024)
        assert result["valid"] is False
        assert "Unsupported file type" in result["error"]

class TestQueueManager:
    """Test QueueManager service"""
    
    @pytest.fixture
    def queue_manager(self):
        """Create a QueueManager instance for testing"""
        return QueueManager(max_concurrent_jobs=2, max_queue_size=10)
    
    @pytest.mark.asyncio
    async def test_job_submission(self, queue_manager):
        """Test job submission"""
        job_id = await queue_manager.submit_job("test_job", JobPriority.NORMAL)
        assert job_id is not None
        assert len(job_id) > 0
        
        # Check job status
        job = await queue_manager.get_job_status(job_id)
        assert job is not None
        assert job.status == JobStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_job_cancellation(self, queue_manager):
        """Test job cancellation"""
        job_id = await queue_manager.submit_job("test_job", JobPriority.NORMAL)
        
        # Cancel the job
        success = await queue_manager.cancel_job(job_id)
        assert success is True
        
        # Check job status
        job = await queue_manager.get_job_status(job_id)
        assert job.status == JobStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_queue_size_limit(self, queue_manager):
        """Test queue size limit enforcement"""
        # Fill the queue
        job_ids = []
        for i in range(10):  # max_queue_size
            job_id = await queue_manager.submit_job(f"test_job_{i}", JobPriority.NORMAL)
            job_ids.append(job_id)
        
        # Try to submit one more
        with pytest.raises(RuntimeError, match="Queue is full"):
            await queue_manager.submit_job("overflow_job", JobPriority.NORMAL)
    
    @pytest.mark.asyncio
    async def test_job_handler_registration(self, queue_manager):
        """Test job handler registration"""
        mock_handler = Mock()
        queue_manager.register_handler("test_job_type", mock_handler)
        
        assert "test_job_type" in queue_manager.job_handlers
        assert queue_manager.job_handlers["test_job_type"] == mock_handler

class TestMonitoringService:
    """Test MonitoringService"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create a MonitoringService instance for testing"""
        return MonitoringService(retention_hours=1)
    
    def test_metric_recording(self, monitoring_service):
        """Test metric recording"""
        monitoring_service.record_metric("test_metric", 42.5, MetricType.GAUGE)
        
        # Check if metric was recorded
        summary = monitoring_service.get_metrics_summary("test_metric")
        assert summary["count"] == 1
        assert summary["mean"] == 42.5
    
    def test_error_recording(self, monitoring_service):
        """Test error recording"""
        monitoring_service.record_error("test_error", "Test error message")
        
        # Check if error was recorded
        error_summary = monitoring_service.get_error_summary()
        assert error_summary["total_errors"] == 1
        assert "test_error" in error_summary["errors"]
    
    def test_operation_timing(self, monitoring_service):
        """Test operation timing"""
        with monitoring_service.time_operation("test_operation"):
            # Simulate some work
            import time
            time.sleep(0.1)
        
        # Check if timing was recorded
        performance_summary = monitoring_service.get_performance_summary()
        assert "test_operation" in performance_summary
        assert performance_summary["test_operation"]["count"] == 1
    
    @pytest.mark.asyncio
    async def test_health_check_registration(self, monitoring_service):
        """Test health check registration and execution"""
        # Register a mock health check
        def mock_health_check():
            return True
        
        monitoring_service.register_health_check("test_check", mock_health_check)
        
        # Run health checks
        results = await monitoring_service.run_health_checks()
        assert "test_check" in results
        assert results["test_check"]["status"] == "healthy"

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_file_manager_with_queue_manager(self):
        """Test FileManager and QueueManager working together"""
        file_manager = FileManager()
        queue_manager = QueueManager()
        
        # Submit a file processing job
        job_id = await queue_manager.submit_job("file_processing", JobPriority.HIGH)
        
        # Check job status
        job = await queue_manager.get_job_status(job_id)
        assert job is not None
        assert job.priority == JobPriority.HIGH
    
    @pytest.mark.asyncio
    async def test_monitoring_with_services(self):
        """Test monitoring integration with other services"""
        monitoring_service = MonitoringService()
        queue_manager = QueueManager()
        
        # Record metrics from queue manager
        stats = await queue_manager.get_queue_stats()
        monitoring_service.record_metric("queue_total_jobs", stats["total_jobs"])
        
        # Check if metric was recorded
        summary = monitoring_service.get_metrics_summary("queue_total_jobs")
        assert summary["count"] == 1

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
