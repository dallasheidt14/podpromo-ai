"""
Queue Manager - Handles request queuing and concurrency management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import uuid
from dataclasses import dataclass
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    """Job priority enumeration"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Job:
    """Job representation"""
    id: str
    type: str
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class QueueManager:
    """Manages job queuing and execution"""
    
    def __init__(self, max_concurrent_jobs: int = 3, max_queue_size: int = 100):
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_queue_size = max_queue_size
        self.jobs: Dict[str, Job] = {}
        self.queue: List[str] = []  # Job IDs in priority order
        self.running_jobs: set = set()
        self.job_handlers: Dict[str, Callable] = {}
        self.lock = asyncio.Lock()
        
        # Start the queue processor
        asyncio.create_task(self._process_queue())
        
        logger.info(f"Queue manager initialized with {max_concurrent_jobs} concurrent jobs, max queue size {max_queue_size}")
    
    async def submit_job(self, job_type: str, priority: JobPriority = JobPriority.NORMAL, 
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a new job to the queue"""
        async with self.lock:
            # Check queue size limit
            if len(self.queue) >= self.max_queue_size:
                raise RuntimeError(f"Queue is full ({self.max_queue_size} jobs)")
            
            # Create job
            job_id = str(uuid.uuid4())
            job = Job(
                id=job_id,
                type=job_type,
                priority=priority,
                status=JobStatus.QUEUED,
                created_at=datetime.now(),
                metadata=metadata or {}
            )
            
            # Add to jobs and queue
            self.jobs[job_id] = job
            self._insert_into_queue(job_id, priority)
            
            logger.info(f"Job {job_id} ({job_type}) queued with priority {priority.name}")
            return job_id
    
    def _insert_into_queue(self, job_id: str, priority: JobPriority):
        """Insert job into queue maintaining priority order"""
        # Find insertion point based on priority
        insert_index = 0
        for i, queued_id in enumerate(self.queue):
            queued_job = self.jobs[queued_id]
            if priority.value > queued_job.priority.value:
                insert_index = i
                break
            insert_index = i + 1
        
        self.queue.insert(insert_index, job_id)
    
    async def get_job_status(self, job_id: str) -> Optional[Job]:
        """Get current status of a job"""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job"""
        async with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            # Remove from queue if queued
            if job_id in self.queue:
                self.queue.remove(job_id)
            
            # Stop if running
            if job_id in self.running_jobs:
                self.running_jobs.remove(job_id)
            
            # Update status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now()
            
            logger.info(f"Job {job_id} cancelled")
            return True
    
    async def _process_queue(self):
        """Background task to process the job queue"""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                async with self.lock:
                    # Check if we can start more jobs
                    if len(self.running_jobs) >= self.max_concurrent_jobs:
                        continue
                    
                    # Get next job from queue
                    if not self.queue:
                        continue
                    
                    job_id = self.queue.pop(0)
                    job = self.jobs[job_id]
                    
                    # Skip if job was cancelled
                    if job.status == JobStatus.CANCELLED:
                        continue
                    
                    # Start the job
                    await self._start_job(job_id)
                    
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    async def _start_job(self, job_id: str):
        """Start execution of a job"""
        job = self.jobs[job_id]
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.now()
        self.running_jobs.add(job_id)
        
        logger.info(f"Starting job {job_id} ({job.type})")
        
        # Execute job in background
        asyncio.create_task(self._execute_job(job_id))
    
    async def _execute_job(self, job_id: str):
        """Execute a job and update its status"""
        job = self.jobs[job_id]
        
        try:
            # Get handler for job type
            handler = self.job_handlers.get(job.type)
            if not handler:
                raise RuntimeError(f"No handler registered for job type: {job.type}")
            
            # Execute job
            result = await handler(job)
            
            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = datetime.now()
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            # Mark as failed
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"Job {job_id} failed: {e}")
        
        finally:
            # Remove from running jobs
            self.running_jobs.discard(job_id)
    
    def register_handler(self, job_type: str, handler: Callable):
        """Register a handler function for a job type"""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self.lock:
            return {
                "total_jobs": len(self.jobs),
                "queued_jobs": len(self.queue),
                "running_jobs": len(self.running_jobs),
                "completed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.COMPLETED]),
                "failed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
                "cancelled_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.CANCELLED]),
                "max_concurrent": self.max_concurrent_jobs,
                "max_queue_size": self.max_queue_size
            }
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed jobs"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        jobs_to_remove = []
        
        async with self.lock:
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                if job_id in self.queue:
                    self.queue.remove(job_id)
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
    
    @asynccontextmanager
    async def job_context(self, job_id: str):
        """Context manager for job execution with progress updates"""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        try:
            yield job
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            raise
        finally:
            if job_id in self.running_jobs:
                self.running_jobs.discard(job_id)
