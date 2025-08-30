"""
Monitoring Service - Performance metrics, error aggregation, and system health
"""

import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import statistics
import psutil
import threading
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Metric type enumeration"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """Metric representation"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecord:
    """Error record for aggregation"""
    error_type: str
    message: str
    timestamp: datetime
    count: int = 1
    last_occurrence: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

class MonitoringService:
    """Central monitoring and metrics service"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.errors: Dict[str, ErrorRecord] = {}
        self.health_checks: Dict[str, callable] = {}
        self.performance_timers: Dict[str, List[float]] = defaultdict(list)
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_usage": [],
            "network_io": []
        }
        
        # Start background monitoring
        self._start_background_monitoring()
        
        logger.info("Monitoring service initialized")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE, 
                     tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric"""
        try:
            metric = Metric(
                name=name,
                type=metric_type,
                value=value,
                timestamp=datetime.now(),
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].append(metric)
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def record_error(self, error_type: str, message: str, stack_trace: str = None, 
                    context: Dict[str, Any] = None):
        """Record an error for aggregation"""
        try:
            if error_type in self.errors:
                # Update existing error record
                error_record = self.errors[error_type]
                error_record.count += 1
                error_record.last_occurrence = datetime.now()
                if stack_trace:
                    error_record.stack_trace = stack_trace
                if context:
                    error_record.context.update(context)
            else:
                # Create new error record
                self.errors[error_type] = ErrorRecord(
                    error_type=error_type,
                    message=message,
                    timestamp=datetime.now(),
                    stack_trace=stack_trace,
                    context=context or {}
                )
            
            # Also record as metric
            self.record_metric("errors_total", 1, MetricType.COUNTER, {"error_type": error_type})
            
        except Exception as e:
            logger.error(f"Failed to record error {error_type}: {e}")
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        return OperationTimer(self, operation_name)
    
    def record_timing(self, operation_name: str, duration: float):
        """Record operation timing"""
        self.performance_timers[operation_name].append(duration)
        
        # Keep only recent timings
        if len(self.performance_timers[operation_name]) > 100:
            self.performance_timers[operation_name] = self.performance_timers[operation_name][-100:]
        
        # Record as metric
        self.record_metric(f"{operation_name}_duration", duration, MetricType.HISTOGRAM)
    
    def register_health_check(self, name: str, check_function: callable):
        """Register a health check function"""
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.record_error("health_check_failed", str(e), context={"check_name": name})
        
        return results
    
    def get_metrics_summary(self, metric_name: str = None, 
                           time_window_hours: int = None) -> Dict[str, Any]:
        """Get summary statistics for metrics"""
        try:
            if metric_name:
                metrics = self.metrics.get(metric_name, [])
            else:
                # Aggregate all metrics
                all_metrics = []
                for metric_list in self.metrics.values():
                    all_metrics.extend(metric_list)
                metrics = all_metrics
            
            if not metrics:
                return {"error": "No metrics found"}
            
            # Apply time window if specified
            if time_window_hours:
                cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
                metrics = [m for m in metrics if m.timestamp > cutoff_time]
            
            if not metrics:
                return {"error": "No metrics in specified time window"}
            
            values = [m.value for m in metrics]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "time_window_hours": time_window_hours,
                "metric_name": metric_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recorded errors"""
        try:
            total_errors = sum(error.count for error in self.errors.values())
            
            # Group by error type
            error_summary = {}
            for error_type, error_record in self.errors.items():
                error_summary[error_type] = {
                    "count": error_record.count,
                    "message": error_record.message,
                    "first_occurrence": error_record.timestamp.isoformat(),
                    "last_occurrence": error_record.last_occurrence.isoformat(),
                    "context": error_record.context
                }
            
            return {
                "total_errors": total_errors,
                "unique_error_types": len(self.errors),
                "errors": error_summary
            }
            
        except Exception as e:
            logger.error(f"Failed to get error summary: {e}")
            return {"error": str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance timing summary"""
        try:
            performance_summary = {}
            
            for operation_name, timings in self.performance_timers.items():
                if timings:
                    performance_summary[operation_name] = {
                        "count": len(timings),
                        "min_ms": min(timings) * 1000,
                        "max_ms": max(timings) * 1000,
                        "mean_ms": statistics.mean(timings) * 1000,
                        "median_ms": statistics.median(timings) * 1000,
                        "p95_ms": statistics.quantiles(timings, n=20)[18] * 1000 if len(timings) >= 20 else None
                    }
            
            return performance_summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
    
    def _start_background_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while True:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_metric("system_cpu_percent", cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.record_metric("system_memory_percent", memory.percent)
                    self.record_metric("system_memory_available_gb", memory.available / (1024**3))
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.record_metric("system_disk_percent", disk.percent)
                    self.record_metric("system_disk_free_gb", disk.free / (1024**3))
                    
                    time.sleep(60)  # Update every minute
                    
                except Exception as e:
                    logger.error(f"System monitoring failed: {e}")
                    time.sleep(60)
        
        # Start in background thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    async def cleanup_old_data(self):
        """Clean up old metrics and error data"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            # Clean old metrics
            for metric_name, metric_list in self.metrics.items():
                # Remove old metrics
                while metric_list and metric_list[0].timestamp < cutoff_time:
                    metric_list.popleft()
            
            # Clean old errors (keep only recent ones)
            errors_to_remove = []
            for error_type, error_record in self.errors.items():
                if error_record.last_occurrence < cutoff_time:
                    errors_to_remove.append(error_type)
            
            for error_type in errors_to_remove:
                del self.errors[error_type]
            
            if errors_to_remove:
                logger.info(f"Cleaned up {len(errors_to_remove)} old error records")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")

class OperationTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitoring_service: MonitoringService, operation_name: str):
        self.monitoring_service = monitoring_service
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitoring_service.record_timing(self.operation_name, duration)
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitoring_service.record_timing(self.operation_name, duration)
