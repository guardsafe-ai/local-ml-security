"""
Worker Manager
Manages distributed workers for task execution
"""

import asyncio
import logging
import subprocess
import psutil
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import os
import signal
import threading

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status enumeration"""
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    IDLE = "IDLE"
    BUSY = "BUSY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class WorkerType(Enum):
    """Worker type enumeration"""
    ATTACK_WORKER = "attack_worker"
    ANALYSIS_WORKER = "analysis_worker"
    CERTIFICATION_WORKER = "certification_worker"
    PRIVACY_WORKER = "privacy_worker"
    GENERAL_WORKER = "general_worker"


@dataclass
class WorkerConfig:
    """Worker configuration"""
    worker_id: str
    worker_type: WorkerType
    concurrency: int = 1
    memory_limit: int = 2048  # MB
    cpu_limit: float = 1.0  # CPU cores
    gpu_enabled: bool = False
    gpu_memory_limit: int = 4096  # MB
    queue_name: str = "default"
    log_level: str = "INFO"
    max_tasks_per_child: int = 1000
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkerInfo:
    """Worker information"""
    worker_id: str
    worker_type: WorkerType
    status: WorkerStatus
    process_id: Optional[int] = None
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    memory_usage: float = 0.0  # MB
    cpu_usage: float = 0.0  # Percentage
    gpu_usage: float = 0.0  # Percentage
    gpu_memory_usage: float = 0.0  # MB
    config: WorkerConfig = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class WorkerManager:
    """
    Worker Manager
    Manages distributed workers for task execution
    """
    
    def __init__(self, 
                 celery_app_name: str = "red_team_tasks",
                 worker_base_dir: str = "/tmp/red_team_workers"):
        """Initialize worker manager"""
        self.celery_app_name = celery_app_name
        self.worker_base_dir = worker_base_dir
        
        # Worker tracking
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_processes: Dict[str, subprocess.Popen] = {}
        
        # Worker monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Create worker directory
        os.makedirs(worker_base_dir, exist_ok=True)
        
        logger.info("✅ Initialized Worker Manager")
    
    async def start_worker(self, config: WorkerConfig) -> bool:
        """Start a worker with given configuration"""
        try:
            worker_id = config.worker_id
            
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already exists")
                return False
            
            # Create worker info
            worker_info = WorkerInfo(
                worker_id=worker_id,
                worker_type=config.worker_type,
                status=WorkerStatus.STARTING,
                config=config,
                started_at=datetime.utcnow()
            )
            
            # Start worker process
            process = await self._start_worker_process(config)
            if process is None:
                worker_info.status = WorkerStatus.ERROR
                self.workers[worker_id] = worker_info
                return False
            
            # Update worker info
            worker_info.process_id = process.pid
            worker_info.status = WorkerStatus.RUNNING
            worker_info.last_heartbeat = datetime.utcnow()
            
            # Store worker info
            self.workers[worker_id] = worker_info
            self.worker_processes[worker_id] = process
            
            # Start monitoring if not already active
            if not self.monitoring_active:
                await self.start_monitoring()
            
            logger.info(f"Worker started: {worker_id} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start worker {config.worker_id}: {e}")
            return False
    
    async def stop_worker(self, worker_id: str, force: bool = False) -> bool:
        """Stop a worker"""
        try:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
            
            worker_info = self.workers[worker_id]
            worker_info.status = WorkerStatus.STOPPING
            
            if worker_id in self.worker_processes:
                process = self.worker_processes[worker_id]
                
                if force:
                    # Force kill
                    process.kill()
                else:
                    # Graceful shutdown
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        process.kill()
                
                del self.worker_processes[worker_id]
            
            worker_info.status = WorkerStatus.STOPPED
            worker_info.last_heartbeat = datetime.utcnow()
            
            logger.info(f"Worker stopped: {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop worker {worker_id}: {e}")
            return False
    
    async def restart_worker(self, worker_id: str) -> bool:
        """Restart a worker"""
        try:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
            
            worker_info = self.workers[worker_id]
            config = worker_info.config
            
            # Stop worker
            await self.stop_worker(worker_id, force=True)
            
            # Wait a bit
            await asyncio.sleep(2)
            
            # Start worker again
            return await self.start_worker(config)
            
        except Exception as e:
            logger.error(f"Failed to restart worker {worker_id}: {e}")
            return False
    
    async def scale_workers(self, 
                          worker_type: WorkerType, 
                          target_count: int,
                          config_template: WorkerConfig = None) -> int:
        """Scale workers of a specific type"""
        try:
            # Get current workers of this type
            current_workers = [
                w for w in self.workers.values() 
                if w.worker_type == worker_type and w.status in [WorkerStatus.RUNNING, WorkerStatus.IDLE, WorkerStatus.BUSY]
            ]
            
            current_count = len(current_workers)
            
            if target_count > current_count:
                # Scale up
                workers_to_add = target_count - current_count
                
                for i in range(workers_to_add):
                    if config_template:
                        # Use template config
                        worker_config = WorkerConfig(
                            worker_id=f"{worker_type.value}_{len(self.workers) + i}",
                            worker_type=worker_type,
                            concurrency=config_template.concurrency,
                            memory_limit=config_template.memory_limit,
                            cpu_limit=config_template.cpu_limit,
                            gpu_enabled=config_template.gpu_enabled,
                            gpu_memory_limit=config_template.gpu_memory_limit,
                            queue_name=config_template.queue_name,
                            log_level=config_template.log_level,
                            max_tasks_per_child=config_template.max_tasks_per_child
                        )
                    else:
                        # Use default config
                        worker_config = WorkerConfig(
                            worker_id=f"{worker_type.value}_{len(self.workers) + i}",
                            worker_type=worker_type
                        )
                    
                    await self.start_worker(worker_config)
                
                logger.info(f"Scaled up {worker_type.value} workers: {current_count} -> {target_count}")
                
            elif target_count < current_count:
                # Scale down
                workers_to_remove = current_count - target_count
                
                # Sort by tasks completed (remove least busy workers)
                current_workers.sort(key=lambda w: w.tasks_completed)
                
                for i in range(min(workers_to_remove, len(current_workers))):
                    worker_id = current_workers[i].worker_id
                    await self.stop_worker(worker_id)
                
                logger.info(f"Scaled down {worker_type.value} workers: {current_count} -> {target_count}")
            
            return target_count
            
        except Exception as e:
            logger.error(f"Failed to scale workers: {e}")
            return len([w for w in self.workers.values() if w.worker_type == worker_type])
    
    async def get_worker_status(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker status"""
        return self.workers.get(worker_id)
    
    async def get_all_workers(self) -> List[WorkerInfo]:
        """Get all workers"""
        return list(self.workers.values())
    
    async def get_workers_by_type(self, worker_type: WorkerType) -> List[WorkerInfo]:
        """Get workers by type"""
        return [w for w in self.workers.values() if w.worker_type == worker_type]
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        try:
            total_workers = len(self.workers)
            running_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING])
            idle_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.IDLE])
            busy_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.BUSY])
            error_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.ERROR])
            
            # Count by type
            type_counts = {}
            for worker_type in WorkerType:
                type_counts[worker_type.value] = len([
                    w for w in self.workers.values() 
                    if w.worker_type == worker_type and w.status in [WorkerStatus.RUNNING, WorkerStatus.IDLE, WorkerStatus.BUSY]
                ])
            
            # Calculate total tasks
            total_tasks_completed = sum(w.tasks_completed for w in self.workers.values())
            total_tasks_failed = sum(w.tasks_failed for w in self.workers.values())
            
            # Calculate resource usage
            total_memory_usage = sum(w.memory_usage for w in self.workers.values())
            total_cpu_usage = sum(w.cpu_usage for w in self.workers.values())
            total_gpu_usage = sum(w.gpu_usage for w in self.workers.values())
            total_gpu_memory_usage = sum(w.gpu_memory_usage for w in self.workers.values())
            
            return {
                "total_workers": total_workers,
                "running_workers": running_workers,
                "idle_workers": idle_workers,
                "busy_workers": busy_workers,
                "error_workers": error_workers,
                "type_counts": type_counts,
                "total_tasks_completed": total_tasks_completed,
                "total_tasks_failed": total_tasks_failed,
                "total_memory_usage_mb": total_memory_usage,
                "total_cpu_usage_percent": total_cpu_usage,
                "total_gpu_usage_percent": total_gpu_usage,
                "total_gpu_memory_usage_mb": total_gpu_memory_usage,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get worker stats: {e}")
            return {}
    
    async def start_monitoring(self):
        """Start worker monitoring"""
        try:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Worker monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop worker monitoring"""
        try:
            self.monitoring_active = False
            
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            logger.info("Worker monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    def _monitoring_loop(self):
        """Worker monitoring loop"""
        while self.monitoring_active:
            try:
                # Update worker statuses
                self._update_worker_statuses()
                
                # Clean up dead workers
                self._cleanup_dead_workers()
                
                # Sleep before next check
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(10)
    
    def _update_worker_statuses(self):
        """Update worker statuses"""
        try:
            for worker_id, worker_info in self.workers.items():
                if worker_id in self.worker_processes:
                    process = self.worker_processes[worker_id]
                    
                    # Check if process is still running
                    if process.poll() is None:
                        # Process is running, update resource usage
                        try:
                            ps_process = psutil.Process(process.pid)
                            worker_info.memory_usage = ps_process.memory_info().rss / 1024 / 1024  # MB
                            worker_info.cpu_usage = ps_process.cpu_percent()
                            
                            # Update GPU usage if available
                            if worker_info.config.gpu_enabled:
                                worker_info.gpu_usage = self._get_gpu_usage()
                                worker_info.gpu_memory_usage = self._get_gpu_memory_usage()
                            
                            worker_info.last_heartbeat = datetime.utcnow()
                            
                            # Update status based on activity
                            if worker_info.status == WorkerStatus.RUNNING:
                                worker_info.status = WorkerStatus.IDLE  # Assume idle if no recent activity
                            
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process died or access denied
                            worker_info.status = WorkerStatus.ERROR
                    else:
                        # Process died
                        worker_info.status = WorkerStatus.STOPPED
                        if worker_id in self.worker_processes:
                            del self.worker_processes[worker_id]
                
        except Exception as e:
            logger.error(f"Failed to update worker statuses: {e}")
    
    def _cleanup_dead_workers(self):
        """Clean up dead workers"""
        try:
            workers_to_remove = []
            
            for worker_id, worker_info in self.workers.items():
                if worker_info.status == WorkerStatus.STOPPED:
                    # Check if worker has been stopped for more than 1 hour
                    if (worker_info.last_heartbeat and 
                        datetime.utcnow() - worker_info.last_heartbeat > timedelta(hours=1)):
                        workers_to_remove.append(worker_id)
            
            for worker_id in workers_to_remove:
                del self.workers[worker_id]
                if worker_id in self.worker_processes:
                    del self.worker_processes[worker_id]
            
            if workers_to_remove:
                logger.info(f"Cleaned up {len(workers_to_remove)} dead workers")
                
        except Exception as e:
            logger.error(f"Failed to cleanup dead workers: {e}")
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage"""
        try:
            # This would need to be implemented based on your GPU monitoring solution
            # For now, return 0
            return 0.0
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in MB"""
        try:
            # This would need to be implemented based on your GPU monitoring solution
            # For now, return 0
            return 0.0
        except Exception:
            return 0.0
    
    async def _start_worker_process(self, config: WorkerConfig) -> Optional[subprocess.Popen]:
        """Start worker process"""
        try:
            # Build Celery worker command
            cmd = [
                "celery",
                "-A", self.celery_app_name,
                "worker",
                "--loglevel", config.log_level,
                "--concurrency", str(config.concurrency),
                "--hostname", config.worker_id,
                "--queues", config.queue_name,
                "--max-tasks-per-child", str(config.max_tasks_per_child),
                "--without-gossip",
                "--without-mingle",
                "--without-heartbeat"
            ]
            
            # Add memory limit if specified
            if config.memory_limit > 0:
                cmd.extend(["--max-memory-per-child", str(config.memory_limit * 1024 * 1024)])
            
            # Set environment variables
            env = os.environ.copy()
            env["CELERY_WORKER_ID"] = config.worker_id
            env["CELERY_WORKER_TYPE"] = config.worker_type.value
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.worker_base_dir
            )
            
            return process
            
        except Exception as e:
            logger.error(f"Failed to start worker process: {e}")
            return None
    
    async def export_worker_data(self, format: str = "json") -> str:
        """Export worker data"""
        try:
            if format.lower() == "json":
                data = {
                    "workers": [asdict(w) for w in self.workers.values()],
                    "stats": await self.get_worker_stats(),
                    "exported_at": datetime.utcnow().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Worker data export failed: {e}")
            return ""
