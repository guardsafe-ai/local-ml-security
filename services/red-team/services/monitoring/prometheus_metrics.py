"""
Prometheus Metrics Collection
Collects and exposes metrics for monitoring red team testing performance and security events.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    from prometheus_client.core import REGISTRY
except ImportError:
    # Fallback for environments without prometheus_client
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class CollectorRegistry:
        def __init__(self): pass
    def generate_latest(registry): return b""
    REGISTRY = CollectorRegistry()

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"

@dataclass
class MetricConfig:
    """Configuration for a metric"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = None
    buckets: List[float] = None

class PrometheusMetrics:
    """Prometheus metrics collector for red team testing"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.metrics: Dict[str, Any] = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup all metrics"""
        try:
            # Attack execution metrics
            self.metrics['attacks_total'] = Counter(
                'redteam_attacks_total',
                'Total number of attacks executed',
                ['attack_type', 'model_type', 'status'],
                registry=self.registry
            )
            
            self.metrics['attack_duration'] = Histogram(
                'redteam_attack_duration_seconds',
                'Duration of attack execution',
                ['attack_type', 'model_type'],
                buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
                registry=self.registry
            )
            
            self.metrics['attack_success_rate'] = Gauge(
                'redteam_attack_success_rate',
                'Success rate of attacks',
                ['attack_type', 'model_type'],
                registry=self.registry
            )
            
            # Model performance metrics
            self.metrics['model_inference_time'] = Histogram(
                'redteam_model_inference_seconds',
                'Model inference time',
                ['model_type', 'model_version'],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
            
            self.metrics['model_memory_usage'] = Gauge(
                'redteam_model_memory_bytes',
                'Model memory usage in bytes',
                ['model_type', 'model_version'],
                registry=self.registry
            )
            
            self.metrics['model_gpu_utilization'] = Gauge(
                'redteam_model_gpu_utilization_percent',
                'GPU utilization percentage',
                ['gpu_id', 'model_type'],
                registry=self.registry
            )
            
            # Security event metrics
            self.metrics['vulnerabilities_detected'] = Counter(
                'redteam_vulnerabilities_detected_total',
                'Total vulnerabilities detected',
                ['severity', 'attack_type', 'model_type'],
                registry=self.registry
            )
            
            self.metrics['compliance_violations'] = Counter(
                'redteam_compliance_violations_total',
                'Total compliance violations detected',
                ['framework', 'control', 'severity'],
                registry=self.registry
            )
            
            self.metrics['privacy_breaches'] = Counter(
                'redteam_privacy_breaches_total',
                'Total privacy breaches detected',
                ['breach_type', 'data_type'],
                registry=self.registry
            )
            
            # System performance metrics
            self.metrics['active_workers'] = Gauge(
                'redteam_active_workers',
                'Number of active worker processes',
                registry=self.registry
            )
            
            self.metrics['queue_size'] = Gauge(
                'redteam_queue_size',
                'Number of tasks in queue',
                ['queue_type'],
                registry=self.registry
            )
            
            self.metrics['cache_hit_rate'] = Gauge(
                'redteam_cache_hit_rate',
                'Cache hit rate percentage',
                ['cache_type'],
                registry=self.registry
            )
            
            # Resource utilization metrics
            self.metrics['cpu_usage'] = Gauge(
                'redteam_cpu_usage_percent',
                'CPU usage percentage',
                ['instance'],
                registry=self.registry
            )
            
            self.metrics['memory_usage'] = Gauge(
                'redteam_memory_usage_bytes',
                'Memory usage in bytes',
                ['instance'],
                registry=self.registry
            )
            
            self.metrics['disk_usage'] = Gauge(
                'redteam_disk_usage_bytes',
                'Disk usage in bytes',
                ['instance', 'mount_point'],
                registry=self.registry
            )
            
            # Network metrics
            self.metrics['network_requests'] = Counter(
                'redteam_network_requests_total',
                'Total network requests',
                ['method', 'endpoint', 'status_code'],
                registry=self.registry
            )
            
            self.metrics['network_duration'] = Histogram(
                'redteam_network_duration_seconds',
                'Network request duration',
                ['method', 'endpoint'],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=self.registry
            )
            
            # Custom metrics
            self.metrics['custom_metrics'] = {}
            
            logger.info("Prometheus metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
    
    def record_attack_execution(self, attack_type: str, model_type: str, 
                              duration: float, success: bool):
        """Record attack execution metrics"""
        try:
            status = "success" if success else "failure"
            self.metrics['attacks_total'].labels(
                attack_type=attack_type,
                model_type=model_type,
                status=status
            ).inc()
            
            self.metrics['attack_duration'].labels(
                attack_type=attack_type,
                model_type=model_type
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to record attack execution: {e}")
    
    def update_attack_success_rate(self, attack_type: str, model_type: str, 
                                 success_rate: float):
        """Update attack success rate"""
        try:
            self.metrics['attack_success_rate'].labels(
                attack_type=attack_type,
                model_type=model_type
            ).set(success_rate)
            
        except Exception as e:
            logger.error(f"Failed to update attack success rate: {e}")
    
    def record_model_inference(self, model_type: str, model_version: str, 
                             duration: float):
        """Record model inference time"""
        try:
            self.metrics['model_inference_time'].labels(
                model_type=model_type,
                model_version=model_version
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to record model inference: {e}")
    
    def update_model_memory_usage(self, model_type: str, model_version: str, 
                                memory_bytes: int):
        """Update model memory usage"""
        try:
            self.metrics['model_memory_usage'].labels(
                model_type=model_type,
                model_version=model_version
            ).set(memory_bytes)
            
        except Exception as e:
            logger.error(f"Failed to update model memory usage: {e}")
    
    def update_gpu_utilization(self, gpu_id: str, model_type: str, 
                             utilization_percent: float):
        """Update GPU utilization"""
        try:
            self.metrics['model_gpu_utilization'].labels(
                gpu_id=gpu_id,
                model_type=model_type
            ).set(utilization_percent)
            
        except Exception as e:
            logger.error(f"Failed to update GPU utilization: {e}")
    
    def record_vulnerability(self, severity: str, attack_type: str, 
                           model_type: str):
        """Record vulnerability detection"""
        try:
            self.metrics['vulnerabilities_detected'].labels(
                severity=severity,
                attack_type=attack_type,
                model_type=model_type
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record vulnerability: {e}")
    
    def record_compliance_violation(self, framework: str, control: str, 
                                  severity: str):
        """Record compliance violation"""
        try:
            self.metrics['compliance_violations'].labels(
                framework=framework,
                control=control,
                severity=severity
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record compliance violation: {e}")
    
    def record_privacy_breach(self, breach_type: str, data_type: str):
        """Record privacy breach"""
        try:
            self.metrics['privacy_breaches'].labels(
                breach_type=breach_type,
                data_type=data_type
            ).inc()
            
        except Exception as e:
            logger.error(f"Failed to record privacy breach: {e}")
    
    def update_worker_count(self, count: int):
        """Update active worker count"""
        try:
            self.metrics['active_workers'].set(count)
        except Exception as e:
            logger.error(f"Failed to update worker count: {e}")
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size"""
        try:
            self.metrics['queue_size'].labels(queue_type=queue_type).set(size)
        except Exception as e:
            logger.error(f"Failed to update queue size: {e}")
    
    def update_cache_hit_rate(self, cache_type: str, hit_rate: float):
        """Update cache hit rate"""
        try:
            self.metrics['cache_hit_rate'].labels(cache_type=cache_type).set(hit_rate)
        except Exception as e:
            logger.error(f"Failed to update cache hit rate: {e}")
    
    def update_resource_usage(self, cpu_percent: float, memory_bytes: int, 
                            disk_usage: Dict[str, int]):
        """Update resource usage metrics"""
        try:
            self.metrics['cpu_usage'].labels(instance="main").set(cpu_percent)
            self.metrics['memory_usage'].labels(instance="main").set(memory_bytes)
            
            for mount_point, usage in disk_usage.items():
                self.metrics['disk_usage'].labels(
                    instance="main",
                    mount_point=mount_point
                ).set(usage)
                
        except Exception as e:
            logger.error(f"Failed to update resource usage: {e}")
    
    def record_network_request(self, method: str, endpoint: str, 
                             status_code: int, duration: float):
        """Record network request"""
        try:
            self.metrics['network_requests'].labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.metrics['network_duration'].labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to record network request: {e}")
    
    def create_custom_metric(self, name: str, description: str, 
                           metric_type: MetricType, labels: List[str] = None):
        """Create a custom metric"""
        try:
            if metric_type == MetricType.COUNTER:
                metric = Counter(name, description, labels or [], registry=self.registry)
            elif metric_type == MetricType.HISTOGRAM:
                metric = Histogram(name, description, labels or [], registry=self.registry)
            elif metric_type == MetricType.GAUGE:
                metric = Gauge(name, description, labels or [], registry=self.registry)
            elif metric_type == MetricType.SUMMARY:
                metric = Summary(name, description, labels or [], registry=self.registry)
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")
            
            self.metrics['custom_metrics'][name] = metric
            return metric
            
        except Exception as e:
            logger.error(f"Failed to create custom metric: {e}")
            return None
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return ""
    
    def get_metric_value(self, metric_name: str, labels: Dict[str, str] = None) -> float:
        """Get value of a specific metric"""
        try:
            if metric_name not in self.metrics:
                return 0.0
            
            metric = self.metrics[metric_name]
            if labels:
                metric = metric.labels(**labels)
            
            # For counters, histograms, and summaries, we need to get samples
            if hasattr(metric, '_value'):
                return float(metric._value)
            elif hasattr(metric, '_sum'):
                return float(metric._sum)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to get metric value: {e}")
            return 0.0
    
    async def start_metrics_collection(self, interval: int = 30):
        """Start periodic metrics collection"""
        try:
            while True:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_bytes = memory.used
            
            # Disk usage
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = usage.used
                except PermissionError:
                    continue
            
            self.update_resource_usage(cpu_percent, memory_bytes, disk_usage)
            
        except ImportError:
            logger.warning("psutil not available for system metrics collection")
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def reset_metrics(self):
        """Reset all metrics"""
        try:
            for metric in self.metrics.values():
                if hasattr(metric, 'clear'):
                    metric.clear()
                elif hasattr(metric, '_value'):
                    metric._value = 0
                    
            logger.info("Metrics reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset metrics: {e}")
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format"""
        try:
            if format_type == "prometheus":
                return self.get_metrics()
            elif format_type == "json":
                return self._export_json()
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return ""
    
    def _export_json(self) -> str:
        """Export metrics as JSON"""
        try:
            import json
            
            metrics_data = {}
            for name, metric in self.metrics.items():
                if hasattr(metric, '_value'):
                    metrics_data[name] = float(metric._value)
                elif hasattr(metric, '_sum'):
                    metrics_data[name] = float(metric._sum)
                else:
                    metrics_data[name] = 0.0
            
            return json.dumps(metrics_data, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to export JSON metrics: {e}")
            return "{}"
