"""
Performance Monitoring Dashboard
Real-time performance monitoring and visualization for red team operations
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_id: str
    metric_name: str
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.alerts: List[PerformanceAlert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {
            "latency": {"warning": 1.0, "critical": 5.0},
            "throughput": {"warning": 0.1, "critical": 0.05},
            "memory_usage": {"warning": 100, "critical": 500},
            "cpu_usage": {"warning": 80, "critical": 95},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "success_rate": {"warning": 0.95, "critical": 0.9}
        }
        self.alert_counter = 0
        self.is_monitoring = False
        self.monitoring_task = None
    
    def add_metric(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """Add a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.metrics_history[name].append(metric)
        
        # Check for alerts
        self._check_alerts(name, value)
        
        logger.debug(f"Added metric {name}: {value}")
    
    def _check_alerts(self, metric_name: str, value: float):
        """Check if metric triggers any alerts"""
        if metric_name not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_name]
        
        for severity, threshold in thresholds.items():
            if self._should_trigger_alert(metric_name, value, threshold, severity):
                self._create_alert(metric_name, threshold, value, severity)
    
    def _should_trigger_alert(self, metric_name: str, value: float, threshold: float, severity: str) -> bool:
        """Check if alert should be triggered"""
        # Check if there's already an unresolved alert for this metric and severity
        existing_alert = any(
            alert.metric_name == metric_name and 
            alert.severity == severity and 
            not alert.resolved
            for alert in self.alerts
        )
        
        if existing_alert:
            return False
        
        # Check threshold based on metric type
        if metric_name in ["latency", "memory_usage", "cpu_usage", "error_rate"]:
            return value > threshold
        elif metric_name in ["throughput", "success_rate"]:
            return value < threshold
        else:
            return False
    
    def _create_alert(self, metric_name: str, threshold: float, value: float, severity: str):
        """Create a new alert"""
        self.alert_counter += 1
        alert_id = f"alert_{self.alert_counter}_{int(time.time())}"
        
        message = f"{metric_name} {severity} threshold exceeded: {value:.2f} > {threshold:.2f}"
        if metric_name in ["throughput", "success_rate"]:
            message = f"{metric_name} {severity} threshold exceeded: {value:.2f} < {threshold:.2f}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_name=metric_name,
            threshold=threshold,
            current_value=value,
            severity=severity,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        logger.warning(f"Performance alert: {message}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Resolved alert {alert_id}")
                break
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get metric history for the last N hours"""
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        history = list(self.metrics_history[metric_name])
        
        return [metric for metric in history if metric.timestamp >= cutoff_time]
    
    def get_metric_summary(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get metric summary for the last N hours"""
        history = self.get_metric_history(metric_name, hours)
        
        if not history:
            return {"error": f"No data for metric {metric_name}"}
        
        values = [metric.value for metric in history]
        
        summary = {
            "metric_name": metric_name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1],
            "latest_timestamp": history[-1].timestamp.isoformat(),
            "trend": self._calculate_trend(values)
        }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active_alerts = self.get_active_alerts()
        
        summary = {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len(self.alerts) - len(active_alerts),
            "alerts_by_severity": defaultdict(int),
            "alerts_by_metric": defaultdict(int)
        }
        
        for alert in active_alerts:
            summary["alerts_by_severity"][alert.severity] += 1
            summary["alerts_by_metric"][alert.metric_name] += 1
        
        return dict(summary)
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get complete dashboard data"""
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "alerts": self.get_alert_summary(),
            "system_health": self._calculate_system_health()
        }
        
        # Get summaries for all metrics
        for metric_name in self.metrics_history.keys():
            dashboard_data["metrics"][metric_name] = self.get_metric_summary(metric_name, hours)
        
        return dashboard_data
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        health_score = 100
        issues = []
        
        # Check active alerts
        active_alerts = self.get_active_alerts()
        critical_alerts = [alert for alert in active_alerts if alert.severity == "critical"]
        warning_alerts = [alert for alert in active_alerts if alert.severity == "warning"]
        
        if critical_alerts:
            health_score -= len(critical_alerts) * 20
            issues.append(f"{len(critical_alerts)} critical alerts")
        
        if warning_alerts:
            health_score -= len(warning_alerts) * 5
            issues.append(f"{len(warning_alerts)} warning alerts")
        
        # Check recent error rates
        error_history = self.get_metric_history("error_rate", 1)  # Last hour
        if error_history:
            recent_error_rate = error_history[-1].value
            if recent_error_rate > 0.1:
                health_score -= 30
                issues.append("High error rate")
            elif recent_error_rate > 0.05:
                health_score -= 15
                issues.append("Elevated error rate")
        
        # Check recent latency
        latency_history = self.get_metric_history("latency", 1)  # Last hour
        if latency_history:
            recent_latency = latency_history[-1].value
            if recent_latency > 5.0:
                health_score -= 25
                issues.append("High latency")
            elif recent_latency > 1.0:
                health_score -= 10
                issues.append("Elevated latency")
        
        health_score = max(0, health_score)
        
        return {
            "score": health_score,
            "status": self._get_health_status(health_score),
            "issues": issues
        }
    
    def _get_health_status(self, score: float) -> str:
        """Get health status based on score"""
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        elif score >= 30:
            return "poor"
        else:
            return "critical"
    
    def set_threshold(self, metric_name: str, severity: str, threshold: float):
        """Set threshold for a metric"""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        self.thresholds[metric_name][severity] = threshold
        logger.info(f"Set {severity} threshold for {metric_name}: {threshold}")
    
    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get all thresholds"""
        return self.thresholds.copy()
    
    def clear_metrics(self, metric_name: str = None):
        """Clear metrics history"""
        if metric_name:
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].clear()
                logger.info(f"Cleared metrics for {metric_name}")
        else:
            self.metrics_history.clear()
            logger.info("Cleared all metrics")
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        self.alert_counter = 0
        logger.info("Cleared all alerts")
    
    def export_data(self, filename: str = None) -> str:
        """Export dashboard data to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_dashboard_{timestamp}.json"
        
        data = {
            "dashboard_data": self.get_dashboard_data(),
            "thresholds": self.get_thresholds(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported dashboard data to {filename}")
        return filename
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        logger.info(f"Starting performance monitoring with {interval_seconds}s interval")
        
        async def monitor_loop():
            while self.is_monitoring:
                try:
                    # Simulate collecting system metrics
                    import psutil
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.add_metric("cpu_usage", cpu_percent)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    memory_mb = memory.used / 1024 / 1024
                    self.add_metric("memory_usage", memory_mb)
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    self.add_metric("disk_usage", disk_percent)
                    
                    logger.debug(f"Collected metrics: CPU={cpu_percent}%, Memory={memory_mb:.1f}MB, Disk={disk_percent:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        self.monitoring_task = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped performance monitoring")
    
    def get_metric_recommendations(self, metric_name: str) -> List[str]:
        """Get recommendations for improving a metric"""
        recommendations = []
        
        if metric_name == "latency":
            recommendations = [
                "Consider implementing caching for frequently accessed data",
                "Optimize database queries and add indexes",
                "Use connection pooling for database connections",
                "Implement request batching for bulk operations",
                "Consider using CDN for static content"
            ]
        elif metric_name == "throughput":
            recommendations = [
                "Increase the number of worker processes",
                "Implement horizontal scaling",
                "Optimize critical code paths",
                "Use asynchronous processing where possible",
                "Consider using faster data structures"
            ]
        elif metric_name == "memory_usage":
            recommendations = [
                "Implement memory pooling for frequently allocated objects",
                "Use generators instead of lists for large datasets",
                "Implement proper garbage collection strategies",
                "Monitor for memory leaks",
                "Consider using more memory-efficient data structures"
            ]
        elif metric_name == "cpu_usage":
            recommendations = [
                "Optimize CPU-intensive algorithms",
                "Implement caching to reduce computation",
                "Use multiprocessing for CPU-bound tasks",
                "Profile code to identify bottlenecks",
                "Consider using compiled extensions for critical paths"
            ]
        elif metric_name == "error_rate":
            recommendations = [
                "Implement comprehensive error handling",
                "Add input validation and sanitization",
                "Improve logging and monitoring",
                "Implement circuit breakers for external services",
                "Add retry mechanisms with exponential backoff"
            ]
        elif metric_name == "success_rate":
            recommendations = [
                "Improve error handling and recovery",
                "Add input validation",
                "Implement fallback mechanisms",
                "Monitor and alert on failures",
                "Regular testing and validation"
            ]
        
        return recommendations
