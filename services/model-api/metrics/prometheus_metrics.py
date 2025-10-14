"""
Comprehensive Prometheus Metrics for Model API Service
Provides complete metrics coverage for monitoring and alerting
"""

from prometheus_client import Counter, Histogram, Gauge, Info, Summary
from typing import Dict, Any

# HTTP Metrics
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

HTTP_REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

HTTP_ERRORS_TOTAL = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)

# Database Metrics
DB_CONNECTIONS = Gauge(
    'database_connections',
    'Current database connections',
    ['state']  # active, idle, total
)

DB_QUERY_DURATION = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation'],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

DB_QUERIES_TOTAL = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'status']
)

# Model Metrics
MODEL_PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_name', 'version', 'status']
)

MODEL_PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Model prediction latency',
    ['model_name', 'version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

MODEL_CONFIDENCE = Histogram(
    'model_confidence',
    'Model prediction confidence distribution',
    ['model_name', 'prediction_class'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

MODEL_LOAD_TIME = Histogram(
    'model_load_duration_seconds',
    'Model loading time',
    ['model_name', 'source'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

MODEL_MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Model memory usage',
    ['model_name', 'version']
)

ACTIVE_MODELS = Gauge(
    'active_models_total',
    'Number of active models',
    ['type']  # pretrained, trained
)

# Cache Metrics
CACHE_OPERATIONS_TOTAL = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result']  # get, set, delete, hit, miss
)

CACHE_SIZE = Gauge(
    'cache_size_bytes',
    'Cache size in bytes'
)

CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio'
)

# Circuit Breaker Metrics
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state',
    ['service', 'breaker_name'],
    states={0: 'closed', 1: 'open', 2: 'half_open'}
)

CIRCUIT_BREAKER_FAILURES = Counter(
    'circuit_breaker_failures_total',
    'Circuit breaker failures',
    ['service', 'breaker_name']
)

# Business Metrics
SECURITY_THREATS_DETECTED = Counter(
    'security_threats_detected_total',
    'Security threats detected',
    ['threat_type', 'model_name', 'severity']
)

PREDICTION_ACCURACY = Gauge(
    'prediction_accuracy',
    'Prediction accuracy',
    ['model_name', 'version']
)

FALSE_POSITIVE_RATE = Gauge(
    'false_positive_rate',
    'False positive rate',
    ['model_name', 'version']
)

FALSE_NEGATIVE_RATE = Gauge(
    'false_negative_rate',
    'False negative rate',
    ['model_name', 'version']
)

# System Metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_bytes',
    'System disk usage in bytes',
    ['device']
)

# MLflow Metrics
MLFLOW_OPERATIONS_TOTAL = Counter(
    'mlflow_operations_total',
    'Total MLflow operations',
    ['operation', 'status']
)

MLFLOW_OPERATION_DURATION = Histogram(
    'mlflow_operation_duration_seconds',
    'MLflow operation duration',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Training Metrics
TRAINING_JOBS_TOTAL = Counter(
    'training_jobs_total',
    'Total training jobs',
    ['model_name', 'status']
)

TRAINING_JOB_DURATION = Histogram(
    'training_job_duration_seconds',
    'Training job duration',
    ['model_name'],
    buckets=[60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0]
)

# Drift Detection Metrics
DRIFT_DETECTIONS_TOTAL = Counter(
    'drift_detections_total',
    'Total drift detections',
    ['model_name', 'drift_type', 'detected']
)

DRIFT_SCORE = Gauge(
    'drift_score',
    'Current drift score',
    ['model_name', 'drift_type']
)

# Ensemble Metrics
ENSEMBLE_PREDICTIONS_TOTAL = Counter(
    'ensemble_predictions_total',
    'Total ensemble predictions',
    ['model_count', 'status']
)

ENSEMBLE_ACCURACY = Gauge(
    'ensemble_accuracy',
    'Ensemble prediction accuracy',
    ['model_combination']
)

# Service Info
SERVICE_INFO = Info(
    'service_info',
    'Service information'
)

# Performance Budget Metrics
PERFORMANCE_BUDGET_VIOLATIONS = Counter(
    'performance_budget_violations_total',
    'Performance budget violations',
    ['endpoint', 'threshold']  # p95, p99
)

# Audit Metrics
AUDIT_EVENTS_TOTAL = Counter(
    'audit_events_total',
    'Total audit events',
    ['event_type', 'severity']
)

# Health Check Metrics
HEALTH_CHECK_DURATION = Histogram(
    'health_check_duration_seconds',
    'Health check duration',
    ['check_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

HEALTH_CHECK_STATUS = Gauge(
    'health_check_status',
    'Health check status',
    ['check_name'],
    states={0: 'unhealthy', 1: 'healthy', 2: 'degraded'}
)

def update_service_info(version: str, build_date: str, git_commit: str):
    """Update service information"""
    SERVICE_INFO.info({
        'version': version,
        'build_date': build_date,
        'git_commit': git_commit,
        'service': 'model-api'
    })

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of all metrics"""
    return {
        'http_requests': HTTP_REQUESTS_TOTAL._value.sum(),
        'active_models': ACTIVE_MODELS._value.sum(),
        'cache_hit_ratio': CACHE_HIT_RATIO._value,
        'security_threats': SECURITY_THREATS_DETECTED._value.sum(),
        'drift_detections': DRIFT_DETECTIONS_TOTAL._value.sum()
    }
