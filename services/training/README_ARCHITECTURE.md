# Training Service - Architecture Documentation

## Overview

The Training Service is a comprehensive, production-ready microservice designed for machine learning model training, data management, and model lifecycle management. It provides a robust, scalable platform for training security-focused ML models with advanced features like data augmentation, efficient data management, and automated retraining capabilities.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Service                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   FastAPI   │  │  Prometheus │  │   Jaeger    │            │
│  │   Server    │  │  Metrics    │  │  Tracing    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Health    │  │   Models    │  │  Training   │            │
│  │   Routes    │  │   Routes    │  │   Routes    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    Data     │  │  Efficient  │  │    Data     │            │
│  │   Routes    │  │    Data     │  │Augmentation │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Training   │  │   Model     │  │   MLflow    │            │
│  │   Queue     │  │  Trainer    │  │  Service    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Efficient  │  │   Shared    │  │  Training   │            │
│  │    Data     │  │    Data     │  │   Config    │            │
│  │  Manager    │  │  Manager    │  │  Service    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. FastAPI Application (`main.py`)
- **Purpose**: Main application entry point and service orchestration
- **Key Features**:
  - CORS middleware for cross-origin requests
  - Prometheus metrics collection
  - Distributed tracing with Jaeger
  - External service integration (Business Metrics, Data Privacy)
  - Startup/shutdown event handling
  - Database initialization
  - MLflow integration

#### 2. Route Modules
- **Health Routes** (`routes/health.py`): Service health monitoring
- **Model Routes** (`routes/models.py`): Model management and registry operations
- **Training Routes** (`routes/training.py`): Core training operations and job management
- **Data Routes** (`routes/data.py`): Basic data upload and management
- **Efficient Data Routes** (`routes/efficient_data.py`): Advanced data management with chunked uploads
- **Data Augmentation Routes** (`routes/data_augmentation.py`): Data augmentation and synthetic data generation
- **Training Queue Routes** (`routes/training_queue.py`): Training job queue management

#### 3. Service Layer
- **ModelTrainer**: Core training logic and model management
- **MLflowService**: MLflow integration for experiment tracking and model registry
- **TrainingQueue**: Asynchronous job queue for training operations
- **TrainingConfigService**: Configuration management for training parameters
- **EfficientDataManager**: Advanced data management with MinIO integration
- **SharedDataManager**: Shared data access across services

#### 4. Data Models
- **Request Models** (`models/requests.py`): Pydantic models for incoming API requests
- **Response Models** (`models/responses.py`): Pydantic models for API responses

## Data Flow

### 1. Training Request Flow

```
Client Request → FastAPI Router → Training Queue → Model Trainer → MLflow → MinIO
     ↓              ↓                ↓              ↓            ↓        ↓
   Validation   Authentication   Job Creation   Model Training  Logging  Storage
     ↓              ↓                ↓              ↓            ↓        ↓
   Response ←   Status Update ←  Queue Status ←  Training ←  Metrics ←  Artifacts
```

### 2. Data Upload Flow

```
File Upload → Input Validation → Chunked Upload → MinIO Storage → Processing
     ↓              ↓                ↓              ↓              ↓
   Sanitization  Size Check      Background      S3 Storage    PII Detection
     ↓              ↓                ↓              ↓              ↓
   Response ←   Status Update ←  Progress ←    File Info ←   Classification
```

### 3. Model Management Flow

```
Model Request → MLflow Service → Model Registry → Model Loading → Cache
     ↓              ↓                ↓              ↓            ↓
   Validation   Version Check    Metadata      Model Load    Memory
     ↓              ↓                ↓              ↓            ↓
   Response ←   Model Info ←    Registry ←    Status ←     Cache Status
```

## Key Features

### 1. Advanced Data Management
- **Chunked Uploads**: Support for large file uploads with progress tracking
- **Data Validation**: Comprehensive validation rules and error handling
- **PII Detection**: Automatic detection of personally identifiable information
- **Data Lifecycle**: Automated cleanup and data retention policies
- **S3 Integration**: Seamless integration with MinIO for object storage

### 2. Training Queue System
- **Priority-based Queue**: Support for different job priorities (LOW, NORMAL, HIGH, URGENT)
- **Resource Management**: CPU and memory limits for training jobs
- **Retry Logic**: Automatic retry for failed jobs with exponential backoff
- **Progress Tracking**: Real-time progress updates and status monitoring
- **Timeout Handling**: Configurable timeouts for long-running jobs

### 3. Data Augmentation
- **Multiple Techniques**: Synonym replacement, back-translation, random operations
- **Configurable Parameters**: Adjustable probabilities and augmentation factors
- **Synthetic Data Generation**: AI-powered synthetic data creation
- **Dataset Balancing**: Automatic oversampling for imbalanced datasets
- **Preview Functionality**: Preview augmentation results before applying

### 4. Model Lifecycle Management
- **Version Control**: Complete model versioning with MLflow
- **Model Registry**: Centralized model storage and metadata
- **Model Promotion**: Staging to production promotion workflow
- **Model Rollback**: Ability to rollback to previous versions
- **Performance Tracking**: Historical performance monitoring

### 5. Monitoring and Observability
- **Prometheus Metrics**: Comprehensive metrics collection
- **Distributed Tracing**: Request flow tracking with Jaeger
- **Health Checks**: Multi-level health monitoring
- **Audit Logging**: Complete audit trail for all operations
- **Performance Monitoring**: Real-time performance metrics

## External Integrations

### 1. Business Metrics Service
- **Purpose**: Track training-related business metrics
- **Integration**: HTTP API calls for metric recording
- **Metrics**: Training job counts, success rates, resource usage

### 2. Data Privacy Service
- **Purpose**: PII detection and data classification
- **Integration**: HTTP API calls for data analysis
- **Features**: Automatic PII detection, sensitivity scoring

### 3. MLflow
- **Purpose**: Experiment tracking and model registry
- **Integration**: Direct MLflow client integration
- **Features**: Experiment logging, model versioning, artifact storage

### 4. MinIO
- **Purpose**: Object storage for training data and artifacts
- **Integration**: S3-compatible API
- **Features**: Chunked uploads, data lifecycle management

### 5. PostgreSQL
- **Purpose**: Metadata storage and job tracking
- **Integration**: Async database connection
- **Features**: Job status, configuration storage, audit logs

## Security Features

### 1. Input Validation
- **Sanitization**: All user inputs are sanitized and validated
- **Size Limits**: Configurable file size and request size limits
- **Type Validation**: Strict type checking with Pydantic models

### 2. Authentication & Authorization
- **API Key Management**: Secure API key handling
- **Role-based Access**: Different access levels for different operations
- **Audit Logging**: Complete audit trail for security monitoring

### 3. Data Protection
- **PII Detection**: Automatic detection of sensitive data
- **Data Encryption**: Encryption at rest and in transit
- **Access Control**: Fine-grained access control for data access

## Performance Optimizations

### 1. Asynchronous Processing
- **Non-blocking I/O**: All database and external service calls are asynchronous
- **Background Tasks**: Long-running operations run in background
- **Concurrent Processing**: Multiple training jobs can run concurrently

### 2. Caching
- **Model Caching**: Frequently used models are cached in memory
- **Configuration Caching**: Training configurations are cached
- **Data Caching**: Frequently accessed data is cached

### 3. Resource Management
- **Memory Limits**: Configurable memory limits for training jobs
- **CPU Limits**: CPU resource allocation and monitoring
- **Queue Management**: Intelligent job queuing and resource allocation

## Deployment Architecture

### 1. Containerization
- **Docker**: Service runs in a Docker container
- **Multi-stage Build**: Optimized Docker image with minimal dependencies
- **Health Checks**: Container health monitoring

### 2. Service Discovery
- **Docker Compose**: Service orchestration and networking
- **Environment Variables**: Configuration through environment variables
- **Service Dependencies**: Proper service startup ordering

### 3. Monitoring
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting
- **Jaeger**: Distributed tracing and debugging

## Configuration

### 1. Environment Variables
```bash
# Core Configuration
MLFLOW_TRACKING_URI=http://mlflow:5000
POSTGRES_URL=postgresql://mlflow:password@postgres:5432/ml_security_consolidated
MINIO_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# External Services
BUSINESS_METRICS_URL=http://business-metrics:8004
DATA_PRIVACY_URL=http://data-privacy:8005

# Resource Limits
NVIDIA_VISIBLE_DEVICES=all
```

### 2. Service Configuration
- **Training Parameters**: Configurable training hyperparameters
- **Queue Settings**: Queue size, worker count, timeout settings
- **Data Management**: File size limits, retention policies
- **Monitoring**: Metrics collection intervals, log levels

## Error Handling

### 1. Exception Handling
- **Graceful Degradation**: Service continues to function even if some components fail
- **Error Recovery**: Automatic retry mechanisms for transient failures
- **User-friendly Messages**: Clear error messages for API consumers

### 2. Logging
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Log Levels**: Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- **Correlation IDs**: Request tracing across service boundaries

### 3. Monitoring
- **Health Checks**: Multiple levels of health monitoring
- **Metrics**: Comprehensive metrics for monitoring and alerting
- **Alerting**: Automated alerting for critical failures

## Scalability

### 1. Horizontal Scaling
- **Stateless Design**: Service can be scaled horizontally
- **Load Balancing**: Support for load balancing across instances
- **Queue Distribution**: Training jobs can be distributed across workers

### 2. Vertical Scaling
- **Resource Allocation**: Configurable CPU and memory limits
- **Performance Tuning**: Optimized for different hardware configurations
- **Resource Monitoring**: Real-time resource usage monitoring

## Future Enhancements

### 1. Planned Features
- **Federated Learning**: Support for federated learning scenarios
- **AutoML**: Automated hyperparameter tuning
- **Model Compression**: Model optimization and compression
- **Edge Deployment**: Support for edge device deployment

### 2. Performance Improvements
- **GPU Acceleration**: Enhanced GPU utilization
- **Distributed Training**: Multi-GPU and multi-node training
- **Model Optimization**: Advanced model optimization techniques

## Conclusion

The Training Service provides a comprehensive, production-ready platform for ML model training with advanced features like data management, augmentation, and automated retraining. Its modular architecture, robust error handling, and extensive monitoring capabilities make it suitable for enterprise environments requiring reliable and scalable ML operations.

The service integrates seamlessly with the broader ML Security platform, providing the foundation for secure, monitored, and efficient machine learning operations.