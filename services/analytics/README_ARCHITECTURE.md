# Analytics Service - Architecture Documentation

## Overview

The Analytics Service is a comprehensive, production-ready microservice designed for machine learning analytics, drift detection, and automated model retraining. It provides advanced analytics capabilities for monitoring model performance, detecting data and model drift, and automatically triggering retraining when needed.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Analytics Service                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   FastAPI   │  │  Prometheus │  │   Jaeger    │            │
│  │   Server    │  │  Metrics    │  │  Tracing    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Health    │  │   Red Team  │  │   Model     │            │
│  │   Routes    │  │   Routes    │  │Performance  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Analytics  │  │   Drift     │  │  Auto       │            │
│  │   Routes    │  │ Detection   │  │ Retrain     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Analytics  │  │   Drift     │  │   Auto      │            │
│  │  Service    │  │  Detector   │  │  Retrain    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Model     │  │   Email     │  │   Database  │            │
│  │  Promotion  │  │ Notifications│  │  Manager    │            │
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
  - Database schema initialization
  - Auto-retrain monitoring startup
  - HTTP client management

#### 2. Route Modules
- **Health Routes** (`routes/health.py`): Service health monitoring
- **Red Team Routes** (`routes/red_team.py`): Red team test result storage and analysis
- **Model Performance Routes** (`routes/model_performance.py`): Model performance tracking
- **Analytics Routes** (`routes/analytics.py`): General analytics and trend analysis
- **Drift Detection Routes** (`routes/drift_detection.py`): Data and model drift detection
- **Auto Retrain Routes** (`routes/auto_retrain.py`): Automated retraining management

#### 3. Service Layer
- **AnalyticsService**: Core analytics and trend analysis
- **DriftDetector**: Data and model drift detection algorithms
- **AutoRetrainService**: Automated retraining orchestration
- **ModelPromotionService**: Model evaluation and promotion logic
- **EmailService**: Email notifications for alerts and events
- **DatabaseManager**: Database connection and query management

#### 4. Data Models
- **Request Models** (`models/requests.py`): Pydantic models for incoming API requests
- **Response Models** (`models/responses.py`): Pydantic models for API responses

## Data Flow

### 1. Drift Detection Flow

```
Data Input → Drift Detector → Statistical Tests → Drift Analysis → Alert Generation
     ↓              ↓              ↓              ↓              ↓
  Validation    KS/Chi2/PSI    Threshold      Notification    Auto-Retrain
     ↓              ↓              ↓              ↓              ↓
  Response ←   Drift Score ←   Drift Status ←   Email Alert ←  Trigger Check
```

### 2. Auto-Retrain Flow

```
Drift Detection → Threshold Check → Retrain Decision → Training Job → Model Promotion
     ↓              ↓              ↓              ↓              ↓
  Monitoring    Cooldown Check   Priority Queue   Training      Evaluation
     ↓              ↓              ↓              ↓              ↓
  Response ←   Status Update ←  Job Status ←   Training ←    Promotion
```

### 3. Model Promotion Flow

```
Model Evaluation → Performance Check → Statistical Test → Promotion Decision → MLflow Update
     ↓              ↓              ↓              ↓              ↓
  Test Data    Accuracy/F1      Significance    Criteria Met    Stage Change
     ↓              ↓              ↓              ↓              ↓
  Response ←   Evaluation ←    Test Results ←   Decision ←    Notification
```

## Key Features

### 1. Drift Detection
- **Data Drift**: Statistical tests (KS, Chi2, PSI) for feature drift detection
- **Model Drift**: Performance drift detection using prediction analysis
- **Sliding Window**: Production data analysis with configurable time windows
- **Threshold Management**: Configurable drift thresholds and severity levels
- **Alert System**: Email notifications for drift alerts

### 2. Auto-Retrain System
- **Automatic Monitoring**: Continuous monitoring of drift conditions
- **Intelligent Triggers**: Multiple trigger conditions (data drift, model drift, performance drop)
- **Cooldown Management**: Prevents excessive retraining with cooldown periods
- **Priority Queue**: Priority-based retraining job management
- **Daily Limits**: Configurable daily retraining limits per model

### 3. Model Promotion
- **Comprehensive Evaluation**: Multi-criteria model evaluation
- **Statistical Testing**: Significance testing for performance improvements
- **Drift Analysis**: Pre-promotion drift analysis
- **Confidence Assessment**: Model confidence stability evaluation
- **MLflow Integration**: Seamless model stage transitions

### 4. Analytics and Reporting
- **Performance Trends**: Historical performance analysis
- **Model Comparison**: Side-by-side model performance comparison
- **Red Team Analysis**: Security test result analysis
- **Custom Metrics**: Comprehensive performance metrics calculation
- **Data Export**: Production data export for retraining

### 5. Monitoring and Alerting
- **Email Notifications**: Automated email alerts for critical events
- **Health Monitoring**: Service health and dependency monitoring
- **Metrics Collection**: Comprehensive Prometheus metrics
- **Distributed Tracing**: Request flow tracking with Jaeger
- **Audit Logging**: Complete audit trail for all operations

## External Integrations

### 1. Training Service
- **Purpose**: Trigger retraining jobs
- **Integration**: HTTP API calls for job submission
- **Features**: Job status monitoring, cancellation, progress tracking

### 2. Model API Service
- **Purpose**: Get model predictions for drift analysis
- **Integration**: HTTP API calls for prediction requests
- **Features**: Model performance evaluation, prediction comparison

### 3. MLflow
- **Purpose**: Model registry and version management
- **Integration**: MLflow client for model operations
- **Features**: Model promotion, version management, metadata storage

### 4. PostgreSQL
- **Purpose**: Analytics data storage and drift history
- **Integration**: Async database connection
- **Features**: Drift history, performance metrics, audit logs

### 5. Email Service
- **Purpose**: Alert notifications
- **Integration**: SMTP client for email sending
- **Features**: Drift alerts, performance notifications, system alerts

## Security Features

### 1. Data Protection
- **Data Anonymization**: Optional data anonymization for sensitive information
- **Access Control**: Fine-grained access control for analytics data
- **Audit Logging**: Complete audit trail for all analytics operations
- **Data Encryption**: Encryption at rest and in transit

### 2. Model Security
- **Model Validation**: Validation of model performance data
- **Secure Evaluation**: Secure model evaluation processes
- **Access Control**: Controlled access to model analytics
- **Integrity Checks**: Model integrity verification

### 3. System Security
- **Input Validation**: Comprehensive input validation and sanitization
- **Rate Limiting**: Protection against excessive API usage
- **Error Handling**: Secure error handling without information leakage
- **Monitoring**: Security monitoring and alerting

## Performance Optimizations

### 1. Efficient Drift Detection
- **Statistical Optimization**: Optimized statistical test implementations
- **Batch Processing**: Batch processing for large datasets
- **Caching**: Intelligent caching of drift detection results
- **Parallel Processing**: Parallel processing for multiple features

### 2. Database Optimization
- **Connection Pooling**: Efficient database connection management
- **Query Optimization**: Optimized database queries
- **Indexing**: Strategic database indexing
- **Data Partitioning**: Time-based data partitioning

### 3. Memory Management
- **Data Streaming**: Streaming processing for large datasets
- **Memory Limits**: Configurable memory limits
- **Garbage Collection**: Optimized garbage collection
- **Resource Monitoring**: Real-time resource usage monitoring

### 4. Caching Strategy
- **Result Caching**: Caching of drift detection results
- **Configuration Caching**: Caching of service configurations
- **Model Caching**: Caching of model metadata
- **Cache Invalidation**: Smart cache invalidation strategies

## Deployment Architecture

### 1. Containerization
- **Docker**: Service runs in a Docker container
- **Multi-stage Build**: Optimized Docker image with minimal dependencies
- **Health Checks**: Container health monitoring
- **Resource Limits**: Configurable CPU and memory limits

### 2. Service Discovery
- **Docker Compose**: Service orchestration and networking
- **Environment Variables**: Configuration through environment variables
- **Service Dependencies**: Proper service startup ordering

### 3. Monitoring
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting
- **Jaeger**: Distributed tracing and debugging
- **Health Checks**: Multiple levels of health monitoring

## Configuration

### 1. Environment Variables
```bash
# Core Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=ml_security_consolidated
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=password

# Service Configuration
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8006
SERVICE_VERSION=1.0.0

# External Services
TRAINING_SERVICE_URL=http://training:8002
MODEL_API_URL=http://model-api:8000
MLFLOW_TRACKING_URI=http://mlflow:5000

# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-password
```

### 2. Service Configuration
- **Drift Detection**: Configurable drift detection parameters
- **Auto-Retrain**: Configurable retraining triggers and limits
- **Model Promotion**: Configurable promotion criteria
- **Email Notifications**: Configurable email settings

## Error Handling

### 1. Exception Handling
- **Graceful Degradation**: Service continues to function even if some components fail
- **Error Recovery**: Automatic retry mechanisms for transient failures
- **User-friendly Messages**: Clear error messages for API consumers
- **Error Classification**: Categorized error handling

### 2. Logging
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Log Levels**: Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- **Correlation IDs**: Request tracing across service boundaries
- **Audit Logging**: Special audit logging for compliance

### 3. Monitoring
- **Health Checks**: Multiple levels of health monitoring
- **Metrics**: Comprehensive metrics for monitoring and alerting
- **Alerting**: Automated alerting for critical failures
- **Performance Monitoring**: Real-time performance monitoring

## Scalability

### 1. Horizontal Scaling
- **Stateless Design**: Service can be scaled horizontally
- **Load Balancing**: Support for load balancing across instances
- **Data Partitioning**: Time-based data partitioning for analytics
- **Cache Distribution**: Distributed caching strategies

### 2. Vertical Scaling
- **Resource Allocation**: Configurable CPU and memory limits
- **Performance Tuning**: Optimized for different hardware configurations
- **Resource Monitoring**: Real-time resource usage monitoring
- **Database Scaling**: Database connection pooling and optimization

## Future Enhancements

### 1. Planned Features
- **Advanced Analytics**: Machine learning-based analytics
- **Real-time Monitoring**: Real-time drift detection
- **A/B Testing**: Built-in A/B testing for models
- **Custom Metrics**: User-defined custom metrics

### 2. Performance Improvements
- **Streaming Analytics**: Real-time streaming analytics
- **Advanced Caching**: More sophisticated caching strategies
- **Database Optimization**: Further database optimization
- **API Optimization**: Enhanced API performance

## Conclusion

The Analytics Service provides a comprehensive, production-ready platform for ML analytics, drift detection, and automated retraining. Its modular architecture, robust error handling, and extensive monitoring capabilities make it suitable for enterprise environments requiring reliable and scalable ML operations.

The service integrates seamlessly with the broader ML Security platform, providing the foundation for intelligent, automated, and monitored machine learning operations.