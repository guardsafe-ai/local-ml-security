# MinIO Storage Service

## Overview

The MinIO Storage Service provides S3-compatible object storage for the ML Security Service. It serves as the central storage backend for MLflow artifacts, training data, model files, and other large data objects, offering high performance, scalability, and reliability.

## Features

### 🗄️ Object Storage
- **S3-Compatible API**: Full S3 API compatibility for seamless integration
- **High Performance**: Optimized for high-throughput workloads
- **Scalability**: Horizontal scaling capabilities
- **Durability**: Data replication and redundancy
- **Security**: Encryption at rest and in transit

### 📦 Data Management
- **Bucket Management**: Create, configure, and manage storage buckets
- **Object Lifecycle**: Automated lifecycle management policies
- **Versioning**: Object versioning and retention
- **Metadata**: Rich metadata support for objects
- **Access Control**: Fine-grained access control and permissions

### 🔧 Integration
- **MLflow Integration**: Seamless integration with MLflow artifact storage
- **Training Data Storage**: Centralized storage for training datasets
- **Model Storage**: Secure storage for trained models
- **Backup and Recovery**: Automated backup and disaster recovery
- **Monitoring**: Comprehensive monitoring and alerting

## Architecture

```
MinIO Storage Service
├── MinIO Server
│   ├── Object Storage Engine
│   ├── S3 API Gateway
│   ├── Metadata Store
│   └── Access Control
├── Storage Backend
│   ├── Local Storage
│   ├── Distributed Storage
│   └── Replication
├── Management
│   ├── Bucket Management
│   ├── User Management
│   ├── Policy Management
│   └── Lifecycle Management
└── Monitoring
    ├── Metrics Collection
    ├── Health Monitoring
    ├── Performance Monitoring
    └── Alerting
```

## Configuration

### Environment Variables

```bash
# MinIO Configuration
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_DEFAULT_BUCKETS=mlflow-artifacts,ml-security,training-data,red-team-data

# Storage Configuration
MINIO_DATA_DIR=/data
MINIO_CONFIG_DIR=/etc/minio
MINIO_CERTS_DIR=/etc/minio/certs

# Network Configuration
MINIO_SERVER_URL=http://minio:9000
MINIO_CONSOLE_ADDRESS=:9001
MINIO_BROWSER_REDIRECT_URL=http://localhost:9001

# Security Configuration
MINIO_KMS_SECRET_KEY=minio-kms-key
MINIO_KMS_MASTER_KEY=minio-kms-master-key
```

### Docker Configuration

```yaml
version: '3.8'
services:
  minio:
    image: minio/minio:latest
    container_name: local-ml-security-minio-1
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
      MINIO_DEFAULT_BUCKETS: mlflow-artifacts,ml-security,training-data,red-team-data
    volumes:
      - minio_data:/data
      - minio_config:/etc/minio
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3
```

## Bucket Structure

### Default Buckets
```
mlflow-artifacts/
├── 114/                          # Experiment ID
│   ├── 8de878208a774784aeac41e66554d0df/  # Run ID
│   │   ├── artifacts/
│   │   │   ├── model/            # Model artifacts
│   │   │   ├── data/             # Dataset artifacts
│   │   │   └── logs/             # Log artifacts
│   │   └── meta.yaml
│   └── ...
├── 113/
└── ...

ml-security/
├── models/                       # Trained models
│   ├── distilbert/
│   │   ├── v1.0.1234/
│   │   └── v1.0.1233/
│   └── bert/
├── data/                         # Training data
│   ├── fresh/                    # Fresh data files
│   ├── used/                     # Used data files
│   └── archived/                 # Archived data files
└── configs/                      # Configuration files

training-data/
├── custom/                       # Custom training data
├── synthetic/                    # Synthetic training data
├── augmented/                    # Augmented training data
└── processed/                    # Processed training data

red-team-data/
├── attacks/                      # Attack patterns
├── results/                      # Test results
├── vulnerabilities/              # Vulnerability data
└── reports/                      # Security reports
```

## API Usage

### S3-Compatible API

#### List Buckets
```bash
curl -X GET http://localhost:9000/ \
  -H "Authorization: AWS4-HMAC-SHA256 ..."
```

#### Create Bucket
```bash
curl -X PUT http://localhost:9000/new-bucket \
  -H "Authorization: AWS4-HMAC-SHA256 ..."
```

#### Upload Object
```bash
curl -X PUT http://localhost:9000/bucket/object-key \
  -H "Authorization: AWS4-HMAC-SHA256 ..." \
  -H "Content-Type: application/octet-stream" \
  --data-binary @file.txt
```

#### Download Object
```bash
curl -X GET http://localhost:9000/bucket/object-key \
  -H "Authorization: AWS4-HMAC-SHA256 ..." \
  -o downloaded-file.txt
```

#### Delete Object
```bash
curl -X DELETE http://localhost:9000/bucket/object-key \
  -H "Authorization: AWS4-HMAC-SHA256 ..."
```

### Python SDK Usage

#### Setup Client
```python
from minio import Minio
from minio.error import S3Error

# Initialize MinIO client
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
```

#### Bucket Operations
```python
# Create bucket
try:
    client.make_bucket("new-bucket")
    print("Bucket created successfully")
except S3Error as e:
    print(f"Error creating bucket: {e}")

# List buckets
buckets = client.list_buckets()
for bucket in buckets:
    print(f"Bucket: {bucket.name}, Created: {bucket.creation_date}")

# Check if bucket exists
if client.bucket_exists("ml-security"):
    print("Bucket exists")
```

#### Object Operations
```python
# Upload object
try:
    client.fput_object(
        "ml-security", 
        "models/distilbert/model.pkl", 
        "/path/to/local/model.pkl"
    )
    print("Object uploaded successfully")
except S3Error as e:
    print(f"Error uploading object: {e}")

# Download object
try:
    client.fget_object(
        "ml-security", 
        "models/distilbert/model.pkl", 
        "/path/to/downloaded/model.pkl"
    )
    print("Object downloaded successfully")
except S3Error as e:
    print(f"Error downloading object: {e}")

# List objects
objects = client.list_objects("ml-security", prefix="models/")
for obj in objects:
    print(f"Object: {obj.object_name}, Size: {obj.size}")

# Delete object
try:
    client.remove_object("ml-security", "models/distilbert/model.pkl")
    print("Object deleted successfully")
except S3Error as e:
    print(f"Error deleting object: {e}")
```

## Integration with ML Security Service

### MLflow Integration
```python
# MLflow configuration for MinIO
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_registry_uri("s3://mlflow-artifacts")

# Configure S3 endpoint
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
```

### Training Service Integration
```python
# Training service MinIO configuration
MINIO_CONFIG = {
    "endpoint": "minio:9000",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "bucket": "ml-security",
    "secure": False
}
```

### Data Management Integration
```python
# Data management service configuration
DATA_BUCKETS = {
    "training_data": "ml-security/data",
    "models": "ml-security/models",
    "artifacts": "mlflow-artifacts",
    "red_team": "red-team-data"
}
```

## Web Console

### Access Console
```bash
# Access MinIO console
open http://localhost:9001
```

### Console Features
- **Bucket Management**: Create, configure, and manage buckets
- **Object Browser**: Browse and manage objects
- **User Management**: Manage users and access policies
- **Monitoring**: View storage metrics and health
- **Logs**: View system logs and events

### Default Credentials
- **Username**: minioadmin
- **Password**: minioadmin

## Security Configuration

### Access Control
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": ["arn:aws:iam::account:user/mlflow"]
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::mlflow-artifacts/*"
      ]
    }
  ]
}
```

### Encryption
```bash
# Enable encryption at rest
export MINIO_KMS_SECRET_KEY=minio-kms-key
export MINIO_KMS_MASTER_KEY=minio-kms-master-key

# Enable encryption in transit
export MINIO_CERTS_DIR=/etc/minio/certs
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check MinIO health
curl http://localhost:9000/minio/health/live
curl http://localhost:9000/minio/health/ready

# Check storage usage
docker exec local-ml-security-minio-1 mc du /data

# Check bucket status
docker exec local-ml-security-minio-1 mc ls /data
```

### Metrics Collection
```bash
# Enable metrics collection
export MINIO_PROMETHEUS_AUTH_TYPE=public
export MINIO_PROMETHEUS_URL=http://prometheus:9090

# Access metrics
curl http://localhost:9000/minio/v2/metrics/cluster
```

### Backup and Recovery
```bash
# Backup data
docker exec local-ml-security-minio-1 mc mirror /data ./minio-backup

# Restore data
docker exec local-ml-security-minio-1 mc mirror ./minio-backup /data
```

## Performance Optimization

### Storage Optimization
- **SSD Storage**: Use SSD storage for better performance
- **RAID Configuration**: Configure RAID for redundancy
- **Network Optimization**: Use high-speed network connections
- **Memory Configuration**: Allocate sufficient memory

### Configuration Tuning
```bash
# Performance tuning
export MINIO_CACHE_DRIVES="/tmp/cache1,/tmp/cache2"
export MINIO_CACHE_EXCLUDE="*.pdf,*.mp4"
export MINIO_CACHE_QUOTA=80
export MINIO_CACHE_AFTER=3
export MINIO_CACHE_WATERMARK_LOW=70
export MINIO_CACHE_WATERMARK_HIGH=90
```

## Troubleshooting

### Common Issues
1. **Connection Refused**: Check MinIO service status
2. **Authentication Failed**: Verify credentials
3. **Bucket Not Found**: Check bucket existence and permissions
4. **Storage Full**: Check disk space and cleanup old data

### Debug Commands
```bash
# Check service status
docker-compose ps minio

# Check service logs
docker logs local-ml-security-minio-1

# Check storage usage
docker exec local-ml-security-minio-1 df -h

# Check MinIO configuration
docker exec local-ml-security-minio-1 cat /etc/minio/config.json
```

### Log Analysis
```bash
# View MinIO logs
docker logs local-ml-security-minio-1 | grep ERROR
docker logs local-ml-security-minio-1 | grep WARN

# Check access logs
docker exec local-ml-security-minio-1 mc admin trace local
```

## Data Lifecycle Management

### Lifecycle Policies
```json
{
  "Rules": [
    {
      "ID": "DeleteOldVersions",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "mlflow-artifacts/"
      },
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    },
    {
      "ID": "ArchiveOldData",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "training-data/archived/"
      },
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

### Cleanup Scripts
```bash
#!/bin/bash
# Cleanup old artifacts
docker exec local-ml-security-minio-1 mc find /data/mlflow-artifacts \
  --older-than 90d --exec "mc rm {}"

# Cleanup old training data
docker exec local-ml-security-minio-1 mc find /data/ml-security/data/used \
  --older-than 30d --exec "mc rm {}"
```

## Dependencies

### Core Dependencies
- `minio`: MinIO Python SDK
- `boto3`: AWS S3 SDK (for compatibility)
- `requests`: HTTP client
- `urllib3`: HTTP library

### Optional Dependencies
- `minio[event]`: Event notifications
- `minio[select]`: S3 Select support
- `minio[encryption]`: Encryption support

## Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for new features
4. Ensure backward compatibility
5. Test with different storage configurations

## License

This service is part of the ML Security Service and follows the same license terms.
