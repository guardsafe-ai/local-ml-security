# 🗄️ MinIO Storage System for ML Security

## Overview

The ML Security system now uses **MinIO** as the centralized storage solution for all training data, model artifacts, and system files. This provides better scalability, consistency, and management compared to local file storage.

## 🏗️ Architecture

### MinIO Configuration
- **Endpoint**: `http://minio:9000` (internal) / `http://localhost:9000` (external)
- **Console**: `http://localhost:9001`
- **Credentials**: `minioadmin` / `minioadmin`
- **Bucket**: `ml-security`

### Folder Structure

```
ml-security/
├── training-data/           # Sample training data
│   ├── sample_training_data_latest.jsonl
│   └── sample_training_data_v1.0.jsonl
├── red-team-data/          # Red team test results
│   ├── red_team_data_test123_20241201_143022.jsonl
│   └── red_team_data_vulnerabilities_20241201_143022.jsonl
├── combined-data/          # Combined training data
│   ├── combined/
│   │   └── combined_data_combined_latest_20241201_143022.jsonl
│   └── distilbert/
│       └── combined_data_distilbert_latest_20241201_143022.jsonl
├── model-specific/         # Model-specific training data
│   ├── distilbert/
│   │   ├── distilbert_specific_prompt_injection_20241201_143022.jsonl
│   │   └── distilbert_specific_jailbreak_20241201_143022.jsonl
│   └── bert/
│       └── bert_specific_code_injection_20241201_143022.jsonl
├── backups/               # System backups
│   ├── training/
│   │   └── backup_training_20241201_143022.jsonl
│   └── red-team/
│       └── backup_red-team_20241201_143022.jsonl
├── logs/                  # System logs
└── configs/               # Configuration files
```

## 🔧 Services Integration

### 1. Training Service (`services/training/main.py`)

#### `/create-sample-data` Endpoint
```python
@app.post("/create-sample-data")
async def create_sample_data():
    # Creates 75 training examples (15 per category)
    # Saves to MinIO: training-data/sample_training_data_latest.jsonl
    # Also saves locally for backward compatibility
```

**Response:**
```json
{
  "message": "Sample data created and saved to MinIO",
  "minio_path": "training-data/sample_training_data_latest.jsonl",
  "local_path": "/app/training_data/sample_training_data.jsonl",
  "count": 75
}
```

### 2. Red Team Service (`services/red-team/main.py`)

#### Data Loading
- **Sample Data**: Loads from MinIO first, falls back to training service
- **Red Team Data**: Loads from MinIO first, falls back to local files

#### Data Saving
- **Test Results**: Saves to `red-team-data/red_team_data_{test_id}_{timestamp}.jsonl`
- **Vulnerabilities**: Saves to `red-team-data/red_team_data_vulnerabilities_{timestamp}.jsonl`
- **Combined Data**: Saves to `combined-data/{model_name}/combined_data_{model_name}_{version}_{timestamp}.jsonl`

### 3. MinIO Storage Service (`services/minio-storage/main.py`)

#### Key Methods

```python
# Training Data
minio_storage.save_sample_training_data(data, version="latest")
minio_storage.load_sample_training_data(version="latest")

# Red Team Data
minio_storage.save_red_team_data(data, test_id)
minio_storage.load_red_team_data(test_id)

# Combined Data
minio_storage.save_combined_data(data, model_name, version="latest")
minio_storage.load_combined_data(model_name, version="latest")

# Model-Specific Data
minio_storage.save_model_specific_data(data, model_name, failure_type)
minio_storage.load_model_specific_data(model_name, failure_type)

# Backups
minio_storage.create_backup(data, backup_type, description)
```

## 📊 Data Flow

### 1. Sample Data Creation
```
curl -X POST "http://localhost:8002/create-sample-data"
    ↓
Training Service creates 75 examples
    ↓
Saves to MinIO: training-data/sample_training_data_latest.jsonl
    ↓
Also saves locally for backward compatibility
```

### 2. Red Team Testing
```
Red Team Service starts test
    ↓
Loads sample data from MinIO
    ↓
Generates attack patterns
    ↓
Tests models
    ↓
Saves results to MinIO: red-team-data/red_team_data_{test_id}_{timestamp}.jsonl
    ↓
Saves vulnerabilities to MinIO: red-team-data/red_team_data_vulnerabilities_{timestamp}.jsonl
```

### 3. Model Retraining
```
Red Team finds vulnerabilities
    ↓
Loads red team data from MinIO
    ↓
Loads sample data from MinIO
    ↓
Combines and deduplicates data
    ↓
Saves combined data to MinIO: combined-data/{model_name}/combined_data_{model_name}_{version}_{timestamp}.jsonl
    ↓
Triggers model retraining with combined data
```

## 🔄 Migration from Local to MinIO

### Current Status
- **Hybrid Approach**: Both MinIO and local storage are used
- **MinIO Primary**: All new data is saved to MinIO first
- **Local Fallback**: Local files are maintained for backward compatibility
- **Gradual Migration**: Services try MinIO first, fall back to local if needed

### Benefits
1. **Centralized Storage**: All data in one place
2. **Scalability**: Easy to scale storage
3. **Consistency**: Same data structure across services
4. **Backup**: Built-in backup and versioning
5. **Access Control**: Fine-grained access control
6. **Monitoring**: Better visibility into data usage

## 🚀 Usage Examples

### 1. Create Sample Training Data
```bash
curl -X POST "http://localhost:8002/create-sample-data"
```

### 2. List Files in MinIO
```python
from services.minio_storage.main import minio_storage

# List all training data files
files = minio_storage.list_files("training_data")
print(files)

# List red team data files
files = minio_storage.list_files("red_team_data")
print(files)
```

### 3. Access MinIO Console
- Open: `http://localhost:9001`
- Username: `minioadmin`
- Password: `minioadmin`
- Navigate to `ml-security` bucket

### 4. Download Files
```python
# Get presigned URL for file download
url = minio_storage.get_file_url("training-data/sample_training_data_latest.jsonl")
print(f"Download URL: {url}")
```

## 🔧 Configuration

### Environment Variables
```bash
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=ml-security
```

### Docker Compose
```yaml
minio:
  image: minio/minio:latest
  ports:
    - "9000:9000"
    - "9001:9001"
  environment:
    - MINIO_ROOT_USER=minioadmin
    - MINIO_ROOT_PASSWORD=minioadmin
  volumes:
    - minio_data:/data
  command: server /data --console-address ":9001"
```

## 📈 Monitoring

### MinIO Metrics
- **Storage Usage**: Total size of data stored
- **Request Count**: Number of API requests
- **Error Rate**: Failed requests percentage
- **Latency**: Average response time

### Data Metrics
- **Training Data**: Number of examples per category
- **Red Team Data**: Test results and vulnerability counts
- **Model Performance**: Detection rates and accuracy
- **Storage Growth**: Data volume over time

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if MinIO is running
   docker ps | grep minio
   
   # Check logs
   docker logs ml-security-minio-1
   ```

2. **Bucket Not Found**
   ```python
   # The service automatically creates the bucket
   # Check if it exists
   minio_storage._ensure_bucket_exists()
   ```

3. **Permission Denied**
   ```bash
   # Check credentials
   echo $MINIO_ACCESS_KEY
   echo $MINIO_SECRET_KEY
   ```

4. **File Not Found**
   ```python
   # List files to see what's available
   files = minio_storage.list_files("training_data")
   print(files)
   ```

## 🔮 Future Enhancements

1. **Data Versioning**: Automatic versioning of training data
2. **Data Lineage**: Track data flow and transformations
3. **Data Quality**: Automated data validation and cleaning
4. **Data Analytics**: Advanced analytics on stored data
5. **Data Archiving**: Automatic archiving of old data
6. **Data Encryption**: Encrypt sensitive data at rest
7. **Data Replication**: Cross-region data replication
8. **Data Lifecycle**: Automatic data lifecycle management

## 📝 Summary

The MinIO storage system provides:

- **Centralized Storage**: All data in one place
- **Proper Organization**: Well-structured folder hierarchy
- **Service Integration**: Seamless integration with all services
- **Backward Compatibility**: Local files maintained for compatibility
- **Scalability**: Easy to scale and manage
- **Monitoring**: Better visibility and control

This system ensures that all training data, model artifacts, and system files are properly organized and accessible across all services, making the ML Security system more robust and maintainable! 🚀
