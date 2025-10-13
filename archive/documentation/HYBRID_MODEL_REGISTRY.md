# 🔄 Hybrid Model Registry Architecture

## Overview

The ML Security Service now uses a **hybrid approach** for model registry management, combining multiple storage layers for optimal performance, reliability, and data persistence.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Training      │    │   Model API     │
│   (Streamlit)   │    │   Service       │    │   Service       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Cache   │◄───┤   MLflow        │    │   Direct        │
│   (1 hour TTL)  │    │   Registry      │    │   MLflow Query  │
└─────────────────┘    │   (Primary)     │    └─────────────────┘
                       └─────────┬───────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │   MinIO Storage │
                       │   (Artifacts)   │
                       └─────────────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │   Local File    │
                       │   (Backup)      │
                       └─────────────────┘
```

## 🎯 Storage Layers

### 1. **Primary Storage: MLflow Model Registry**
- **Purpose**: Authoritative source of truth
- **Features**: Versioning, staging, metadata
- **Persistence**: Permanent (never expires)
- **Access**: Via MLflow API

```python
# Model registration in MLflow
mlflow.pytorch.log_model(
    pytorch_model=trainer.model,
    artifact_path=f"models/{model_name}",
    registered_model_name=f"security_{model_name}",
    extra_files=[...]
)

# Set model stage
client.set_registered_model_alias(f"security_{model_name}", "latest", model_version)
```

### 2. **Cache Layer: Redis**
- **Purpose**: Fast access to registry data
- **TTL**: 1 hour (3600 seconds)
- **Features**: In-memory performance
- **Fallback**: If expired, queries MLflow

```python
# Cache model registry in Redis
cache_data = {
    "latest_models": self.model_registry["latest"],
    "best_models": self.model_registry["best"],
    "versions": self.model_registry["versions"],
    "timestamp": datetime.now().isoformat()
}
self.redis_client.setex("model_registry", 3600, json.dumps(cache_data))
```

### 3. **Backup Storage: Local File**
- **Purpose**: Disaster recovery
- **Location**: `/app/data/model_registry.json`
- **Features**: JSON format, human-readable
- **Usage**: Fallback when both Redis and MLflow fail

```python
# Save registry to local file
registry_file = Path("/app/data/model_registry.json")
with open(registry_file, 'w') as f:
    json.dump(self.model_registry, f, indent=2)
```

## 🔄 Data Flow

### **Loading Models (Training Service)**

1. **Redis Cache Check**
   ```python
   cached_data = self.redis_client.get("model_registry")
   if cached_data:
       # Use cached data
       return json.loads(cached_data)
   ```

2. **MLflow Query (Primary)**
   ```python
   client = MlflowClient()
   registered_models = client.search_registered_models()
   # Process models and build registry
   ```

3. **File Backup (Fallback)**
   ```python
   if not self.model_registry["latest"]:
       self.load_registry_from_file()
   ```

4. **Cache Update**
   ```python
   self.save_model_registry()  # Updates Redis + file
   ```

### **Loading Models (Model API Service)**

1. **Training Service API**
   ```python
   response = requests.get("http://training:8002/models/latest")
   if response.status_code == 200:
       return response.json()
   ```

2. **Direct MLflow Query**
   ```python
   client = MlflowClient()
   model_versions = client.get_latest_versions(f"security_{model_name}")
   ```

3. **Pre-trained Fallback**
   ```python
   # Load from Hugging Face if no trained model
   model = AutoModel.from_pretrained(config["path"])
   ```

## 📊 Benefits

### **Performance**
- ✅ **Fast Access**: Redis cache provides sub-millisecond response times
- ✅ **Reduced Latency**: Avoids MLflow API calls for cached data
- ✅ **Scalability**: Can handle high-frequency requests

### **Reliability**
- ✅ **Data Persistence**: MLflow ensures data never expires
- ✅ **Multiple Fallbacks**: Redis → MLflow → File → Pre-trained
- ✅ **Fault Tolerance**: System continues working even if one layer fails

### **Consistency**
- ✅ **Single Source of Truth**: MLflow is the authoritative source
- ✅ **Automatic Sync**: Cache is updated when models are trained
- ✅ **Version Control**: MLflow handles model versioning

## 🛠️ Implementation Details

### **Training Service Updates**

```python
class ModelTrainer:
    def save_model_registry(self):
        """Hybrid approach: MLflow (primary) + Redis (cache) + File (backup)"""
        # Primary: Already saved to MLflow in train_model()
        
        # Cache: Redis with 1-hour TTL
        if self.redis_client:
            cache_data = {...}
            self.redis_client.setex("model_registry", 3600, json.dumps(cache_data))
        
        # Backup: Local file
        self.save_registry_to_file()
    
    def load_existing_models_from_mlflow(self):
        """Load using hybrid approach: Redis -> MLflow -> File"""
        # Step 1: Try Redis cache
        if self.redis_client:
            cached_data = self.redis_client.get("model_registry")
            if cached_data:
                return json.loads(cached_data)
        
        # Step 2: Fallback to MLflow
        # ... MLflow query logic ...
        
        # Step 3: Final fallback to file
        if not self.model_registry["latest"]:
            self.load_registry_from_file()
```

### **Model API Service Updates**

```python
class ModelManager:
    def get_trained_model_info(self, model_name: str):
        """Hybrid approach: Training API -> MLflow -> Fallback"""
        # Step 1: Try training service (Redis cache)
        response = requests.get("http://training:8002/models/latest")
        if response.status_code == 200:
            return response.json()
        
        # Step 2: Direct MLflow query
        client = MlflowClient()
        model_versions = client.get_latest_versions(f"security_{model_name}")
        
        # Step 3: No trained model found
        return None
```

## 🔍 Monitoring & Debugging

### **Log Messages**
- `[HYBRID]` - Hybrid approach operations
- `[CACHE]` - Redis cache operations
- `[MLFLOW]` - MLflow operations
- `[FALLBACK]` - Fallback operations

### **Health Checks**
```python
# Check Redis cache
redis_client.get("model_registry")

# Check MLflow
client.search_registered_models()

# Check local file
Path("/app/data/model_registry.json").exists()
```

### **Test Script**
```bash
python test_hybrid_approach.py
```

## ⚙️ Configuration

### **Redis TTL**
```python
# Current: 1 hour
self.redis_client.setex("model_registry", 3600, json.dumps(cache_data))

# Adjustable based on needs:
# - 30 minutes (1800): More frequent MLflow sync
# - 2 hours (7200): Less frequent sync, more cache hits
# - No expiration (-1): Manual cache management
```

### **File Backup Location**
```python
# Current: /app/data/model_registry.json
registry_file = Path("/app/data/model_registry.json")

# Can be changed to:
# - /app/backups/model_registry.json
# - /shared/model_registry.json
# - /tmp/model_registry.json (not recommended)
```

## 🚀 Usage Examples

### **Training a New Model**
```python
# 1. Model is trained and saved to MLflow
trainer.train_model(model_name, data_path, config)

# 2. Registry is automatically updated
trainer._add_model_aliases(model_name, version, metrics)

# 3. Cache is updated
trainer.save_model_registry()  # Updates Redis + file
```

### **Loading Models for Inference**
```python
# 1. Model API queries training service
response = requests.get("http://training:8002/models/latest")

# 2. If Redis cache hit, return immediately
# 3. If cache miss, training service queries MLflow
# 4. Model API loads from MLflow using returned URI
model = mlflow.pytorch.load_model(model_uri)
```

### **Handling Cache Expiration**
```python
# When Redis TTL expires:
# 1. Training service detects empty cache
# 2. Queries MLflow for latest models
# 3. Updates Redis cache with fresh data
# 4. Model API gets updated data
```

## 🔧 Troubleshooting

### **Common Issues**

1. **Redis Cache Expired**
   - **Symptom**: Models not found in registry
   - **Solution**: System automatically falls back to MLflow
   - **Prevention**: Monitor Redis TTL

2. **MLflow Unavailable**
   - **Symptom**: Training fails, models not registered
   - **Solution**: Check MLflow service status
   - **Fallback**: Use local file backup

3. **File Backup Missing**
   - **Symptom**: No fallback data available
   - **Solution**: Train new models to recreate registry
   - **Prevention**: Regular backup verification

### **Debug Commands**
```bash
# Check Redis cache
docker exec -it ml-security-redis-1 redis-cli get "model_registry"

# Check MLflow models
docker exec -it ml-security-mlflow-1 mlflow models list

# Check local file
docker exec -it ml-security-training-1 cat /app/data/model_registry.json
```

## 📈 Performance Metrics

### **Expected Performance**
- **Redis Cache Hit**: < 1ms
- **MLflow Query**: 100-500ms
- **File Read**: 10-50ms
- **Model Loading**: 1-10s (depending on size)

### **Cache Hit Rate**
- **Target**: > 90% for frequently accessed models
- **Monitoring**: Track via Redis metrics
- **Optimization**: Adjust TTL based on usage patterns

## 🎯 Best Practices

1. **Monitor Cache TTL**: Set up alerts for low TTL values
2. **Regular Backups**: Verify file backup integrity
3. **MLflow Health**: Monitor MLflow service availability
4. **Log Analysis**: Watch for fallback patterns
5. **Performance Testing**: Regular load testing

## 🔮 Future Enhancements

1. **Distributed Cache**: Redis Cluster for high availability
2. **Metrics Collection**: Prometheus metrics for monitoring
3. **Auto-scaling**: Dynamic TTL based on usage
4. **Compression**: Compress cached data for memory efficiency
5. **Encryption**: Encrypt sensitive registry data

---

This hybrid approach ensures that the ML Security Service is both performant and reliable, with multiple layers of fallback to guarantee continuous operation even in the face of individual component failures.
