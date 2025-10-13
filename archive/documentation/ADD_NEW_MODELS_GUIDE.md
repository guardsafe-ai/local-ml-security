# How to Add New Models to the ML Security Service

This guide explains how to properly add new models to the ML Security Service system.

## Current Model Loading System

The system has **two model loading mechanisms**:

1. **Pre-trained Models**: Loaded from Hugging Face Hub automatically
2. **Trained Models**: Loaded from MLflow/MinIO after training

## Method 1: Add Pre-trained Models (Recommended)

### Step 1: Modify Model Configuration

Edit `services/model-api/main.py` and add your model to the `model_configs` dictionary:

```python
# In services/model-api/main.py, around line 216
self.model_configs = {
    "distilbert": {
        "type": "pytorch",
        "path": "distilbert-base-uncased",
        "priority": 1
    },
    "bert-base": {
        "type": "pytorch",
        "path": "bert-base-uncased", 
        "priority": 2
    },
    "roberta-base": {
        "type": "pytorch", 
        "path": "roberta-base",
        "priority": 3
    },
    "deberta-v3-base": {
        "type": "pytorch",
        "path": "microsoft/deberta-v3-base",
        "priority": 4
    },
    # ADD YOUR NEW MODEL HERE
    "your-new-model": {
        "type": "pytorch",
        "path": "huggingface/model-name",
        "priority": 5
    }
}
```

### Step 2: Supported Model Types

The system supports these model types:

#### PyTorch Models (Recommended)
```python
"model-name": {
    "type": "pytorch",
    "path": "huggingface/model-name",
    "priority": 1-10  # Higher number = higher priority
}
```

#### Scikit-learn Models
```python
"model-name": {
    "type": "sklearn", 
    "path": "path/to/model.pkl",
    "priority": 1-10
}
```

### Step 3: Model Requirements

For security classification, models must:

1. **Support sequence classification** (5 classes)
2. **Have these labels**:
   - `prompt_injection`
   - `jailbreak`
   - `system_extraction` 
   - `code_injection`
   - `benign`

### Step 4: Restart Services

After adding a model:

```bash
# Restart the model-api service
docker-compose restart model-api

# Or restart all services
docker-compose down && docker-compose up -d
```

### Step 5: Verify Model Loading

```bash
# Check if model is loaded
curl http://localhost:8000/models

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text", "models": ["your-new-model_pretrained"]}'
```

## Method 2: Add Custom Models

### Step 1: Prepare Your Model

If you have a custom model, ensure it:

1. **Saves in the correct format**:
   ```python
   # For PyTorch models
   model.save_pretrained("./path/to/model")
   tokenizer.save_pretrained("./path/to/model")
   
   # For scikit-learn models
   joblib.dump(model, "./path/to/model.pkl")
   ```

2. **Has the correct configuration**:
   ```json
   {
     "model_type": "your_model_type",
     "num_labels": 5,
     "id2label": {
       "0": "prompt_injection",
       "1": "jailbreak", 
       "2": "system_extraction",
       "3": "code_injection",
       "4": "benign"
     },
     "label2id": {
       "prompt_injection": 0,
       "jailbreak": 1,
       "system_extraction": 2,
       "code_injection": 3,
       "benign": 4
     }
   }
   ```

### Step 2: Add to Configuration

```python
"your-custom-model": {
    "type": "pytorch",  # or "sklearn"
    "path": "./path/to/your/model",  # Local path
    "priority": 5
}
```

## Method 3: Train New Models

### Step 1: Use the Training Service

```bash
# Create sample data
curl -X POST http://localhost:8002/create-sample-data

# Train a new model
curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "your-new-model",
    "training_data_path": "/app/training_data/sample_training_data.jsonl"
  }'
```

### Step 2: The trained model will be automatically available

Trained models are automatically loaded from MLflow/MinIO and will appear as `{model_name}_trained`.

## Example: Adding a New Hugging Face Model

Let's add `microsoft/DialoGPT-medium` as an example:

### Step 1: Edit Configuration

```python
# In services/model-api/main.py
self.model_configs = {
    # ... existing models ...
    "dialogpt-medium": {
        "type": "pytorch",
        "path": "microsoft/DialoGPT-medium",
        "priority": 5
    }
}
```

### Step 2: Restart Service

```bash
docker-compose restart model-api
```

### Step 3: Test

```bash
# Check if loaded
curl http://localhost:8000/models | grep dialogpt

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test text", "models": ["dialogpt-medium_pretrained"]}'
```

## Example: Adding a Custom Local Model

### Step 1: Prepare Model Directory

```bash
mkdir -p ./data/models/my-custom-model
# Copy your model files there
```

### Step 2: Add Configuration

```python
"my-custom-model": {
    "type": "pytorch",
    "path": "./data/models/my-custom-model",
    "priority": 6
}
```

### Step 3: Restart and Test

```bash
docker-compose restart model-api
curl http://localhost:8000/models
```

## Troubleshooting

### Model Not Loading

1. **Check logs**:
   ```bash
   docker-compose logs model-api | grep "your-model"
   ```

2. **Verify model path**:
   ```bash
   # For Hugging Face models
   curl -I https://huggingface.co/your-model-name
   
   # For local models
   ls -la ./path/to/your/model
   ```

3. **Check model compatibility**:
   - Ensure it supports sequence classification
   - Verify it has the correct number of labels (5)
   - Check if it's compatible with transformers library

### Model Loading Slowly

1. **Check internet connection** (for Hugging Face models)
2. **Monitor memory usage**:
   ```bash
   docker stats model-api
   ```
3. **Consider using smaller models** or model quantization

### Model Not Appearing in UI

1. **Check if model is loaded**:
   ```bash
   curl http://localhost:8000/models
   ```

2. **Restart monitoring service**:
   ```bash
   docker-compose restart monitoring
   ```

3. **Check browser cache** and refresh the UI

## Best Practices

### 1. Model Selection

- **Start with smaller models** for testing
- **Use models designed for classification**
- **Consider memory requirements**
- **Test performance before production**

### 2. Configuration

- **Use descriptive model names**
- **Set appropriate priorities**
- **Group related models together**
- **Document model purposes**

### 3. Testing

- **Test with sample data first**
- **Verify prediction accuracy**
- **Check response times**
- **Monitor resource usage**

### 4. Production

- **Use trained models for production**
- **Monitor model performance**
- **Set up alerts for model failures**
- **Regular model updates**

## Integration with setup_models.py

The `setup_models.py` file you showed is **not directly integrated** with the main system. To use it:

### Option 1: Modify setup_models.py

Update it to work with the main system:

```python
# In setup_models.py, modify the models dictionary
models = {
    "deberta-v3-base": "microsoft/deberta-v3-base",  # Already in system
    "roberta-base": "roberta-base",                  # Already in system
    "bert-base": "bert-base-uncased",                # Already in system
    "distilbert": "distilbert-base-uncased",         # Already in system
    # Add your new models here
    "your-new-model": "huggingface/your-model-name"
}
```

### Option 2: Use the Training Service Instead

Instead of using `setup_models.py`, use the built-in training service:

```bash
# This will automatically handle model downloading and configuration
curl -X POST http://localhost:8002/train \
  -H "Content-Type: application/json" \
  -d '{"model_name": "your-model-name"}'
```

## Summary

To add new models to the ML Security Service:

1. **For pre-trained models**: Edit `services/model-api/main.py` and add to `model_configs`
2. **For custom models**: Prepare your model and add to `model_configs`
3. **For trained models**: Use the training service API
4. **Restart services** after making changes
5. **Test the model** to ensure it works correctly

The system will automatically handle model loading, caching, and serving once properly configured.
