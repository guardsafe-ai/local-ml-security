# Quick Reference: Adding Models to ML Security Service

## 🚀 Quick Start

### Add a Single Model (Easiest)

```bash
# Add a Hugging Face model
python add_model.py electra-base google/electra-base-discriminator

# Add with custom priority
python add_model.py albert-base albert-base-v2 pytorch 6

# Add a local model
python add_model.py my-model ./path/to/local/model pytorch 7
```

### Add Multiple Models

```bash
# Use the integrated setup script
python setup_models_integrated.py
```

## 📋 Current Models

The system currently includes these pre-configured models:

- **distilbert** (`distilbert-base-uncased`) - Priority 1
- **bert-base** (`bert-base-uncased`) - Priority 2  
- **roberta-base** (`roberta-base`) - Priority 3
- **deberta-v3-base** (`microsoft/deberta-v3-base`) - Priority 4

## 🔧 Manual Configuration

To add models manually, edit `services/model-api/main.py`:

```python
self.model_configs = {
    # ... existing models ...
    "your-new-model": {
        "type": "pytorch",
        "path": "huggingface/model-name",
        "priority": 5
    }
}
```

## ✅ Verification Steps

After adding a model:

1. **Restart the service**:
   ```bash
   docker-compose restart model-api
   ```

2. **Check if loaded**:
   ```bash
   curl http://localhost:8000/models | grep your-model
   ```

3. **Test prediction**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Test text", "models": ["your-model_pretrained"]}'
   ```

4. **Check UI**:
   - Open http://localhost:8501
   - Go to "Models" page
   - Look for your new model

## 🎯 Model Requirements

Models must support **5-class classification**:
- `prompt_injection`
- `jailbreak`
- `system_extraction`
- `code_injection`
- `benign`

## 🔍 Troubleshooting

### Model Not Loading
```bash
# Check logs
docker-compose logs model-api | grep your-model

# Check service health
curl http://localhost:8000/health
```

### Model Not Appearing in UI
```bash
# Restart monitoring service
docker-compose restart monitoring

# Refresh browser cache
```

### Slow Loading
- Check internet connection (for Hugging Face models)
- Monitor memory usage: `docker stats model-api`
- Consider using smaller models

## 📚 Full Documentation

For complete details, see:
- [ADD_NEW_MODELS_GUIDE.md](./ADD_NEW_MODELS_GUIDE.md) - Comprehensive guide
- [API_DOCUMENTATION_INDEX.md](./API_DOCUMENTATION_INDEX.md) - All API endpoints
