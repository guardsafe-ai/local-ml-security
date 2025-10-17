# Dynamic Batching System Guide 🚀

## Overview

Our ML Security Platform implements a **sophisticated dynamic batching system** that intelligently optimizes inference performance based on request patterns, system load, and use case requirements.

## Why Dynamic Batching? 🤔

### **Performance Benefits**
- **3-10x throughput improvement** compared to single-request processing
- **Better GPU utilization** - GPUs are most efficient with batch sizes of 8-32+ samples
- **Reduced overhead** - Fewer model loading/unloading cycles
- **Memory efficiency** - Better memory bandwidth utilization

### **When to Use Each Strategy**

| Strategy | Use Case | Latency | Throughput | Best For |
|----------|----------|---------|------------|----------|
| **Single Request** | Real-time inference | ⭐⭐⭐ Low | ⭐ Low | Interactive applications, real-time APIs |
| **Batch Optimized** | High-throughput processing | ⭐ Medium | ⭐⭐⭐ High | Batch processing, data pipelines |
| **Adaptive** | Mixed workloads | ⭐⭐ Variable | ⭐⭐ Variable | Production systems with varying load |

## Configuration Options ⚙️

### **Environment Variables**
```bash
# Enable/disable dynamic batching
ENABLE_DYNAMIC_BATCHING=true

# Choose batching strategy
BATCHING_STRATEGY=single_request    # single_request, batch_optimized, adaptive
```

### **Runtime Configuration**
```bash
# Get current batching stats
curl http://localhost:8000/batching/stats

# Configure batching settings
curl -X POST http://localhost:8000/batching/configure \
  -H "Content-Type: application/json" \
  -d '{
    "enable": true,
    "strategy": "batch_optimized",
    "max_batch_size": 16,
    "max_wait_time_ms": 100
  }'
```

## API Endpoints 📡

### **Single Request Endpoints** (Low Latency)
```bash
# Regular prediction - optimized for latency
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message", "models": ["bert-base"]}'

# Trained model prediction - optimized for latency
curl -X POST http://localhost:8000/predict/trained \
  -H "Content-Type: application/json" \
  -d '{"text": "Test message", "models": ["bert-base_trained"]}'
```

### **Batch Endpoints** (High Throughput)
```bash
# Batch prediction - optimized for throughput
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Message 1", "models": ["bert-base"]},
    {"text": "Message 2", "models": ["bert-base"]},
    {"text": "Message 3", "models": ["bert-base"]}
  ]'
```

### **Monitoring Endpoints**
```bash
# Get batching statistics
curl http://localhost:8000/batching/stats

# Configure batching settings
curl -X POST http://localhost:8000/batching/configure \
  -H "Content-Type: application/json" \
  -d '{"enable": true, "strategy": "adaptive"}'
```

## Batching Strategies 🎯

### **1. Single Request Strategy**
- **Purpose**: Optimize for low latency
- **Use Case**: Real-time applications, interactive APIs
- **Behavior**: Direct inference without batching
- **Latency**: 50-200ms typical
- **Throughput**: Lower but consistent

### **2. Batch Optimized Strategy**
- **Purpose**: Maximize throughput
- **Use Case**: Batch processing, data pipelines
- **Behavior**: Aggressive batching with configurable parameters
- **Latency**: 100-500ms typical
- **Throughput**: 3-10x higher than single requests

### **3. Adaptive Strategy**
- **Purpose**: Automatically choose optimal strategy
- **Use Case**: Production systems with varying load
- **Behavior**: Monitors system load and latency to choose strategy
- **Latency**: Variable based on conditions
- **Throughput**: Optimized for current conditions

## Configuration Parameters 🔧

### **Basic Batching Parameters**
```python
batch_config = BatchConfig(
    max_batch_size=8,        # Maximum requests per batch
    max_wait_time_ms=50,     # Max time to wait for batch formation
    min_batch_size=1,        # Minimum requests before processing
    max_queue_size=100       # Maximum pending requests
)
```

### **Smart Batching Parameters**
```python
smart_batch_config = SmartBatchConfig(
    max_batch_size=16,           # Larger batches for better GPU utilization
    max_wait_time_ms=100,        # Longer wait for better batching
    min_batch_size=2,            # Require at least 2 requests
    enable_adaptive=True,        # Enable adaptive strategy selection
    load_threshold=0.7,          # CPU/GPU utilization threshold
    latency_threshold_ms=200     # Max acceptable latency
)
```

## Performance Monitoring 📊

### **Key Metrics**
- **Throughput**: Requests per second (RPS)
- **Latency**: P50, P95, P99 response times
- **Batch Efficiency**: Average batch size, batch utilization
- **Queue Health**: Queue size, overflow rate
- **Resource Utilization**: CPU, GPU, memory usage

### **Monitoring Commands**
```bash
# Get comprehensive batching stats
curl http://localhost:8000/batching/stats | jq

# Monitor in real-time
watch -n 1 'curl -s http://localhost:8000/batching/stats | jq'
```

## Best Practices 🏆

### **For Low Latency Applications**
```bash
# Use single request strategy
export BATCHING_STRATEGY=single_request
export ENABLE_DYNAMIC_BATCHING=false

# Or use adaptive with low latency threshold
export BATCHING_STRATEGY=adaptive
export LATENCY_THRESHOLD_MS=100
```

### **For High Throughput Applications**
```bash
# Use batch optimized strategy
export BATCHING_STRATEGY=batch_optimized
export ENABLE_DYNAMIC_BATCHING=true

# Optimize batch parameters
export MAX_BATCH_SIZE=16
export MAX_WAIT_TIME_MS=100
```

### **For Production Systems**
```bash
# Use adaptive strategy with monitoring
export BATCHING_STRATEGY=adaptive
export ENABLE_DYNAMIC_BATCHING=true

# Enable comprehensive monitoring
export ENABLE_METRICS=true
export ENABLE_LOGGING=true
```

## Troubleshooting 🔧

### **Common Issues**

1. **High Latency with Batching**
   - Reduce `max_wait_time_ms`
   - Increase `min_batch_size`
   - Check system load

2. **Low Throughput**
   - Increase `max_batch_size`
   - Enable batch optimization
   - Check GPU utilization

3. **Queue Overflows**
   - Increase `max_queue_size`
   - Reduce `max_wait_time_ms`
   - Scale horizontally

### **Debug Commands**
```bash
# Check current configuration
curl http://localhost:8000/batching/stats

# Monitor system performance
curl http://localhost:8000/health

# Check model loading status
curl http://localhost:8000/models/status
```

## Implementation Details 🔍

### **Architecture**
```
Request → Strategy Selection → Batching Decision → Inference → Response
    ↓
Single Request: Direct Inference (Low Latency)
Batch Optimized: Queue → Batch Formation → Batch Inference (High Throughput)
Adaptive: Monitor → Choose Strategy → Execute
```

### **Key Components**
- **SmartBatcher**: Intelligent batching decisions
- **DynamicBatcher**: Traditional dynamic batching
- **StrategySelector**: Chooses optimal strategy
- **PerformanceMonitor**: Tracks metrics and adjusts

## Future Enhancements 🚀

- **GPU-aware batching**: Optimize batch sizes based on GPU memory
- **Model-specific tuning**: Different batching strategies per model
- **Load balancing**: Distribute batches across multiple workers
- **Predictive batching**: Use ML to predict optimal batch sizes
- **A/B testing**: Compare different batching strategies

## Conclusion ✅

Our dynamic batching system provides:

✅ **Flexible configuration** for different use cases
✅ **Intelligent strategy selection** based on system conditions  
✅ **Comprehensive monitoring** and observability
✅ **Runtime configuration** without restarts
✅ **Production-ready** with proper error handling

**The system is designed to be both powerful and easy to use, providing the right balance of performance and simplicity for your ML Security Platform.**
