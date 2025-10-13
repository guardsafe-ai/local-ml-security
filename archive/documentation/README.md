# Local ML Security Service

A complete local machine learning security service for prompt injection and jailbreak detection with continuous learning capabilities.

## Features

- 🔒 **Multi-Model Security Detection**: DeBERTa, RoBERTa, and specialized security models
- 🤖 **Continuous Learning**: Automated red-teaming and model updates
- 📊 **Real-time Monitoring**: Comprehensive dashboard and alerting
- 🚀 **Local Deployment**: No cloud dependencies, runs entirely on your machine
- 🔄 **Model Versioning**: MLflow integration for experiment tracking
- 📈 **Performance Analytics**: Detailed metrics and reporting

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo>
cd local-ml-security
chmod +x setup.sh
./setup.sh

# 2. Start services
docker-compose up -d

# 3. Access dashboard
open http://localhost:8501
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Red Team      │    │   Model Training │    │   Model Serving │
│   Service       │───▶│   Pipeline       │───▶│   Infrastructure│
│   (Docker)      │    │   (MLflow)       │    │   (k3s)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Attack        │    │   Model          │    │   Load Balancer │
│   Generator     │    │   Registry       │    │   + Monitoring  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Services

- **Red Team Service**: Generates attack patterns and tests models
- **Model Training**: Automated training pipeline with MLflow
- **Model Serving**: Kubernetes-based model serving
- **Monitoring**: Real-time dashboard and alerting
- **Storage**: MinIO for model artifacts, PostgreSQL for metadata

## Requirements

- Docker & Docker Compose
- Python 3.9+
- 8GB+ RAM
- 50GB+ free disk space
