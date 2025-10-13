#!/bin/bash

# ML Security Service Startup Script

echo "🔒 Starting ML Security Service"
echo "==============================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

services=("mlflow:5000" "postgres:5432" "redis:6379" "minio:9000" "red-team:8001" "training:8002" "model-api:8000" "model-serving:8080" "model-cache:8003" "business-metrics:8004" "data-privacy:8005" "monitoring:8501")

for service in "${services[@]}"; do
    IFS=':' read -r host port <<< "$service"
    if nc -z $host $port 2>/dev/null; then
        echo "✅ $service is running"
    else
        echo "❌ $service is not responding"
    fi
done

echo ""
echo "🎉 ML Security Service is ready!"
echo ""
echo "Access points:"
echo "  📊 Dashboard:     http://localhost:8501"
echo "  🔬 MLflow UI:     http://localhost:5000"
echo "  📈 Grafana:       http://localhost:3000 (admin/admin)"
echo "  📊 Prometheus:    http://localhost:9090"
echo "  🗄️  MinIO:        http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "API Endpoints:"
echo "  🔴 Red Team:      http://localhost:8001"
echo "  🏋️  Training:     http://localhost:8002"
echo "  🤖 Model API:     http://localhost:8000"
echo "  🚀 Model Serving: http://localhost:8080"
echo "  🚀 Model Cache:   http://localhost:8003"
echo "  📊 Business KPIs: http://localhost:8004"
echo "  🔒 Data Privacy:  http://localhost:8005"
echo ""
echo "To stop the service: ./stop.sh"
