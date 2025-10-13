#!/bin/bash

echo "Starting Docker Desktop and ML Security Services..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Starting Docker Desktop..."
    open /Applications/Docker.app
    
    # Wait for Docker to start
    echo "Waiting for Docker to start..."
    while ! docker info > /dev/null 2>&1; do
        sleep 2
        echo -n "."
    done
    echo ""
    echo "Docker is now running!"
fi

# Start the services
echo "Starting ML Security services..."
docker-compose up -d postgres redis minio

# Wait for core services to be ready
echo "Waiting for core services to be ready..."
sleep 10

# Start the remaining services
echo "Starting remaining services..."
docker-compose up -d

echo "All services started! You can access:"
echo "- Enterprise Dashboard: http://localhost:3000"
echo "- MLflow: http://localhost:5000"
echo "- Grafana: http://localhost:3001 (admin/admin)"
echo "- Jaeger: http://localhost:16686"
echo "- MinIO: http://localhost:9001 (minioadmin/minioadmin)"
