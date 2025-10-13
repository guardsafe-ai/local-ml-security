#!/bin/bash

# ML Security Service Cleanup Script

echo "🧹 Cleaning up ML Security Service"
echo "=================================="

# Stop services
echo "Stopping services..."
docker-compose down

# Remove containers
echo "Removing containers..."
docker-compose rm -f

# Remove volumes (this will delete all data!)
read -p "⚠️  This will delete ALL data. Are you sure? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing volumes..."
    docker-compose down -v
    
    echo "Removing data directories..."
    rm -rf data/
    rm -rf mlflow/
    
    echo "Removing Docker images..."
    docker rmi $(docker images "local-ml-security*" -q) 2>/dev/null || true
fi

echo "✅ Cleanup completed"
