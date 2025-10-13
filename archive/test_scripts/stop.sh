#!/bin/bash

# ML Security Service Stop Script

echo "🛑 Stopping ML Security Service"
echo "==============================="

# Stop services
docker-compose down

# Optional: Remove volumes (uncomment if you want to reset data)
# echo "🗑️  Removing volumes..."
# docker-compose down -v

echo "✅ ML Security Service stopped"
