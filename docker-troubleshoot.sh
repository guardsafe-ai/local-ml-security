#!/bin/bash

echo "=== Docker Troubleshooting Script ==="
echo ""

# Check Docker installation
echo "1. Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed: $(docker --version)"
else
    echo "✗ Docker is not installed"
    exit 1
fi

# Check Docker Desktop
echo ""
echo "2. Checking Docker Desktop..."
if [ -d "/Applications/Docker.app" ]; then
    echo "✓ Docker Desktop is installed"
else
    echo "✗ Docker Desktop is not installed"
    exit 1
fi

# Check Docker daemon
echo ""
echo "3. Checking Docker daemon..."
if docker info > /dev/null 2>&1; then
    echo "✓ Docker daemon is running"
    docker system info | grep -E "(Server Version|Storage Driver|Logging Driver)"
else
    echo "✗ Docker daemon is not running"
    echo "Trying to start Docker Desktop..."
    open /Applications/Docker.app
    echo "Please wait for Docker Desktop to start and run this script again"
    exit 1
fi

# Check Docker Compose
echo ""
echo "4. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    echo "✓ Docker Compose is available: $(docker-compose --version)"
else
    echo "✗ Docker Compose is not available"
fi

# Check available resources
echo ""
echo "5. Checking system resources..."
echo "Available memory: $(sysctl -n hw.memsize | awk '{print $0/1024/1024/1024 " GB"}')"
echo "Available disk space: $(df -h / | tail -1 | awk '{print $4}')"

# Check for port conflicts
echo ""
echo "6. Checking for port conflicts..."
ports=(3000 5000 8000 8001 8002 8003 8004 8005 8006 8007 8008 9000 9001 9090 16686)
for port in "${ports[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠ Port $port is in use: $(lsof -i :$port | tail -1 | awk '{print $1, $2}')"
    else
        echo "✓ Port $port is available"
    fi
done

# Check Docker Compose configuration
echo ""
echo "7. Checking Docker Compose configuration..."
if docker-compose config > /dev/null 2>&1; then
    echo "✓ Docker Compose configuration is valid"
else
    echo "✗ Docker Compose configuration has errors:"
    docker-compose config
fi

echo ""
echo "=== Troubleshooting Complete ==="
