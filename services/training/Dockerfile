# Multi-architecture Dockerfile with CUDA support and CPU fallback
FROM python:3.9-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with automatic CUDA detection
# This will install CUDA version if available, CPU version otherwise
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/training_data /app/logs

# Expose port
EXPOSE 8002

# Start the service directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
