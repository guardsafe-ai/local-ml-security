FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/cache /app/models /app/logs

# Expose port
EXPOSE 8003

# Run the application
CMD ["python", "main.py"]
