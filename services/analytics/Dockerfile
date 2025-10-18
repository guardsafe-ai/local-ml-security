FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Copy environment file if it exists
COPY .env* ./

# Expose port
EXPOSE 8006

# Set default environment variables for email notifications
ENV EMAIL_NOTIFICATIONS_ENABLED=true
ENV SMTP_SERVER=smtp.gmail.com
ENV SMTP_PORT=587
ENV SENDER_EMAIL=noreply@guardsafe-ai.com
ENV SENDER_PASSWORD=dummy_password
ENV RECIPIENT_EMAILS=admin@guardsafe-ai.com

# Run the application
CMD ["python", "main.py"]
