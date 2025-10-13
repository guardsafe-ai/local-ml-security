#!/bin/bash

# MinIO Setup Script
# This script sets up MinIO with proper permissions and creates necessary buckets

echo "🚀 Setting up MinIO with proper permissions and buckets..."

# MinIO configuration
MINIO_ENDPOINT="http://localhost:9000"
MINIO_CONSOLE="http://localhost:9001"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"

# Install MinIO client if not already installed
if ! command -v mc &> /dev/null; then
    echo "📦 Installing MinIO client..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install minio/stable/mc
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        wget https://dl.min.io/client/mc/release/linux-amd64/mc
        chmod +x mc
        sudo mv mc /usr/local/bin/
    else
        echo "❌ Unsupported OS. Please install MinIO client manually."
        exit 1
    fi
fi

# Configure MinIO client
echo "🔧 Configuring MinIO client..."
mc alias set local $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY

# Create buckets
echo "📁 Creating buckets..."

# Create mlflow-artifacts bucket
echo "Creating mlflow-artifacts bucket..."
mc mb local/mlflow-artifacts --ignore-existing

# Create ml-security bucket
echo "Creating ml-security bucket..."
mc mb local/ml-security --ignore-existing

# Create additional buckets for different services
echo "Creating additional service buckets..."
mc mb local/training-data --ignore-existing
mc mb local/model-cache --ignore-existing
mc mb local/red-team-data --ignore-existing
mc mb local/analytics-data --ignore-existing
mc mb local/business-metrics --ignore-existing
mc mb local/privacy-data --ignore-existing

# Set bucket policies for public read access (for applications)
echo "🔐 Setting bucket policies..."

# Set policy for mlflow-artifacts (read/write for applications)
mc anonymous set public local/mlflow-artifacts

# Set policy for ml-security (read/write for applications)
mc anonymous set public local/ml-security

# Set policy for other buckets
mc anonymous set public local/training-data
mc anonymous set public local/model-cache
mc anonymous set public local/red-team-data
mc anonymous set public local/analytics-data
mc anonymous set public local/business-metrics
mc anonymous set public local/privacy-data

# Create a user for applications with specific permissions
echo "👤 Creating application user..."
mc admin user add local appuser apppassword

# Create policy for application user
cat > /tmp/app-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::*"
            ]
        }
    ]
}
EOF

# Apply policy to application user
mc admin policy add local app-policy /tmp/app-policy.json
mc admin policy set local app-policy user=appuser

# Clean up
rm -f /tmp/app-policy.json

# List all buckets
echo "📋 Listing all buckets:"
mc ls local

# Show bucket policies
echo "🔍 Checking bucket policies:"
for bucket in mlflow-artifacts ml-security training-data model-cache red-team-data analytics-data business-metrics privacy-data; do
    echo "Policy for $bucket:"
    mc anonymous get local/$bucket
    echo "---"
done

echo "✅ MinIO setup complete!"
echo ""
echo "🌐 Access URLs:"
echo "   MinIO Console: $MINIO_CONSOLE"
echo "   MinIO API: $MINIO_ENDPOINT"
echo ""
echo "🔑 Credentials:"
echo "   Admin: $MINIO_ACCESS_KEY / $MINIO_SECRET_KEY"
echo "   App User: appuser / apppassword"
echo ""
echo "📁 Available buckets:"
echo "   - mlflow-artifacts (MLflow experiment artifacts)"
echo "   - ml-security (General ML security data)"
echo "   - training-data (Training datasets)"
echo "   - model-cache (Model cache files)"
echo "   - red-team-data (Red team testing data)"
echo "   - analytics-data (Analytics data)"
echo "   - business-metrics (Business metrics data)"
echo "   - privacy-data (Privacy compliance data)"
