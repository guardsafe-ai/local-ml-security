#!/bin/bash

# MinIO Advanced Policy Setup Script
# This script creates more granular policies for different services

echo "🔐 Setting up advanced MinIO policies..."

# MinIO configuration
MINIO_ENDPOINT="http://localhost:9000"
MINIO_ACCESS_KEY="minioadmin"
MINIO_SECRET_KEY="minioadmin"

# Configure MinIO client
mc alias set local $MINIO_ENDPOINT $MINIO_ACCESS_KEY $MINIO_SECRET_KEY

# Create policy for MLflow service
echo "📝 Creating MLflow policy..."
cat > /tmp/mlflow-policy.json << EOF
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
                "arn:aws:s3:::mlflow-artifacts",
                "arn:aws:s3:::mlflow-artifacts/*"
            ]
        }
    ]
}
EOF

# Create policy for Training service
echo "📝 Creating Training service policy..."
cat > /tmp/training-policy.json << EOF
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
                "arn:aws:s3:::training-data",
                "arn:aws:s3:::training-data/*",
                "arn:aws:s3:::ml-security",
                "arn:aws:s3:::ml-security/*"
            ]
        }
    ]
}
EOF

# Create policy for Model API service
echo "📝 Creating Model API policy..."
cat > /tmp/model-api-policy.json << EOF
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
                "arn:aws:s3:::model-cache",
                "arn:aws:s3:::model-cache/*",
                "arn:aws:s3:::mlflow-artifacts",
                "arn:aws:s3:::mlflow-artifacts/*"
            ]
        }
    ]
}
EOF

# Create policy for Red Team service
echo "📝 Creating Red Team policy..."
cat > /tmp/red-team-policy.json << EOF
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
                "arn:aws:s3:::red-team-data",
                "arn:aws:s3:::red-team-data/*",
                "arn:aws:s3:::ml-security",
                "arn:aws:s3:::ml-security/*"
            ]
        }
    ]
}
EOF

# Create policy for Analytics service
echo "📝 Creating Analytics policy..."
cat > /tmp/analytics-policy.json << EOF
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
                "arn:aws:s3:::analytics-data",
                "arn:aws:s3:::analytics-data/*",
                "arn:aws:s3:::ml-security",
                "arn:aws:s3:::ml-security/*"
            ]
        }
    ]
}
EOF

# Create policy for Business Metrics service
echo "📝 Creating Business Metrics policy..."
cat > /tmp/business-metrics-policy.json << EOF
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
                "arn:aws:s3:::business-metrics",
                "arn:aws:s3:::business-metrics/*",
                "arn:aws:s3:::ml-security",
                "arn:aws:s3:::ml-security/*"
            ]
        }
    ]
}
EOF

# Create policy for Data Privacy service
echo "📝 Creating Data Privacy policy..."
cat > /tmp/privacy-policy.json << EOF
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
                "arn:aws:s3:::privacy-data",
                "arn:aws:s3:::privacy-data/*",
                "arn:aws:s3:::ml-security",
                "arn:aws:s3:::ml-security/*"
            ]
        }
    ]
}
EOF

# Create policy for read-only access (for monitoring/dashboard)
echo "📝 Creating Read-Only policy..."
cat > /tmp/readonly-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::*"
            ]
        }
    ]
}
EOF

# Apply all policies
echo "🔧 Applying policies..."

mc admin policy create local mlflow-policy /tmp/mlflow-policy.json
mc admin policy create local training-policy /tmp/training-policy.json
mc admin policy create local model-api-policy /tmp/model-api-policy.json
mc admin policy create local red-team-policy /tmp/red-team-policy.json
mc admin policy create local analytics-policy /tmp/analytics-policy.json
mc admin policy create local business-metrics-policy /tmp/business-metrics-policy.json
mc admin policy create local privacy-policy /tmp/privacy-policy.json
mc admin policy create local readonly-policy /tmp/readonly-policy.json

# Create service users and assign policies
echo "👥 Creating service users..."

# MLflow user
mc admin user add local mlflow-user mlflow-password
mc admin policy attach local mlflow-policy --user mlflow-user

# Training user
mc admin user add local training-user training-password
mc admin policy attach local training-policy --user training-user

# Model API user
mc admin user add local model-api-user model-api-password
mc admin policy attach local model-api-policy --user model-api-user

# Red Team user
mc admin user add local red-team-user red-team-password
mc admin policy attach local red-team-policy --user red-team-user

# Analytics user
mc admin user add local analytics-user analytics-password
mc admin policy attach local analytics-policy --user analytics-user

# Business Metrics user
mc admin user add local business-metrics-user business-metrics-password
mc admin policy attach local business-metrics-policy --user business-metrics-user

# Privacy user
mc admin user add local privacy-user privacy-password
mc admin policy attach local privacy-policy --user privacy-user

# Dashboard user (read-only)
mc admin user add local dashboard-user dashboard-password
mc admin policy attach local readonly-policy --user dashboard-user

# Clean up
rm -f /tmp/*-policy.json

echo "✅ Advanced MinIO policies setup complete!"
echo ""
echo "👥 Service Users Created:"
echo "   - mlflow-user / mlflow-password (MLflow artifacts)"
echo "   - training-user / training-password (Training data)"
echo "   - model-api-user / model-api-password (Model cache)"
echo "   - red-team-user / red-team-password (Red team data)"
echo "   - analytics-user / analytics-password (Analytics data)"
echo "   - business-metrics-user / business-metrics-password (Business metrics)"
echo "   - privacy-user / privacy-password (Privacy data)"
echo "   - dashboard-user / dashboard-password (Read-only access)"
echo ""
echo "🔐 Each user has access only to their specific buckets and ml-security bucket"
echo "📊 Dashboard user has read-only access to all buckets for monitoring"
