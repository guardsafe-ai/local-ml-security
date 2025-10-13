#!/usr/bin/env python3
"""
Script to check what models are stored in MinIO and MLflow
"""

import os
import sys
import requests
import json
from datetime import datetime

def check_minio_buckets():
    """Check MinIO buckets and contents"""
    print("🔍 Checking MinIO buckets...")
    
    try:
        # Check if MinIO is accessible
        response = requests.get("http://localhost:9001", timeout=5)
        if response.status_code == 200:
            print("✅ MinIO is accessible at http://localhost:9001")
            print("   Login: minioadmin / minioadmin")
        else:
            print("❌ MinIO not accessible")
            return
    except Exception as e:
        print(f"❌ Cannot connect to MinIO: {e}")
        return
    
    # You can also check via MinIO API if needed
    print("\n📦 To check MinIO contents manually:")
    print("   1. Go to http://localhost:9001")
    print("   2. Login with minioadmin/minioadmin")
    print("   3. Look for 'mlflow-artifacts' bucket")

def check_mlflow_models():
    """Check MLflow for registered models"""
    print("\n🔍 Checking MLflow models...")
    
    try:
        # Check MLflow UI
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow UI is accessible at http://localhost:5000")
        else:
            print("❌ MLflow UI not accessible")
            return
    except Exception as e:
        print(f"❌ Cannot connect to MLflow: {e}")
        return
    
    # Check via MLflow API
    try:
        response = requests.get("http://localhost:5000/api/2.0/mlflow/registered-models/search", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('registered_models', [])
            if models:
                print(f"✅ Found {len(models)} registered models in MLflow:")
                for model in models:
                    name = model.get('name', 'Unknown')
                    versions = model.get('latest_versions', [])
                    print(f"   - {name} ({len(versions)} versions)")
            else:
                print("❌ No registered models found in MLflow")
        else:
            print(f"❌ MLflow API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking MLflow API: {e}")

def check_model_api():
    """Check Model API for available models"""
    print("\n🔍 Checking Model API...")
    
    try:
        response = requests.get("http://localhost:8000/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model API response:")
            print(f"   Available models: {data.get('available_models', [])}")
            print(f"   MLflow models: {data.get('mlflow_models', [])}")
            
            models = data.get('models', {})
            for name, info in models.items():
                if isinstance(info, dict) and 'loaded' in info:
                    status = "✅ Loaded" if info['loaded'] else "❌ Not loaded"
                    print(f"   - {name}: {status}")
        else:
            print(f"❌ Model API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking Model API: {e}")

def check_training_service():
    """Check Training service status"""
    print("\n🔍 Checking Training service...")
    
    try:
        response = requests.get("http://localhost:8002/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Training service response:")
            print(f"   Available models: {data.get('models', [])}")
        else:
            print(f"❌ Training service error: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking Training service: {e}")

def main():
    print("🔒 ML Security Service - Model Storage Check")
    print("=" * 50)
    
    check_minio_buckets()
    check_mlflow_models()
    check_model_api()
    check_training_service()
    
    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("1. If no models are in MinIO, you need to train a model first")
    print("2. Go to http://localhost:8501 and use the Training tab")
    print("3. Or use: curl -X POST 'http://localhost:8002/train' -H 'Content-Type: application/json' -d '{\"model_name\": \"distilbert\"}'")
    print("4. After training, models should appear in MinIO and MLflow")

if __name__ == "__main__":
    main()
