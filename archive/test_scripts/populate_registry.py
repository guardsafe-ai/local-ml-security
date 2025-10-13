#!/usr/bin/env python3
"""
Script to populate the model registry with existing trained models from MLflow
"""
import requests
import json
from datetime import datetime

def populate_model_registry():
    """Populate model registry with existing MLflow models"""
    
    # Get registered models from MLflow
    mlflow_url = "http://localhost:5000/api/2.0/mlflow/registered-models/search"
    response = requests.get(mlflow_url)
    
    if response.status_code != 200:
        print(f"Error getting MLflow models: {response.status_code}")
        return
    
    mlflow_models = response.json()
    print(f"Found {len(mlflow_models['registered_models'])} registered models in MLflow")
    
    # Build model registry
    model_registry = {
        "latest": {},
        "best": {},
        "versions": {}
    }
    
    for model in mlflow_models['registered_models']:
        model_name = model['name']
        latest_version = model['latest_versions'][0]
        
        # Extract model name without 'security_' prefix
        base_name = model_name.replace('security_', '')
        
        # Get model info
        model_info = {
            "model_name": base_name,
            "version": latest_version['version'],
            "run_id": latest_version['run_id'],
            "f1_score": 0.0,  # We don't have this info from MLflow API
            "accuracy": 0.0,   # We don't have this info from MLflow API
            "timestamp": datetime.fromtimestamp(latest_version['creation_timestamp'] / 1000).isoformat(),
            "mlflow_uri": f"models:/{model_name}/latest"
        }
        
        # Add to registry
        model_registry["latest"][base_name] = model_info
        model_registry["best"][base_name] = model_info
        model_registry["versions"][base_name] = [model_info]
        
        print(f"Added {base_name} to registry: {model_info['mlflow_uri']}")
    
    # Send to training service
    training_url = "http://localhost:8002/models/registry"
    registry_data = {
        "model_registry": model_registry,
        "timestamp": datetime.now().isoformat()
    }
    
    response = requests.post(training_url, json=registry_data)
    if response.status_code == 200:
        print("✅ Model registry populated successfully!")
    else:
        print(f"❌ Error populating registry: {response.status_code} - {response.text}")

if __name__ == "__main__":
    populate_model_registry()
