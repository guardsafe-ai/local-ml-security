#!/usr/bin/env python3
"""
Simple script to load pre-trained models for the ML security system
"""

import os
import sys
import requests
import json
from pathlib import Path

def load_model_via_api(model_name: str):
    """Load a model via the model-api service"""
    try:
        # First, let's try to load a pre-trained model from Hugging Face
        print(f"🔄 Loading {model_name} model...")
        
        # Check if model-api is running
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ Model API service is not running")
            return False
        
        # Try to load the model
        load_data = {
            "model_name": model_name,
            "model_path": f"microsoft/{model_name}",  # Use Hugging Face model
            "model_type": "pytorch"
        }
        
        response = requests.post(
            "http://localhost:8000/models/load",
            json=load_data
        )
        
        if response.status_code == 200:
            print(f"✅ Successfully loaded {model_name}")
            return True
        else:
            print(f"❌ Failed to load {model_name}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return False

def check_available_models():
    """Check what models are available"""
    try:
        response = requests.get("http://localhost:8000/models")
        if response.status_code == 200:
            data = response.json()
            print("📊 Available Models:")
            for model_name, info in data.get("models", {}).items():
                status = "✅ Loaded" if info.get("loaded") else "❌ Not Loaded"
                print(f"  - {model_name}: {status}")
        else:
            print("❌ Could not fetch model information")
    except Exception as e:
        print(f"❌ Error checking models: {e}")

def main():
    """Main function"""
    print("🤖 ML Security Model Loader")
    print("=" * 40)
    
    # Check current model status
    check_available_models()
    
    print("\n🔄 Attempting to load models...")
    
    # Try to load different models
    models_to_load = [
        "deberta-v3-large",
        "roberta-large", 
        "bert-base-uncased",
        "distilbert-base-uncased"
    ]
    
    success_count = 0
    for model in models_to_load:
        if load_model_via_api(model):
            success_count += 1
        print()
    
    print(f"📊 Summary: {success_count}/{len(models_to_load)} models loaded successfully")
    
    # Check final status
    print("\n📊 Final Model Status:")
    check_available_models()

if __name__ == "__main__":
    main()
