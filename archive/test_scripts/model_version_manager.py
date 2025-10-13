#!/usr/bin/env python3
"""
Model Version Management Script
Helps manage and verify model versions in the ML Security Service
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional

class ModelVersionManager:
    def __init__(self, training_url: str = "http://localhost:8002", model_api_url: str = "http://localhost:8000"):
        self.training_url = training_url
        self.model_api_url = model_api_url
    
    def get_latest_models(self) -> Dict:
        """Get latest version of all models"""
        try:
            response = requests.get(f"{self.training_url}/models/latest", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_best_models(self) -> Dict:
        """Get best performing models"""
        try:
            response = requests.get(f"{self.training_url}/models/best", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_versions(self, model_name: str) -> Dict:
        """Get all versions of a specific model"""
        try:
            response = requests.get(f"{self.training_url}/models/versions/{model_name}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_model_api_status(self) -> Dict:
        """Get model API status and loaded models"""
        try:
            response = requests.get(f"{self.model_api_url}/models", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def verify_latest_usage(self) -> Dict:
        """Verify that model API is using latest trained models"""
        print("🔍 Verifying Latest Model Usage...")
        
        # Get latest models from training service
        latest_models = self.get_latest_models()
        if "error" in latest_models:
            return {"error": f"Failed to get latest models: {latest_models['error']}"}
        
        # Get model API status
        model_api_status = self.get_model_api_status()
        if "error" in model_api_status:
            return {"error": f"Failed to get model API status: {model_api_status['error']}"}
        
        verification_results = {}
        
        for model_name, latest_info in latest_models.get("latest_models", {}).items():
            print(f"\n📊 Checking {model_name}...")
            
            if model_name in model_api_status.get("models", {}):
                api_model = model_api_status["models"][model_name]
                
                verification_results[model_name] = {
                    "latest_version": latest_info.get("version"),
                    "api_version": api_model.get("model_version"),
                    "api_source": api_model.get("model_source"),
                    "api_path": api_model.get("path"),
                    "is_latest": latest_info.get("version") == api_model.get("model_version"),
                    "is_trained": api_model.get("model_source") == "MLflow/MinIO"
                }
                
                print(f"  ✅ Latest Version: {latest_info.get('version')}")
                print(f"  📍 API Version: {api_model.get('model_version')}")
                print(f"  🏷️ API Source: {api_model.get('model_source')}")
                print(f"  📂 API Path: {api_model.get('path')}")
                print(f"  🎯 Using Latest: {'✅ Yes' if verification_results[model_name]['is_latest'] else '❌ No'}")
                print(f"  🏋️ Using Trained: {'✅ Yes' if verification_results[model_name]['is_trained'] else '❌ No'}")
            else:
                verification_results[model_name] = {
                    "error": "Model not found in API"
                }
                print(f"  ❌ Model not found in API")
        
        return verification_results
    
    def print_model_summary(self):
        """Print a summary of all model versions"""
        print("=" * 80)
        print("📊 MODEL VERSION SUMMARY")
        print("=" * 80)
        
        # Latest models
        print("\n🆕 LATEST MODELS:")
        latest = self.get_latest_models()
        if "latest_models" in latest:
            for model_name, info in latest["latest_models"].items():
                print(f"  {model_name}:")
                print(f"    Version: {info.get('version')}")
                print(f"    Timestamp: {info.get('timestamp')}")
                print(f"    F1 Score: {info.get('f1_score', 'N/A')}")
                print(f"    MLflow URI: {info.get('mlflow_uri')}")
        
        # Best models
        print("\n🏆 BEST MODELS:")
        best = self.get_best_models()
        if "best_models" in best:
            for model_name, info in best["best_models"].items():
                print(f"  {model_name}:")
                print(f"    Version: {info.get('version')}")
                print(f"    F1 Score: {info.get('f1_score', 'N/A')}")
                print(f"    Timestamp: {info.get('timestamp')}")
        
        # Model API status
        print("\n🤖 MODEL API STATUS:")
        api_status = self.get_model_api_status()
        if "models" in api_status:
            for model_name, info in api_status["models"].items():
                print(f"  {model_name}:")
                print(f"    Loaded: {'✅ Yes' if info.get('loaded') else '❌ No'}")
                print(f"    Source: {info.get('model_source', 'Unknown')}")
                print(f"    Version: {info.get('model_version', 'Unknown')}")
                print(f"    Path: {info.get('path', 'Unknown')}")

def main():
    manager = ModelVersionManager()
    
    print("🚀 ML Security Service - Model Version Manager")
    print("=" * 60)
    
    # Print summary
    manager.print_model_summary()
    
    # Verify latest usage
    print("\n" + "=" * 60)
    verification = manager.verify_latest_usage()
    
    print("\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY:")
    print("=" * 60)
    
    all_using_latest = True
    all_using_trained = True
    
    for model_name, result in verification.items():
        if "error" not in result:
            using_latest = result.get("is_latest", False)
            using_trained = result.get("is_trained", False)
            
            status = "✅" if using_latest and using_trained else "❌"
            print(f"{status} {model_name}: Latest={using_latest}, Trained={using_trained}")
            
            if not using_latest or not using_trained:
                all_using_latest = False
                all_using_trained = False
        else:
            print(f"❌ {model_name}: {result['error']}")
            all_using_latest = False
            all_using_trained = False
    
    print("\n" + "=" * 60)
    if all_using_latest and all_using_trained:
        print("🎉 ALL MODELS ARE USING LATEST TRAINED VERSIONS!")
    else:
        print("⚠️  SOME MODELS ARE NOT USING LATEST TRAINED VERSIONS!")
    print("=" * 60)

if __name__ == "__main__":
    main()
