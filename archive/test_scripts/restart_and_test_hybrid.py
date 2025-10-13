#!/usr/bin/env python3
"""
Restart services and test the hybrid model registry approach
"""

import subprocess
import time
import requests
import json

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def wait_for_service(url, service_name, max_attempts=30):
    """Wait for a service to be available"""
    print(f"⏳ Waiting for {service_name} to be ready...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print(f"❌ {service_name} failed to start within {max_attempts * 2} seconds")
    return False

def test_hybrid_approach():
    """Test the hybrid approach after restart"""
    print("\n🧪 Testing Hybrid Approach...")
    
    # Test training service
    try:
        response = requests.get("http://localhost:8002/models/latest", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Training service: {len(data.get('latest_models', {}))} models")
            print(f"   Source: {data.get('source', 'unknown')}")
        else:
            print(f"❌ Training service error: {response.status_code}")
    except Exception as e:
        print(f"❌ Training service test failed: {e}")
    
    # Test model API
    try:
        response = requests.get("http://localhost:8000/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', {})
            loaded_models = [name for name, info in models.items() if info.get('loaded', False)]
            print(f"✅ Model API: {len(loaded_models)} models loaded")
            
            # Show model sources
            for model_name in loaded_models:
                info = models[model_name]
                source = info.get('model_source', 'Unknown')
                version = info.get('model_version', 'Unknown')
                print(f"   🤖 {model_name}: {source} v{version}")
        else:
            print(f"❌ Model API error: {response.status_code}")
    except Exception as e:
        print(f"❌ Model API test failed: {e}")
    
    # Test prediction
    try:
        prediction_data = {
            "text": "This is a test prompt injection attempt",
            "models": ["distilbert_pretrained"],
            "ensemble": False
        }
        
        response = requests.post("http://localhost:8000/predict", json=prediction_data, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction test successful")
            
            for model_name, prediction in result.get('predictions', {}).items():
                print(f"   🎯 {model_name}: {prediction.get('prediction', 'unknown')} (confidence: {prediction.get('confidence', 0):.2f})")
        else:
            print(f"❌ Prediction test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction test error: {e}")

def main():
    """Main function"""
    print("🚀 Restarting Services and Testing Hybrid Approach")
    print("=" * 60)
    
    # Step 1: Stop services
    print("\n1️⃣ Stopping services...")
    run_command("docker-compose down", "Stopping all services")
    
    # Step 2: Rebuild services
    print("\n2️⃣ Rebuilding services...")
    run_command("docker-compose build --no-cache", "Rebuilding services")
    
    # Step 3: Start services
    print("\n3️⃣ Starting services...")
    run_command("docker-compose up -d", "Starting services")
    
    # Step 4: Wait for services to be ready
    print("\n4️⃣ Waiting for services to be ready...")
    
    services = [
        ("http://localhost:8000/health", "Model API"),
        ("http://localhost:8002/health", "Training Service"),
        ("http://localhost:8501", "Monitoring UI"),
        ("http://localhost:5000", "MLflow UI")
    ]
    
    all_ready = True
    for url, name in services:
        if not wait_for_service(url, name):
            all_ready = False
    
    if not all_ready:
        print("\n❌ Some services failed to start. Check logs with: docker-compose logs")
        return
    
    # Step 5: Test hybrid approach
    print("\n5️⃣ Testing hybrid approach...")
    test_hybrid_approach()
    
    print("\n" + "=" * 60)
    print("🎯 Hybrid Approach Implementation Complete!")
    print("\n📋 What was implemented:")
    print("   ✅ Primary Storage: MLflow Model Registry")
    print("   ✅ Cache Layer: Redis (1 hour TTL)")
    print("   ✅ Backup: Local file storage")
    print("   ✅ Fallback: Direct MLflow queries")
    print("\n🔗 Access URLs:")
    print("   🌐 Monitoring UI: http://localhost:8501")
    print("   🔬 MLflow UI: http://localhost:5000")
    print("   📊 Model API: http://localhost:8000/docs")
    print("   🏋️ Training API: http://localhost:8002/docs")
    print("\n🧪 Run detailed test: python test_hybrid_approach.py")

if __name__ == "__main__":
    main()
