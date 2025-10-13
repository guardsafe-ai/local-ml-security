#!/usr/bin/env python3
"""
Debug script for red team service
"""
import requests
import json

def test_red_team():
    """Test red team service with minimal data"""
    
    # Test 1: Check if red team service is healthy
    print("🔍 Testing red team service health...")
    try:
        response = requests.get("http://localhost:8001/health", timeout=10)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Health data: {response.json()}")
        else:
            print(f"Health error: {response.text}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    # Test 2: Check available models
    print("\n🔍 Testing available models...")
    try:
        response = requests.get("http://localhost:8001/models", timeout=10)
        print(f"Models check: {response.status_code}")
        if response.status_code == 200:
            print(f"Models data: {response.json()}")
        else:
            print(f"Models error: {response.text}")
    except Exception as e:
        print(f"Models check failed: {e}")
    
    # Test 3: Test model API directly
    print("\n🔍 Testing model API directly...")
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": "test", "model_name": "distilbert_pretrained"},
            timeout=10
        )
        print(f"Model API check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Model API data: {json.dumps(data, indent=2)}")
            print(f"Prediction type: {type(data.get('prediction', ''))}")
            print(f"Prediction value: {data.get('prediction', '')}")
        else:
            print(f"Model API error: {response.text}")
    except Exception as e:
        print(f"Model API check failed: {e}")
    
    # Test 4: Test red team with minimal data
    print("\n🔍 Testing red team with minimal data...")
    try:
        response = requests.post(
            "http://localhost:8001/test",
            json={"model_name": "distilbert_pretrained", "batch_size": 1},
            timeout=30
        )
        print(f"Red team test: {response.status_code}")
        if response.status_code == 200:
            print(f"Red team data: {response.json()}")
        else:
            print(f"Red team error: {response.text}")
    except Exception as e:
        print(f"Red team test failed: {e}")

if __name__ == "__main__":
    test_red_team()
