#!/usr/bin/env python3
"""
Test script for the comprehensive dashboard architecture
Tests all core service endpoints to ensure they're accessible
"""

import asyncio
import httpx
import json
from datetime import datetime

# Service URLs
SERVICES = {
    "model_api": "http://localhost:8000",
    "training": "http://localhost:8002", 
    "model_cache": "http://localhost:8003",
    "business_metrics": "http://localhost:8004",
    "analytics": "http://localhost:8006",
    "data_privacy": "http://localhost:8008",
    "tracing": "http://localhost:8009",
    "mlflow": "http://localhost:5000",
    "enterprise_dashboard": "http://localhost:8007"
}

async def test_service_endpoints():
    """Test all core service endpoints"""
    print("🚀 Testing Comprehensive Dashboard Architecture")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = {}
        
        # Test each service
        for service_name, service_url in SERVICES.items():
            print(f"\n📡 Testing {service_name.upper()} ({service_url})")
            print("-" * 40)
            
            service_results = {}
            
            try:
                # Test health endpoint
                health_response = await client.get(f"{service_url}/health")
                
                # Handle both JSON and plain text responses
                try:
                    health_data = health_response.json() if health_response.status_code == 200 else None
                except (json.JSONDecodeError, ValueError):
                    # Handle plain text responses (like MLflow's "OK")
                    health_data = health_response.text if health_response.status_code == 200 else None
                
                service_results["health"] = {
                    "status": health_response.status_code,
                    "response_time": health_response.elapsed.total_seconds() * 1000,
                    "data": health_data
                }
                print(f"✅ Health: {health_response.status_code} ({service_results['health']['response_time']:.1f}ms)")
                
            except Exception as e:
                service_results["health"] = {"error": str(e)}
                print(f"❌ Health: {e}")
            
            # Test specific endpoints based on service
            if service_name == "model_api":
                try:
                    models_response = await client.get(f"{service_url}/models")
                    service_results["models"] = {
                        "status": models_response.status_code,
                        "data": models_response.json() if models_response.status_code == 200 else None
                    }
                    print(f"✅ Models: {models_response.status_code}")
                except Exception as e:
                    service_results["models"] = {"error": str(e)}
                    print(f"❌ Models: {e}")
                
                try:
                    predict_response = await client.post(f"{service_url}/predict", json={
                        "text": "This is a test message",
                        "models": ["distilbert"]
                    })
                    service_results["predict"] = {
                        "status": predict_response.status_code,
                        "data": predict_response.json() if predict_response.status_code == 200 else None
                    }
                    print(f"✅ Predict: {predict_response.status_code}")
                except Exception as e:
                    service_results["predict"] = {"error": str(e)}
                    print(f"❌ Predict: {e}")
            
            elif service_name == "training":
                try:
                    jobs_response = await client.get(f"{service_url}/jobs")
                    service_results["jobs"] = {
                        "status": jobs_response.status_code,
                        "data": jobs_response.json() if jobs_response.status_code == 200 else None
                    }
                    print(f"✅ Jobs: {jobs_response.status_code}")
                except Exception as e:
                    service_results["jobs"] = {"error": str(e)}
                    print(f"❌ Jobs: {e}")
            
            elif service_name == "analytics":
                try:
                    drift_response = await client.get(f"{service_url}/drift/history")
                    service_results["drift"] = {
                        "status": drift_response.status_code,
                        "data": drift_response.json() if drift_response.status_code == 200 else None
                    }
                    print(f"✅ Drift: {drift_response.status_code}")
                except Exception as e:
                    service_results["drift"] = {"error": str(e)}
                    print(f"❌ Drift: {e}")
            
            elif service_name == "business_metrics":
                try:
                    metrics_response = await client.get(f"{service_url}/metrics")
                    service_results["metrics"] = {
                        "status": metrics_response.status_code,
                        "data": metrics_response.json() if metrics_response.status_code == 200 else None
                    }
                    print(f"✅ Metrics: {metrics_response.status_code}")
                except Exception as e:
                    service_results["metrics"] = {"error": str(e)}
                    print(f"❌ Metrics: {e}")
            
            elif service_name == "data_privacy":
                try:
                    privacy_response = await client.get(f"{service_url}/health")
                    service_results["privacy"] = {
                        "status": privacy_response.status_code,
                        "data": privacy_response.json() if privacy_response.status_code == 200 else None
                    }
                    print(f"✅ Privacy: {privacy_response.status_code}")
                except Exception as e:
                    service_results["privacy"] = {"error": str(e)}
                    print(f"❌ Privacy: {e}")
            
            elif service_name == "mlflow":
                try:
                    experiments_response = await client.get(f"{service_url}/api/2.0/mlflow/experiments/list")
                    service_results["experiments"] = {
                        "status": experiments_response.status_code,
                        "data": experiments_response.json() if experiments_response.status_code == 200 else None
                    }
                    print(f"✅ Experiments: {experiments_response.status_code}")
                except Exception as e:
                    service_results["experiments"] = {"error": str(e)}
                    print(f"❌ Experiments: {e}")
            
            results[service_name] = service_results
        
        # Test Enterprise Dashboard endpoints
        print(f"\n📡 Testing ENTERPRISE DASHBOARD ({SERVICES['enterprise_dashboard']})")
        print("-" * 40)
        
        try:
            dashboard_response = await client.get(f"{SERVICES['enterprise_dashboard']}/dashboard/metrics")
            print(f"✅ Dashboard Metrics: {dashboard_response.status_code}")
            if dashboard_response.status_code == 200:
                metrics = dashboard_response.json()
                print(f"   📊 Total Models: {metrics.get('total_models', 0)}")
                print(f"   📊 Active Jobs: {metrics.get('active_jobs', 0)}")
                print(f"   📊 System Health: {metrics.get('system_health', 0)}%")
        except Exception as e:
            print(f"❌ Dashboard Metrics: {e}")
        
        try:
            services_response = await client.get(f"{SERVICES['enterprise_dashboard']}/services/health")
            print(f"✅ Services Health: {services_response.status_code}")
            if services_response.status_code == 200:
                health_data = services_response.json()
                healthy_count = sum(1 for service in health_data if service.get('status') == 'healthy')
                print(f"   🏥 Healthy Services: {healthy_count}/{len(health_data)}")
        except Exception as e:
            print(f"❌ Services Health: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_services = len(SERVICES)
        healthy_services = 0
        
        for service_name, service_results in results.items():
            if service_results.get("health", {}).get("status") == 200:
                healthy_services += 1
                print(f"✅ {service_name.upper()}: HEALTHY")
            else:
                print(f"❌ {service_name.upper()}: UNHEALTHY")
        
        health_percentage = (healthy_services / total_services) * 100
        print(f"\n🏥 Overall Health: {healthy_services}/{total_services} ({health_percentage:.1f}%)")
        
        if health_percentage >= 80:
            print("🎉 EXCELLENT! System is running well")
        elif health_percentage >= 60:
            print("⚠️  GOOD: Some services need attention")
        else:
            print("🚨 CRITICAL: Multiple services are down")
        
        print(f"\n⏰ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(test_service_endpoints())

