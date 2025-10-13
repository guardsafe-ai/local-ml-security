#!/usr/bin/env python3
"""
Real-time monitoring script to detect flickering and infinite loading issues
"""

import requests
import time
import json
import subprocess
import threading
from datetime import datetime
from collections import defaultdict, deque

class FlickeringMonitor:
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.request_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.loading_states = defaultdict(list)
        self.last_check = time.time()
        self.monitoring = True
        
    def log_with_timestamp(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {level}: {message}")
    
    def check_frontend_requests(self):
        """Monitor frontend request patterns"""
        try:
            # Check if frontend is making excessive requests
            response = requests.get("http://localhost:3000", timeout=5)
            if response.status_code == 200:
                self.log_with_timestamp("Frontend accessible")
            else:
                self.log_with_timestamp(f"Frontend error: {response.status_code}", "ERROR")
        except Exception as e:
            self.log_with_timestamp(f"Frontend error: {str(e)}", "ERROR")
    
    def check_backend_requests(self):
        """Monitor backend request patterns and response times"""
        endpoints = [
            "/health",
            "/dashboard/metrics", 
            "/services/health",
            "/dashboard/red-team/overview",
            "/dashboard/training/overview",
            "/models/available"
        ]
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"http://localhost:8007{endpoint}", timeout=5)
                response_time = (time.time() - start_time) * 1000
                
                self.request_counts[endpoint] += 1
                self.request_times[endpoint].append(response_time)
                
                if response.status_code != 200:
                    self.error_counts[endpoint] += 1
                    self.log_with_timestamp(f"Backend error {endpoint}: {response.status_code}", "ERROR")
                else:
                    self.log_with_timestamp(f"Backend {endpoint}: {response_time:.1f}ms")
                    
            except Exception as e:
                self.error_counts[endpoint] += 1
                self.log_with_timestamp(f"Backend error {endpoint}: {str(e)}", "ERROR")
    
    def check_service_health(self):
        """Check all service health status"""
        services = [
            ("training", "http://localhost:8002/health"),
            ("red-team", "http://localhost:8001/health"),
            ("model-api", "http://localhost:8000/health"),
            ("analytics", "http://localhost:8006/health"),
            ("business-metrics", "http://localhost:8004/health"),
            ("data-privacy", "http://localhost:8008/health"),
            ("model-cache", "http://localhost:8003/health")
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    self.log_with_timestamp(f"Service {service_name}: Healthy")
                else:
                    self.log_with_timestamp(f"Service {service_name}: Error {response.status_code}", "WARN")
            except Exception as e:
                self.log_with_timestamp(f"Service {service_name}: Error - {str(e)}", "WARN")
    
    def analyze_patterns(self):
        """Analyze request patterns for flickering indicators"""
        current_time = time.time()
        time_diff = current_time - self.last_check
        
        # Check for excessive requests
        total_requests = sum(self.request_counts.values())
        if total_requests > 0:
            requests_per_second = total_requests / time_diff if time_diff > 0 else 0
            
            if requests_per_second > 2:  # More than 2 requests per second
                self.log_with_timestamp(f"⚠️  HIGH REQUEST RATE: {requests_per_second:.1f} req/sec", "WARN")
                
                # Identify problematic endpoints
                for endpoint, count in self.request_counts.items():
                    if count > 5:  # More than 5 requests to same endpoint
                        self.log_with_timestamp(f"   🔥 {endpoint}: {count} requests", "WARN")
        
        # Check for slow responses
        for endpoint, times in self.request_times.items():
            if times:
                avg_time = sum(times) / len(times)
                if avg_time > 1000:  # More than 1 second
                    self.log_with_timestamp(f"⚠️  SLOW RESPONSE {endpoint}: {avg_time:.1f}ms avg", "WARN")
        
        # Check for errors
        total_errors = sum(self.error_counts.values())
        if total_errors > 0:
            self.log_with_timestamp(f"❌ TOTAL ERRORS: {total_errors}", "ERROR")
            for endpoint, errors in self.error_counts.items():
                if errors > 0:
                    self.log_with_timestamp(f"   ❌ {endpoint}: {errors} errors", "ERROR")
        
        # Reset counters
        self.request_counts.clear()
        self.request_times.clear()
        self.error_counts.clear()
        self.last_check = current_time
    
    def monitor_docker_logs(self):
        """Monitor Docker container logs for issues"""
        try:
            # Get recent logs from all services
            result = subprocess.run([
                "docker-compose", "logs", "--tail=10", "--timestamps"
            ], capture_output=True, text=True, cwd="/Users/arpitsrivastava/Desktop/ITRIcometax/local-ml-security")
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:  # Show last 5 log lines
                    if any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'timeout']):
                        self.log_with_timestamp(f"Docker: {line}", "WARN")
            else:
                self.log_with_timestamp(f"Docker logs error: {result.stderr}", "ERROR")
        except Exception as e:
            self.log_with_timestamp(f"Docker monitoring error: {str(e)}", "ERROR")
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        self.log_with_timestamp("=" * 60)
        self.log_with_timestamp("MONITORING CYCLE START")
        
        # Check frontend
        self.log_with_timestamp("Checking Frontend...")
        self.check_frontend_requests()
        
        # Check backend
        self.log_with_timestamp("Checking Backend...")
        self.check_backend_requests()
        
        # Check services
        self.log_with_timestamp("Checking Services...")
        self.check_service_health()
        
        # Analyze patterns
        self.log_with_timestamp("Analyzing Patterns...")
        self.analyze_patterns()
        
        # Check Docker logs
        self.log_with_timestamp("Checking Docker Logs...")
        self.monitor_docker_logs()
        
        self.log_with_timestamp("MONITORING CYCLE COMPLETE")
        self.log_with_timestamp("=" * 60)
    
    def start_monitoring(self, interval=5):
        """Start continuous monitoring"""
        self.log_with_timestamp("🚀 STARTING FLICKERING MONITOR")
        self.log_with_timestamp("Monitoring interval: 5 seconds")
        self.log_with_timestamp("Press Ctrl+C to stop")
        self.log_with_timestamp("=" * 60)
        
        try:
            while self.monitoring:
                self.run_monitoring_cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.log_with_timestamp("🛑 MONITORING STOPPED BY USER")
        except Exception as e:
            self.log_with_timestamp(f"❌ MONITORING ERROR: {str(e)}", "ERROR")

def main():
    monitor = FlickeringMonitor()
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
