#!/usr/bin/env python3
"""
Performance Test Runner
Runs comprehensive performance tests based on configuration
"""

import asyncio
import argparse
import logging
import sys
import os
import yaml
from typing import Dict, Any, List
from datetime import datetime
import json

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

from services.benchmarking import PerformanceBenchmark
from services.monitoring import PerformanceDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTestRunner:
    """Performance test runner with configuration support"""
    
    def __init__(self, config_file: str = "performance_config.yaml"):
        self.config = self._load_config(config_file)
        self.benchmark = PerformanceBenchmark()
        self.dashboard = PerformanceDashboard()
        self.results = []
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "general": {
                "output_dir": "./performance_results",
                "log_level": "INFO",
                "max_workers": 4,
                "timeout_seconds": 300
            },
            "benchmarks": {
                "latency": {"iterations": 100, "warmup_iterations": 10},
                "throughput": {"duration_seconds": 60, "max_concurrent": 10},
                "memory": {"iterations": 50, "gc_before_each": True},
                "attack_success": {"iterations": 10, "test_cases_per_iteration": 5}
            },
            "thresholds": {
                "latency": {"warning": 1.0, "critical": 5.0},
                "throughput": {"warning": 0.1, "critical": 0.05},
                "memory_usage": {"warning": 100, "critical": 500},
                "cpu_usage": {"warning": 80, "critical": 95},
                "error_rate": {"warning": 0.05, "critical": 0.1},
                "success_rate": {"warning": 0.95, "critical": 0.9}
            }
        }
    
    async def run_latency_benchmarks(self):
        """Run latency benchmarks"""
        logger.info("Running latency benchmarks...")
        
        config = self.config.get("benchmarks", {}).get("latency", {})
        iterations = config.get("iterations", 100)
        warmup_iterations = config.get("warmup_iterations", 10)
        
        # Mock operations for testing
        def mock_operation():
            time.sleep(0.01)  # Simulate work
            return "result"
        
        def mock_heavy_operation():
            time.sleep(0.1)  # Simulate heavier work
            return "result"
        
        # Run latency benchmarks
        await self.benchmark.benchmark_latency(
            "mock_operation",
            mock_operation,
            iterations=iterations,
            warmup_iterations=warmup_iterations
        )
        
        await self.benchmark.benchmark_latency(
            "mock_heavy_operation",
            mock_heavy_operation,
            iterations=iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Add metrics to dashboard
        self.dashboard.add_metric("latency", 0.01, {"operation": "mock_operation"})
        self.dashboard.add_metric("latency", 0.1, {"operation": "mock_heavy_operation"})
    
    async def run_throughput_benchmarks(self):
        """Run throughput benchmarks"""
        logger.info("Running throughput benchmarks...")
        
        config = self.config.get("benchmarks", {}).get("throughput", {})
        duration_seconds = config.get("duration_seconds", 60)
        max_concurrent = config.get("max_concurrent", 10)
        
        def mock_operation():
            time.sleep(0.01)  # Simulate work
            return "result"
        
        # Run throughput benchmarks
        await self.benchmark.benchmark_throughput(
            "mock_operation",
            mock_operation,
            duration_seconds=duration_seconds,
            max_concurrent=max_concurrent
        )
        
        # Add metrics to dashboard
        self.dashboard.add_metric("throughput", 100.0, {"operation": "mock_operation"})
    
    async def run_memory_benchmarks(self):
        """Run memory benchmarks"""
        logger.info("Running memory benchmarks...")
        
        config = self.config.get("benchmarks", {}).get("memory", {})
        iterations = config.get("iterations", 50)
        
        def memory_intensive_operation():
            # Simulate memory-intensive operation
            data = [i for i in range(10000)]
            result = sum(data)
            return result
        
        # Run memory benchmarks
        await self.benchmark.benchmark_memory_usage(
            "memory_intensive_operation",
            memory_intensive_operation,
            iterations=iterations
        )
        
        # Add metrics to dashboard
        self.dashboard.add_metric("memory_usage", 50.0, {"operation": "memory_intensive_operation"})
    
    async def run_attack_success_benchmarks(self):
        """Run attack success rate benchmarks"""
        logger.info("Running attack success rate benchmarks...")
        
        config = self.config.get("benchmarks", {}).get("attack_success", {})
        iterations = config.get("iterations", 10)
        test_cases_per_iteration = config.get("test_cases_per_iteration", 5)
        
        # Generate test cases
        test_cases = [
            {"text": f"Test case {i}", "target": 1}
            for i in range(test_cases_per_iteration)
        ]
        
        def mock_attack(test_case):
            return {
                "success": True,
                "confidence": 0.8,
                "perturbation": 0.1
            }
        
        # Run attack success benchmarks
        await self.benchmark.benchmark_attack_success_rate(
            "mock_attack",
            mock_attack,
            test_cases,
            iterations=iterations
        )
        
        # Add metrics to dashboard
        self.dashboard.add_metric("success_rate", 0.8, {"attack_type": "mock_attack"})
        self.dashboard.add_metric("error_rate", 0.2, {"attack_type": "mock_attack"})
    
    async def run_concurrent_benchmarks(self):
        """Run concurrent operation benchmarks"""
        logger.info("Running concurrent operation benchmarks...")
        
        def concurrent_operation():
            time.sleep(0.01)  # Simulate work
            return "result"
        
        # Run concurrent benchmarks
        await self.benchmark.benchmark_concurrent_operations(
            "concurrent_operation",
            concurrent_operation,
            concurrent_levels=[1, 5, 10, 20],
            duration_seconds=20
        )
    
    async def run_all_benchmarks(self):
        """Run all benchmarks"""
        logger.info("Starting comprehensive performance test suite...")
        
        start_time = datetime.now()
        
        try:
            # Run all benchmark categories
            await self.run_latency_benchmarks()
            await self.run_throughput_benchmarks()
            await self.run_memory_benchmarks()
            await self.run_attack_success_benchmarks()
            await self.run_concurrent_benchmarks()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"Performance test suite completed in {duration}")
            
            # Generate summary
            summary = self.benchmark.get_performance_summary()
            logger.info(f"Performance summary: {summary}")
            
            # Get dashboard data
            dashboard_data = self.dashboard.get_dashboard_data()
            logger.info(f"Dashboard data: {dashboard_data}")
            
            # Export results
            self._export_results()
            
            return summary
            
        except Exception as e:
            logger.error(f"Performance test suite failed: {e}")
            raise
    
    def _export_results(self):
        """Export test results"""
        output_dir = self.config.get("general", {}).get("output_dir", "./performance_results")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export benchmark results
        benchmark_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
        self.benchmark.export_results(benchmark_file)
        
        # Export dashboard data
        dashboard_file = os.path.join(output_dir, f"dashboard_data_{timestamp}.json")
        self.dashboard.export_data(dashboard_file)
        
        # Export combined results
        combined_file = os.path.join(output_dir, f"performance_results_{timestamp}.json")
        combined_data = {
            "benchmark_results": [asdict(r) for r in self.benchmark.results],
            "attack_results": [asdict(r) for r in self.benchmark.attack_results],
            "dashboard_data": self.dashboard.get_dashboard_data(),
            "summary": self.benchmark.get_performance_summary(),
            "timestamp": timestamp
        }
        
        with open(combined_file, 'w') as f:
            json.dump(combined_data, f, indent=2, default=str)
        
        logger.info(f"Results exported to {output_dir}")
    
    def check_thresholds(self):
        """Check performance against thresholds"""
        logger.info("Checking performance thresholds...")
        
        thresholds = self.config.get("thresholds", {})
        alerts = []
        
        for result in self.benchmark.results:
            metric_name = result.name
            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                value = result.throughput if result.operation == "throughput" else result.execution_time
                
                if "warning" in threshold and value > threshold["warning"]:
                    alerts.append(f"WARNING: {metric_name} exceeded warning threshold: {value:.2f} > {threshold['warning']:.2f}")
                
                if "critical" in threshold and value > threshold["critical"]:
                    alerts.append(f"CRITICAL: {metric_name} exceeded critical threshold: {value:.2f} > {threshold['critical']:.2f}")
        
        if alerts:
            logger.warning("Performance threshold violations detected:")
            for alert in alerts:
                logger.warning(alert)
        else:
            logger.info("All performance thresholds met")
        
        return alerts
    
    def generate_report(self):
        """Generate performance report"""
        logger.info("Generating performance report...")
        
        summary = self.benchmark.get_performance_summary()
        dashboard_data = self.dashboard.get_dashboard_data()
        
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "dashboard_data": dashboard_data,
            "threshold_violations": self.check_thresholds(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        summary = self.benchmark.get_performance_summary()
        
        if summary.get("average_latency_p95", 0) > 1.0:
            recommendations.append("Consider optimizing high-latency operations")
        
        if summary.get("average_throughput", 0) < 0.1:
            recommendations.append("Consider implementing caching or horizontal scaling")
        
        if summary.get("average_memory_usage", 0) > 100:
            recommendations.append("Consider optimizing memory usage or implementing memory pooling")
        
        if summary.get("total_errors", 0) > 0:
            recommendations.append("Investigate and fix error conditions")
        
        return recommendations

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance Test Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="performance_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["all", "latency", "throughput", "memory", "attack_success", "concurrent"],
        default="all",
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Check performance against thresholds"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate performance report"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    runner = PerformanceTestRunner(args.config)
    
    if args.output_dir:
        runner.config["general"]["output_dir"] = args.output_dir
    
    try:
        if args.benchmark_type == "all":
            summary = await runner.run_all_benchmarks()
        else:
            if args.benchmark_type == "latency":
                await runner.run_latency_benchmarks()
            elif args.benchmark_type == "throughput":
                await runner.run_throughput_benchmarks()
            elif args.benchmark_type == "memory":
                await runner.run_memory_benchmarks()
            elif args.benchmark_type == "attack_success":
                await runner.run_attack_success_benchmarks()
            elif args.benchmark_type == "concurrent":
                await runner.run_concurrent_benchmarks()
            
            summary = runner.benchmark.get_performance_summary()
        
        print(f"\nPerformance tests completed successfully!")
        print(f"Summary: {summary}")
        
        if args.check_thresholds:
            violations = runner.check_thresholds()
            if violations:
                print(f"Threshold violations: {len(violations)}")
            else:
                print("All thresholds met")
        
        if args.generate_report:
            report = runner.generate_report()
            print(f"Report generated: {report}")
        
    except Exception as e:
        logger.error(f"Performance tests failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
