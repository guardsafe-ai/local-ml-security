"""
Performance Benchmarking Module
Provides comprehensive performance benchmarks for latency, throughput, and attack success rate validation.
"""

import time
import asyncio
import statistics
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    name: str
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_count: int
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AttackBenchmarkResult:
    """Attack benchmark result data structure"""
    attack_type: str
    success_rate: float
    average_confidence: float
    average_perturbation: float
    execution_time: float
    memory_usage: float
    throughput: float
    error_count: int
    timestamp: datetime
    metadata: Dict[str, Any]

class PerformanceBenchmark:
    """Performance benchmarking for red team operations"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.attack_results: List[AttackBenchmarkResult] = []
        self.baseline_metrics: Dict[str, float] = {}
        
    async def benchmark_latency(
        self,
        operation_name: str,
        operation_func,
        *args,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark operation latency"""
        logger.info(f"Benchmarking latency for {operation_name}")
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*args, **kwargs)
                else:
                    operation_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Benchmark
        execution_times = []
        memory_usage = []
        cpu_usage = []
        error_count = 0
        
        for i in range(iterations):
            try:
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                cpu_before = process.cpu_percent()
                
                # Execute operation
                start_time = time.time()
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*args, **kwargs)
                else:
                    operation_func(*args, **kwargs)
                end_time = time.time()
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                cpu_after = process.cpu_percent()
                
                execution_times.append(end_time - start_time)
                memory_usage.append(memory_after - memory_before)
                cpu_usage.append(cpu_after - cpu_before)
                
            except Exception as e:
                logger.error(f"Benchmark iteration {i} failed: {e}")
                error_count += 1
        
        # Calculate statistics
        if execution_times:
            latency_p50 = np.percentile(execution_times, 50)
            latency_p95 = np.percentile(execution_times, 95)
            latency_p99 = np.percentile(execution_times, 99)
            avg_execution_time = statistics.mean(execution_times)
            throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0
        else:
            latency_p50 = latency_p95 = latency_p99 = avg_execution_time = throughput = 0
        
        success_rate = (iterations - error_count) / iterations if iterations > 0 else 0
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        avg_cpu_usage = statistics.mean(cpu_usage) if cpu_usage else 0
        
        result = BenchmarkResult(
            name=operation_name,
            operation="latency",
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            cpu_usage=avg_cpu_usage,
            success_rate=success_rate,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_count=error_count,
            timestamp=datetime.now(),
            metadata={"iterations": iterations, "warmup_iterations": warmup_iterations}
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_throughput(
        self,
        operation_name: str,
        operation_func,
        *args,
        duration_seconds: int = 60,
        max_concurrent: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark operation throughput"""
        logger.info(f"Benchmarking throughput for {operation_name}")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        completed_operations = 0
        error_count = 0
        execution_times = []
        memory_usage = []
        cpu_usage = []
        
        async def run_operation():
            nonlocal completed_operations, error_count
            while time.time() < end_time:
                try:
                    # Measure memory before
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_before = process.cpu_percent()
                    
                    # Execute operation
                    op_start = time.time()
                    if asyncio.iscoroutinefunction(operation_func):
                        await operation_func(*args, **kwargs)
                    else:
                        operation_func(*args, **kwargs)
                    op_end = time.time()
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    cpu_after = process.cpu_percent()
                    
                    execution_times.append(op_end - op_start)
                    memory_usage.append(memory_after - memory_before)
                    cpu_usage.append(cpu_after - cpu_before)
                    completed_operations += 1
                    
                except Exception as e:
                    logger.error(f"Throughput benchmark operation failed: {e}")
                    error_count += 1
        
        # Run concurrent operations
        tasks = [run_operation() for _ in range(max_concurrent)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate statistics
        actual_duration = time.time() - start_time
        throughput = completed_operations / actual_duration if actual_duration > 0 else 0
        
        if execution_times:
            latency_p50 = np.percentile(execution_times, 50)
            latency_p95 = np.percentile(execution_times, 95)
            latency_p99 = np.percentile(execution_times, 99)
            avg_execution_time = statistics.mean(execution_times)
        else:
            latency_p50 = latency_p95 = latency_p99 = avg_execution_time = 0
        
        success_rate = (completed_operations - error_count) / completed_operations if completed_operations > 0 else 0
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        avg_cpu_usage = statistics.mean(cpu_usage) if cpu_usage else 0
        
        result = BenchmarkResult(
            name=operation_name,
            operation="throughput",
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            cpu_usage=avg_cpu_usage,
            success_rate=success_rate,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_count=error_count,
            timestamp=datetime.now(),
            metadata={
                "duration_seconds": actual_duration,
                "max_concurrent": max_concurrent,
                "completed_operations": completed_operations
            }
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_attack_success_rate(
        self,
        attack_type: str,
        attack_func,
        test_cases: List[Dict[str, Any]],
        iterations: int = 10,
        **kwargs
    ) -> AttackBenchmarkResult:
        """Benchmark attack success rate"""
        logger.info(f"Benchmarking attack success rate for {attack_type}")
        
        successful_attacks = 0
        total_attacks = 0
        confidences = []
        perturbations = []
        execution_times = []
        memory_usage = []
        error_count = 0
        
        for iteration in range(iterations):
            for test_case in test_cases:
                try:
                    # Measure memory before
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Execute attack
                    start_time = time.time()
                    if asyncio.iscoroutinefunction(attack_func):
                        result = await attack_func(test_case, **kwargs)
                    else:
                        result = attack_func(test_case, **kwargs)
                    end_time = time.time()
                    
                    # Measure memory after
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    execution_times.append(end_time - start_time)
                    memory_usage.append(memory_after - memory_before)
                    
                    if result and result.get("success", False):
                        successful_attacks += 1
                        confidences.append(result.get("confidence", 0))
                        perturbations.append(result.get("perturbation", 0))
                    
                    total_attacks += 1
                    
                except Exception as e:
                    logger.error(f"Attack benchmark failed: {e}")
                    error_count += 1
        
        # Calculate statistics
        success_rate = successful_attacks / total_attacks if total_attacks > 0 else 0
        avg_confidence = statistics.mean(confidences) if confidences else 0
        avg_perturbation = statistics.mean(perturbations) if perturbations else 0
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        throughput = total_attacks / sum(execution_times) if execution_times else 0
        
        result = AttackBenchmarkResult(
            attack_type=attack_type,
            success_rate=success_rate,
            average_confidence=avg_confidence,
            average_perturbation=avg_perturbation,
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            throughput=throughput,
            error_count=error_count,
            timestamp=datetime.now(),
            metadata={
                "iterations": iterations,
                "test_cases": len(test_cases),
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks
            }
        )
        
        self.attack_results.append(result)
        return result
    
    async def benchmark_memory_usage(
        self,
        operation_name: str,
        operation_func,
        *args,
        iterations: int = 100,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark memory usage"""
        logger.info(f"Benchmarking memory usage for {operation_name}")
        
        # Force garbage collection
        gc.collect()
        
        # Measure baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_usage = []
        execution_times = []
        error_count = 0
        
        for i in range(iterations):
            try:
                # Force garbage collection before each iteration
                gc.collect()
                
                # Measure memory before
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Execute operation
                start_time = time.time()
                if asyncio.iscoroutinefunction(operation_func):
                    await operation_func(*args, **kwargs)
                else:
                    operation_func(*args, **kwargs)
                end_time = time.time()
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                
                memory_usage.append(memory_after - memory_before)
                execution_times.append(end_time - start_time)
                
            except Exception as e:
                logger.error(f"Memory benchmark iteration {i} failed: {e}")
                error_count += 1
        
        # Calculate statistics
        avg_memory_usage = statistics.mean(memory_usage) if memory_usage else 0
        max_memory_usage = max(memory_usage) if memory_usage else 0
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        success_rate = (iterations - error_count) / iterations if iterations > 0 else 0
        throughput = 1.0 / avg_execution_time if avg_execution_time > 0 else 0
        
        result = BenchmarkResult(
            name=operation_name,
            operation="memory",
            execution_time=avg_execution_time,
            memory_usage=avg_memory_usage,
            cpu_usage=0,  # Not measured in memory benchmark
            success_rate=success_rate,
            throughput=throughput,
            latency_p50=avg_execution_time,
            latency_p95=avg_execution_time,
            latency_p99=avg_execution_time,
            error_count=error_count,
            timestamp=datetime.now(),
            metadata={
                "iterations": iterations,
                "baseline_memory": baseline_memory,
                "max_memory_usage": max_memory_usage
            }
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_concurrent_operations(
        self,
        operation_name: str,
        operation_func,
        *args,
        concurrent_levels: List[int] = [1, 5, 10, 20, 50],
        duration_seconds: int = 30,
        **kwargs
    ) -> List[BenchmarkResult]:
        """Benchmark concurrent operations at different levels"""
        logger.info(f"Benchmarking concurrent operations for {operation_name}")
        
        results = []
        
        for concurrent_level in concurrent_levels:
            logger.info(f"Testing {concurrent_level} concurrent operations")
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            completed_operations = 0
            error_count = 0
            execution_times = []
            
            async def run_operation():
                nonlocal completed_operations, error_count
                while time.time() < end_time:
                    try:
                        op_start = time.time()
                        if asyncio.iscoroutinefunction(operation_func):
                            await operation_func(*args, **kwargs)
                        else:
                            operation_func(*args, **kwargs)
                        op_end = time.time()
                        
                        execution_times.append(op_end - op_start)
                        completed_operations += 1
                        
                    except Exception as e:
                        logger.error(f"Concurrent operation failed: {e}")
                        error_count += 1
            
            # Run concurrent operations
            tasks = [run_operation() for _ in range(concurrent_level)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate statistics
            actual_duration = time.time() - start_time
            throughput = completed_operations / actual_duration if actual_duration > 0 else 0
            
            if execution_times:
                latency_p50 = np.percentile(execution_times, 50)
                latency_p95 = np.percentile(execution_times, 95)
                latency_p99 = np.percentile(execution_times, 99)
                avg_execution_time = statistics.mean(execution_times)
            else:
                latency_p50 = latency_p95 = latency_p99 = avg_execution_time = 0
            
            success_rate = (completed_operations - error_count) / completed_operations if completed_operations > 0 else 0
            
            result = BenchmarkResult(
                name=f"{operation_name}_concurrent_{concurrent_level}",
                operation="concurrent",
                execution_time=avg_execution_time,
                memory_usage=0,  # Not measured in concurrent benchmark
                cpu_usage=0,  # Not measured in concurrent benchmark
                success_rate=success_rate,
                throughput=throughput,
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                error_count=error_count,
                timestamp=datetime.now(),
                metadata={
                    "concurrent_level": concurrent_level,
                    "duration_seconds": actual_duration,
                    "completed_operations": completed_operations
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def set_baseline_metrics(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics
        logger.info(f"Set baseline metrics: {metrics}")
    
    def compare_with_baseline(self, result: BenchmarkResult) -> Dict[str, float]:
        """Compare result with baseline metrics"""
        if not self.baseline_metrics:
            return {}
        
        comparison = {}
        baseline_key = f"{result.name}_{result.operation}"
        
        if baseline_key in self.baseline_metrics:
            baseline_value = self.baseline_metrics[baseline_key]
            current_value = result.throughput if result.operation == "throughput" else result.execution_time
            
            if baseline_value > 0:
                improvement = ((current_value - baseline_value) / baseline_value) * 100
                comparison[f"{baseline_key}_improvement"] = improvement
        
        return comparison
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.results:
            return {"message": "No benchmark results available"}
        
        summary = {
            "total_benchmarks": len(self.results),
            "total_attack_benchmarks": len(self.attack_results),
            "average_throughput": statistics.mean([r.throughput for r in self.results if r.throughput > 0]),
            "average_latency_p50": statistics.mean([r.latency_p50 for r in self.results if r.latency_p50 > 0]),
            "average_latency_p95": statistics.mean([r.latency_p95 for r in self.results if r.latency_p95 > 0]),
            "average_latency_p99": statistics.mean([r.latency_p99 for r in self.results if r.latency_p99 > 0]),
            "average_success_rate": statistics.mean([r.success_rate for r in self.results]),
            "average_memory_usage": statistics.mean([r.memory_usage for r in self.results if r.memory_usage > 0]),
            "total_errors": sum(r.error_count for r in self.results),
            "attack_success_rates": {r.attack_type: r.success_rate for r in self.attack_results},
            "attack_throughputs": {r.attack_type: r.throughput for r in self.attack_results}
        }
        
        return summary
    
    def export_results(self, filename: str = None) -> str:
        """Export benchmark results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        import json
        
        data = {
            "benchmark_results": [
                {
                    "name": r.name,
                    "operation": r.operation,
                    "execution_time": r.execution_time,
                    "memory_usage": r.memory_usage,
                    "cpu_usage": r.cpu_usage,
                    "success_rate": r.success_rate,
                    "throughput": r.throughput,
                    "latency_p50": r.latency_p50,
                    "latency_p95": r.latency_p95,
                    "latency_p99": r.latency_p99,
                    "error_count": r.error_count,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in self.results
            ],
            "attack_results": [
                {
                    "attack_type": r.attack_type,
                    "success_rate": r.success_rate,
                    "average_confidence": r.average_confidence,
                    "average_perturbation": r.average_perturbation,
                    "execution_time": r.execution_time,
                    "memory_usage": r.memory_usage,
                    "throughput": r.throughput,
                    "error_count": r.error_count,
                    "timestamp": r.timestamp.isoformat(),
                    "metadata": r.metadata
                }
                for r in self.attack_results
            ],
            "summary": self.get_performance_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported benchmark results to {filename}")
        return filename
    
    def clear_results(self):
        """Clear all benchmark results"""
        self.results.clear()
        self.attack_results.clear()
        logger.info("Cleared all benchmark results")
