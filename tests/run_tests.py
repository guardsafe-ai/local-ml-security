#!/usr/bin/env python3
"""
Comprehensive Test Runner for ML Security Platform
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run ML Security Platform Tests")
    parser.add_argument("--type", choices=["unit", "integration", "e2e", "performance", "platform", "backend", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory based on type
    if args.type == "unit":
        cmd.append("tests/platform/unit/")
    elif args.type == "integration":
        cmd.append("tests/platform/integration/")
    elif args.type == "e2e":
        cmd.append("tests/platform/e2e/")
    elif args.type == "performance":
        cmd.append("tests/platform/performance/")
    elif args.type == "platform":
        cmd.append("tests/platform/")
    elif args.type == "backend":
        cmd.append("tests/services/enterprise-dashboard-backend/")
    else:  # all
        cmd.append("tests/")
    
    # Add options
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=services", "--cov-report=html", "--cov-report=term"])
    
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    if args.html:
        cmd.extend(["--html=tests/reports/report.html", "--self-contained-html"])
    
    if args.benchmark:
        cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])
    
    # Add markers
    cmd.extend(["-m", "not slow"])  # Skip slow tests by default
    
    # Run tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if success:
        print(f"\n🎉 All {args.type} tests passed!")
        return 0
    else:
        print(f"\n💥 Some {args.type} tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
