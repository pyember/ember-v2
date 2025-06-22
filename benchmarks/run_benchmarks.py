#!/usr/bin/env python3
"""Run Ember performance benchmarks.

This script runs comprehensive performance benchmarks and can output
results in different formats for tracking performance over time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.benchmarks.test_comprehensive_performance import run_benchmarks


def main():
    parser = argparse.ArgumentParser(description="Run Ember performance benchmarks")
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--compare",
        type=str,
        help="Compare with previous results file"
    )
    
    args = parser.parse_args()
    
    print("Running Ember Performance Benchmarks...")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run benchmarks
    start_time = time.time()
    run_benchmarks()
    total_time = time.time() - start_time
    
    print(f"\nTotal benchmark time: {total_time:.1f}s")
    
    # TODO: Add result collection and comparison logic
    if args.output:
        print(f"\nResults would be saved to: {args.output}")
    
    if args.compare:
        print(f"\nResults would be compared with: {args.compare}")


if __name__ == "__main__":
    main()