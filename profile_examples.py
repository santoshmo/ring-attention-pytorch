#!/usr/bin/env python3
"""
Tree Attention Profiling Examples

This script demonstrates different profiling scenarios for tree attention decoding
to help analyze computation vs communication trade-offs.
"""

import subprocess
import sys

def run_profile_test(name, args, description):
    """Run a profiling test with given arguments"""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Description: {description}")
    print(f"Command: python assert_tree_attn.py {' '.join(args)}")
    print('='*60)
    
    try:
        result = subprocess.run([sys.executable, "assert_tree_attn.py"] + args, 
                              capture_output=False, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running test: {e}")
    
    print(f"\nCompleted: {name}")

def main():
    """Run various profiling scenarios"""
    
    print("Tree Attention Profiling Examples")
    print("This will run several profiling scenarios to demonstrate:")
    print("1. Small vs large sequence lengths")
    print("2. Different world sizes (communication overhead)")
    print("3. CPU vs CUDA performance characteristics")
    
    # Test 1: Small scale CPU test
    run_profile_test(
        "Small Scale CPU",
        ["--world-size", "2", "--seq-len", "128", "--enable-profiling", "--profile-iterations", "5"],
        "Small problem size on CPU to see baseline computation vs communication"
    )
    
    # Test 2: Medium scale CPU test
    run_profile_test(
        "Medium Scale CPU",
        ["--world-size", "4", "--seq-len", "1024", "--enable-profiling", "--profile-iterations", "5"],
        "Medium problem size with more processes to see communication scaling"
    )
    
    # Test 3: Large world size to see communication overhead
    run_profile_test(
        "High Communication Overhead",
        ["--world-size", "8", "--seq-len", "512", "--enable-profiling", "--profile-iterations", "3"],
        "Many processes with smaller per-process work to highlight communication costs"
    )
    
    # Test 4: CUDA tests (if available)
    try:
        import torch
        if torch.cuda.is_available():
            run_profile_test(
                "CUDA Performance",
                ["--world-size", "2", "--seq-len", "2048", "--use-cuda", "--enable-profiling", "--profile-iterations", "10"],
                "CUDA performance with larger sequence length"
            )
            
            run_profile_test(
                "CUDA Large Scale",
                ["--world-size", "4", "--seq-len", "4096", "--use-cuda", "--enable-profiling", "--profile-iterations", "5"],
                "Large scale CUDA test to see GPU compute vs communication balance"
            )
        else:
            print("\nSkipping CUDA tests - CUDA not available")
    except ImportError:
        print("\nSkipping CUDA tests - PyTorch not available")

if __name__ == "__main__":
    main() 