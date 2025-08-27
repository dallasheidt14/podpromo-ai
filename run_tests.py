#!/usr/bin/env python3
"""
Simple test runner for PodPromo tests.
Run with: python run_tests.py
"""

import sys
import os
import importlib.util

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def run_test_file(test_file):
    """Run all test functions in a test file"""
    print(f"\n=== Running {test_file} ===")
    
    # Import the test module
    spec = importlib.util.spec_from_file_location("test_module", test_file)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    
    # Find all test functions
    test_functions = [name for name in dir(test_module) if name.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for test_name in test_functions:
        try:
            test_func = getattr(test_module, test_name)
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: {e}")
            failed += 1
    
    return passed, failed

def main():
    """Run all tests"""
    print("PodPromo Test Suite")
    print("=" * 50)
    
    test_files = [
        "tests/test_scoring.py",
        "tests/test_text_detectors.py",
        "tests/test_config_loader.py",
        # Note: test_rank_candidates.py requires pytest for monkeypatch
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            passed, failed = run_test_file(test_file)
            total_passed += passed
            total_failed += failed
        else:
            print(f"Warning: {test_file} not found")
    
    print("\n" + "=" * 50)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    
    if total_failed > 0:
        sys.exit(1)
    else:
        print("All tests passed! 🎉")

if __name__ == "__main__":
    main()
