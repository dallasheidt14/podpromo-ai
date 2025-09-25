#!/usr/bin/env python3
"""
Run regression tests for the critical fixes we implemented.
These tests guard against the specific failure modes we fixed:
- NoneType crashes in telemetry
- 'str' object has no attribute 'get' in finish confidence
- Tail snap overextension
- finish_threshold_for with wrong input types
"""

import sys
import os
import subprocess

def main():
    # Add the backend directory to Python path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, backend_dir)
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    print("Running regression tests for critical fixes...")
    print("=" * 50)
    
    # Run pytest with verbose output
    cmd = [sys.executable, "-m", "pytest", "-v", "tests/test_util_words_and_eos.py", 
           "tests/test_clip_score_telemetry.py", "tests/test_tail_snap_and_gates.py", 
           "tests/test_finish_thresholds.py"]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All regression tests passed!")
        print("The critical fixes are working correctly.")
        return 0
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        print(f"Exit code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("❌ pytest not found. Install it with: pip install pytest")
        return 1

if __name__ == "__main__":
    sys.exit(main())
