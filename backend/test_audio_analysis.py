#!/usr/bin/env python3
"""
Test script for the new audio analysis functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from services.secret_sauce_pkg import _audio_prosody_score, compute_audio_energy
    import numpy as np
    
    print("🎵 Testing Audio Analysis Functions")
    print("=" * 50)
    
    # Test 1: Import successful
    print("✅ Audio analysis functions imported successfully")
    
    # Test 2: Test compute_audio_energy with dummy data
    print("\n🧪 Testing compute_audio_energy with dummy data...")
    
    # Create dummy audio data (random noise)
    dummy_y = np.random.randn(16000)  # 1 second of audio at 16kHz
    dummy_sr = 16000
    
    try:
        energy_score = compute_audio_energy(dummy_y, dummy_sr)
        print(f"✅ compute_audio_energy: {energy_score:.3f}")
        print(f"   - Expected range: 0.0 to 1.0")
        print(f"   - Actual result: {energy_score:.3f}")
        
        if 0.0 <= energy_score <= 1.0:
            print("   ✅ Score is within valid range")
        else:
            print("   ❌ Score is outside valid range")
            
    except Exception as e:
        print(f"❌ compute_audio_energy failed: {e}")
    
    # Test 3: Test _audio_prosody_score (this will fail without real audio file, but should handle gracefully)
    print("\n🧪 Testing _audio_prosody_score error handling...")
    
    try:
        # This should fail gracefully since we don't have a real audio file
        score = _audio_prosody_score("nonexistent_file.mp3", 0.0, 5.0)
        print(f"✅ _audio_prosody_score returned: {score:.3f}")
        print("   - Expected: 0.0 (fallback value)")
        print("   - Actual: {score:.3f}")
        
        if score == 0.0:
            print("   ✅ Graceful fallback working")
        else:
            print("   ⚠️ Unexpected fallback value")
            
    except Exception as e:
        print(f"❌ _audio_prosody_score failed: {e}")
    
    print("\n🎉 Audio analysis test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
