#!/usr/bin/env python3
"""
Simple test to check if imports are working
"""

print("Starting simple test...")

try:
    from services.secret_sauce_pkg import GenreAwareScorer
    print("✅ GenreAwareScorer imported successfully")
    
    scorer = GenreAwareScorer()
    print("✅ GenreAwareScorer instantiated successfully")
    
    # Test auto-detection
    text = "fantasy football waiver wire picks for this week"
    detected = scorer.auto_detect_genre(text)
    print(f"✅ Auto-detection test: '{text[:30]}...' -> {detected}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
