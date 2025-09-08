#!/usr/bin/env python3
"""
Simple test script to verify the refactoring works
"""

def test_imports():
    """Test that all imports work"""
    try:
        print("Testing imports...")
        
        # Test core imports
        from services.secret_sauce_pkg import compute_features_v4, score_segment_v4
        print("✅ Core imports work")
        
        # Test individual modules
        from services.secret_sauce_pkg.scoring import get_clip_weights
        print("✅ Scoring module works")
        
        from services.secret_sauce_pkg.features import _hook_score
        print("✅ Features module works")
        
        from services.secret_sauce_pkg.genres import GenreProfile
        print("✅ Genres module works")
        
        # Test a simple function
        weights = get_clip_weights()
        print(f"✅ get_clip_weights() returns: {type(weights)}")
        
        # Test hook scoring
        hook_score = _hook_score("How I made $10,000 in one month")
        print(f"✅ _hook_score() works: {hook_score}")
        
        # Test V4 scoring
        features = {
            'hook_score': 0.8,
            'arousal_score': 0.6,
            'emotion_score': 0.4,
            'payoff_score': 0.7,
            'info_density': 0.5,
            'question_score': 0.3,
            'loopability': 0.6,
            'platform_len_match': 0.8,
            'insight_score': 0.4
        }
        result = score_segment_v4(features)
        print(f"✅ score_segment_v4() works: {result['final_score']:.3f}")
        
        print("\n🎉 All tests passed! Refactoring is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
