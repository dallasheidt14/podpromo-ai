#!/usr/bin/env python3
"""
Test script for the improved ViralMomentDetector
Tests the tighter duration constraints and better moment detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.viral_moment_detector import ViralMomentDetector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sports_moment_detection():
    """Test sports genre moment detection"""
    print("=== Testing Sports Genre Moment Detection ===")
    
    transcript = [
        {'text': "So one time I was watching this game and it was incredible", 'start': 0, 'end': 5},
        {'text': "and this player made the most incredible play I've ever seen", 'start': 5, 'end': 10},
        {'text': "nobody saw it coming, it was completely unexpected", 'start': 10, 'end': 15},
        {'text': "and that's how they won the championship that year", 'start': 15, 'end': 20}
    ]
    
    detector = ViralMomentDetector(genre='sports')
    moments = detector.find_moments(transcript)
    
    print(f"Found {len(moments)} moments")
    for i, moment in enumerate(moments):
        duration = moment['end'] - moment['start']
        print(f"  Moment {i+1}: {moment['type']} ({duration:.1f}s)")
        print(f"    Text: {moment['text'][:100]}...")
        print(f"    Confidence: {moment['confidence']}")
        print()
    
    # Assertions
    assert len(moments) > 0, "Should find at least one moment"
    assert moments[0]['end'] - moments[0]['start'] <= 30, "Story should be <= 30 seconds"
    assert 'story' in moments[0].get('type', ''), "Should be a story type"

def test_comedy_moment_detection():
    """Test comedy genre moment detection"""
    print("=== Testing Comedy Genre Moment Detection ===")
    
    transcript = [
        {'text': "So this one time I was at this party and it was wild", 'start': 0, 'end': 5},
        {'text': "and the funniest thing happened that night", 'start': 5, 'end': 10},
        {'text': "this guy walks in wearing a chicken costume", 'start': 10, 'end': 15},
        {'text': "and nobody knew why he was dressed like that", 'start': 15, 'end': 20},
        {'text': "turns out it was a dare from his friends", 'start': 20, 'end': 25}
    ]
    
    detector = ViralMomentDetector(genre='comedy')
    moments = detector.find_moments(transcript)
    
    print(f"Found {len(moments)} moments")
    for i, moment in enumerate(moments):
        duration = moment['end'] - moment['start']
        print(f"  Moment {i+1}: {moment['type']} ({duration:.1f}s)")
        print(f"    Text: {moment['text'][:100]}...")
        print(f"    Confidence: {moment['confidence']}")
        print()
    
    assert len(moments) > 0, "Should find at least one moment"
    assert moments[0]['end'] - moments[0]['start'] <= 30, "Story should be <= 30 seconds"

def test_insight_detection():
    """Test insight moment detection"""
    print("=== Testing Insight Moment Detection ===")
    
    transcript = [
        {'text': "Here's the thing people don't understand", 'start': 0, 'end': 5},
        {'text': "the key to success isn't talent", 'start': 5, 'end': 10},
        {'text': "it's consistency and hard work", 'start': 10, 'end': 15},
        {'text': "every single day", 'start': 15, 'end': 20}
    ]
    
    detector = ViralMomentDetector(genre='business')
    moments = detector.find_moments(transcript)
    
    print(f"Found {len(moments)} moments")
    for i, moment in enumerate(moments):
        duration = moment['end'] - moment['start']
        print(f"  Moment {i+1}: {moment['type']} ({duration:.1f}s)")
        print(f"    Text: {moment['text'][:100]}...")
        print(f"    Confidence: {moment['confidence']}")
        print()
    
    assert len(moments) > 0, "Should find at least one moment"
    assert moments[0]['end'] - moments[0]['start'] <= 25, "Insight should be <= 25 seconds"

def test_hot_take_detection():
    """Test hot take moment detection"""
    print("=== Testing Hot Take Moment Detection ===")
    
    transcript = [
        {'text': "Unpopular opinion but", 'start': 0, 'end': 4},
        {'text': "this player is completely overrated", 'start': 4, 'end': 8},
        {'text': "the stats don't lie", 'start': 8, 'end': 12}
    ]
    
    detector = ViralMomentDetector(genre='sports')
    moments = detector.find_moments(transcript)
    
    print(f"Found {len(moments)} moments")
    for i, moment in enumerate(moments):
        duration = moment['end'] - moment['start']
        print(f"  Moment {i+1}: {moment['type']} ({duration:.1f}s)")
        print(f"    Text: {moment['text'][:100]}...")
        print(f"    Confidence: {moment['confidence']}")
        print()
    
    assert len(moments) > 0, "Should find at least one moment"
    assert moments[0]['end'] - moments[0]['start'] <= 20, "Hot take should be <= 20 seconds"

def test_self_contained_check():
    """Test self-contained text checking"""
    print("=== Testing Self-Contained Text Check ===")
    
    detector = ViralMomentDetector()
    
    # Good examples (self-contained)
    good_texts = [
        "So one time I was at the store and this crazy thing happened",
        "Here's what I learned about success",
        "Unpopular opinion but this is actually true"
    ]
    
    # Bad examples (obviously context-dependent)
    bad_texts = [
        "As I mentioned before",
        "Going back to what we discussed",
        "As I told you earlier",
        "Returning to the topic"
    ]
    
    print("Testing good (self-contained) texts:")
    for text in good_texts:
        is_contained = detector._is_self_contained(text)
        print(f"  '{text[:50]}...' -> {is_contained}")
        assert is_contained, f"Should be self-contained: {text}"
    
    print("\nTesting bad (obviously context-dependent) texts:")
    for text in bad_texts:
        is_contained = detector._is_self_contained(text)
        print(f"  '{text[:50]}...' -> {is_contained}")
        assert not is_contained, f"Should not be self-contained: {text}"

def test_strong_opening_check():
    """Test strong opening hook detection"""
    print("=== Testing Strong Opening Hook Detection ===")
    
    detector = ViralMomentDetector()
    
    # Good examples (strong openings)
    good_openings = [
        "So one time I was watching this game",
        "Let me tell you about this crazy story",
        "Here's what happened yesterday",
        "This is the funniest thing"
    ]
    
    # Bad examples (weak openings)
    bad_openings = [
        "this guy was amazing",
        "it was incredible",
        "and then we went",
        "so anyway"
    ]
    
    print("Testing good (strong) openings:")
    for text in good_openings:
        has_strong = detector._has_strong_opening(text, detector.moment_patterns['story']['starts'])
        print(f"  '{text[:50]}...' -> {has_strong}")
        assert has_strong, f"Should have strong opening: {text}"
    
    print("\nTesting bad (weak) openings:")
    for text in bad_openings:
        has_strong = detector._has_strong_opening(text, detector.moment_patterns['story']['starts'])
        print(f"  '{text[:50]}...' -> {has_strong}")
        assert not has_strong, f"Should not have strong opening: {text}"

def main():
    """Run all tests"""
    print("Testing Improved ViralMomentDetector")
    print("=" * 50)
    
    try:
        test_sports_moment_detection()
        test_comedy_moment_detection()
        test_insight_detection()
        test_hot_take_detection()
        test_self_contained_check()
        test_strong_opening_check()
        
        print("\n🎉 All tests passed! The improved ViralMomentDetector is working correctly.")
        print("\nKey improvements verified:")
        print("- Tighter duration constraints (stories ≤30s, insights ≤25s, hot takes ≤20s)")
        print("- Strong opening hook detection")
        print("- Self-contained text validation")
        print("- Genre-specific pattern matching")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
