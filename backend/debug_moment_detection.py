#!/usr/bin/env python3
"""
Debug script for ViralMomentDetector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.viral_moment_detector import ViralMomentDetector
import re

def debug_story_detection():
    """Debug the story detection process"""
    print("=== Debugging Story Detection ===")
    
    transcript = [
        {'text': "So one time I was watching this game and it was incredible", 'start': 0, 'end': 5},
        {'text': "and this player made the most incredible play I've ever seen", 'start': 5, 'end': 10},
        {'text': "nobody saw it coming, it was completely unexpected", 'start': 10, 'end': 15},
        {'text': "and that's how they won the championship that year", 'start': 15, 'end': 20}
    ]
    
    detector = ViralMomentDetector(genre='sports')
    
    print("Transcript:")
    for i, seg in enumerate(transcript):
        print(f"  {i}: '{seg['text']}' ({seg['start']}-{seg['end']})")
    
    print("\nStrong story start patterns:")
    for pattern in detector.moment_patterns['story']['starts']:
        print(f"  {pattern}")
    
    print("\nTesting each segment:")
    for i, segment in enumerate(transcript):
        text = segment.get('text', '')
        print(f"\nSegment {i}: '{text}'")
        
        # Test strong opening
        has_strong = detector._has_strong_opening(text, detector.moment_patterns['story']['starts'])
        print(f"  Has strong opening: {has_strong}")
        
        # Test each pattern individually
        for j, pattern in enumerate(detector.moment_patterns['story']['starts']):
            matches = re.match(pattern, text.lower().strip())
            print(f"    Pattern {j}: {pattern} -> {bool(matches)}")
        
        # Test additional strong indicators
        strong_indicators = [
            r"^(?:I was|We were|He was|She was|They were)",
            r"^(?:This is|That was|It was|There was)",
            r"^(?:Last week|Yesterday|Earlier|This morning|Tonight)",
            r"^(?:I'm telling you|Listen|Look|Check this out|Get this)"
        ]
        
        print("  Additional strong indicators:")
        for j, pattern in enumerate(strong_indicators):
            matches = re.match(pattern, text.lower().strip())
            print(f"    {pattern} -> {bool(matches)}")
    
    print("\nTesting story extraction:")
    # Test _extract_story_arc directly
    story = detector._extract_story_arc(transcript, 0, max_duration=30)
    print(f"Story extraction result: {story}")
    
    if story:
        print(f"Story text: '{story['text']}'")
        print(f"Story duration: {story['end'] - story['start']:.1f}s")
        
        # Test self-contained check
        is_contained = detector._is_self_contained(story['text'])
        print(f"Is self-contained: {is_contained}")
        
        # Debug self-contained check
        text_lower = story['text'].lower().strip()
        print(f"Text for self-contained check: '{text_lower}'")
        
        # Test the subject-action pattern
        subject_action_match = re.search(r'\b(?:I|we|he|she|they|it|this|that)\s+\w+', text_lower)
        print(f"Subject-action pattern match: {bool(subject_action_match)}")
        if subject_action_match:
            print(f"  Matched: '{subject_action_match.group()}'")
    
    print("\nTesting moment detection:")
    moments = detector.find_moments(transcript)
    print(f"Found {len(moments)} moments")
    
    for i, moment in enumerate(moments):
        print(f"  Moment {i+1}: {moment}")

if __name__ == "__main__":
    debug_story_detection()
