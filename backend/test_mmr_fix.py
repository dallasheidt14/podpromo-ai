#!/usr/bin/env python3
"""
Test the MMR selector fix
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.clip_score import _mmr_select_jaccard

def test_mmr_selector():
    """Test the MMR selector with various inputs"""
    
    print("Testing MMR selector fix...")
    
    # Test 1: Empty input
    result = _mmr_select_jaccard([], 5)
    print(f"Empty input: {len(result)} items (should be 0)")
    assert len(result) == 0, "Empty input should return empty list"
    
    # Test 2: Basic functionality
    items = [
        {"text": "This is a test clip about technology", "rank_score": 0.8},
        {"text": "Another test clip about business", "rank_score": 0.7},
        {"text": "A third clip about science", "rank_score": 0.6},
        {"text": "Technology clip with different content", "rank_score": 0.5},
        {"text": "Business advice and tips", "rank_score": 0.4},
    ]
    
    result = _mmr_select_jaccard(items, 3)
    print(f"Basic test: {len(result)} items selected")
    print("Selected items:")
    for i, item in enumerate(result):
        print(f"  {i+1}: {item['text'][:50]}... (score: {item['rank_score']:.2f})")
    
    assert len(result) == 3, "Should select 3 items"
    assert all(isinstance(item, dict) for item in result), "Should return list of dicts"
    
    # Test 3: With different text_key
    items_with_title = [
        {"title": "Tech news update", "rank_score": 0.9},
        {"title": "Business strategy", "rank_score": 0.8},
        {"title": "Science discovery", "rank_score": 0.7},
    ]
    
    result = _mmr_select_jaccard(items_with_title, 2, text_key="title")
    print(f"Different text_key test: {len(result)} items selected")
    print("Selected items:")
    for i, item in enumerate(result):
        print(f"  {i+1}: {item['title']} (score: {item['rank_score']:.2f})")
    
    assert len(result) == 2, "Should select 2 items"
    
    # Test 4: Fallback to raw_score
    items_raw_score = [
        {"text": "Clip with raw_score", "raw_score": 0.6},
        {"text": "Another raw_score clip", "raw_score": 0.5},
    ]
    
    result = _mmr_select_jaccard(items_raw_score, 2)
    print(f"Raw score fallback test: {len(result)} items selected")
    assert len(result) == 2, "Should select 2 items"
    
    # Test 5: Request more items than available
    result = _mmr_select_jaccard(items[:2], 5)
    print(f"Request more than available: {len(result)} items selected (should be 2)")
    assert len(result) == 2, "Should return only available items"
    
    print("✅ All MMR selector tests passed!")
    return True

if __name__ == "__main__":
    success = test_mmr_selector()
    exit(0 if success else 1)
