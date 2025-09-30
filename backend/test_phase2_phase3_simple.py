#!/usr/bin/env python3
"""
Simple test to verify Phase 2 & 3 fixes are working
"""

import asyncio
import sys
import json
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.secret_sauce_pkg.features import build_sentence_spans
from services.clip_score import soft_start_pad_and_trim, dedupe_keep_best, last_sentence_of

def test_phase2_phase3_simple():
    """Test Phase 2 & 3 fixes with simple examples"""
    
    print("🧪 Testing Phase 2 & 3 Fixes - Simple Test")
    print("=" * 50)
    
    try:
        # Test 1: Sentence spans building (Phase 2)
        print("\n🔍 Test 1: Sentence Spans (Phase 2)")
        
        # Load words from the latest episode
        episode_id = "09045058-b454-45eb-b24e-9f34874e302c"
        words_path = f"uploads/transcripts/{episode_id}/words.json"
        
        if not os.path.exists(words_path):
            print(f"❌ Words file not found: {words_path}")
            return False
            
        with open(words_path, 'r', encoding='utf-8') as f:
            raw_words = json.load(f)
        
        print(f"  - Loaded {len(raw_words)} words")
        
        # Convert words to the expected format (word -> text, add after field)
        words = []
        for word in raw_words:
            converted_word = {
                'start': word['start'],
                'end': word['end'],
                'text': word['word'].strip(),
                'after': ''  # No punctuation in after field for this format
            }
            words.append(converted_word)
        
        # Test sentence spans
        sentence_spans = build_sentence_spans(words)
        print(f"  - Generated {len(sentence_spans)} sentence spans")
        
        # Check if we have the "Ever watched" context
        ever_watched_found = False
        for span in sentence_spans:
            if 'ever watched' in span.text.lower():
                ever_watched_found = True
                print(f"  ✅ Found 'Ever watched' context: {span.text[:80]}...")
                break
        
        if not ever_watched_found:
            print("  ⚠️  'Ever watched' context not found in sentence spans")
        
        # Test 2: Soft start pad and trim (Phase 3)
        print("\n🔍 Test 2: Soft Start Pad and Trim (Phase 3)")
        
        # Create test clip
        original_clip = {'start': 10.0, 'end': 20.0}
        test_clip = original_clip.copy()
        
        # Create test words with filler (using correct structure with 'w' key)
        test_words = [
            {'t': 9.5, 'w': 'uh', 'start': 9.5, 'end': 9.8, 'after': ''},
            {'t': 9.8, 'w': 'well', 'start': 9.8, 'end': 10.0, 'after': ''},
            {'t': 10.0, 'w': 'so', 'start': 10.0, 'end': 10.2, 'after': ''},
            {'t': 10.2, 'w': 'this', 'start': 10.2, 'end': 10.5, 'after': ''},
            {'t': 10.5, 'w': 'is', 'start': 10.5, 'end': 10.7, 'after': ''},
            {'t': 10.7, 'w': 'important', 'start': 10.7, 'end': 11.0, 'after': ''},
        ]
        
        result = soft_start_pad_and_trim(test_clip, test_words, pad=0.45, max_trim=0.6)
        print(f"  - Original: {original_clip['start']:.2f}-{original_clip['end']:.2f}")
        print(f"  - Result: {result['start']:.2f}-{result['end']:.2f}")
        print(f"  - Trimmed: {result['start'] - original_clip['start']:.2f}s")
        
        # Test 3: Deduplication (Phase 3)
        print("\n🔍 Test 3: Deduplication (Phase 3)")
        
        test_clips = [
            {'text': 'This is a test clip about important topics', 'final_score': 0.8},
            {'text': 'This is a test clip about important topics', 'final_score': 0.9},  # Duplicate
            {'text': 'Another completely different clip here', 'final_score': 0.7},
            {'text': 'This is a test clip about important topics', 'final_score': 0.6},  # Duplicate
        ]
        
        deduped = dedupe_keep_best(test_clips)
        print(f"  - Original: {len(test_clips)} clips")
        print(f"  - After dedupe: {len(deduped)} clips")
        print(f"  - Removed: {len(test_clips) - len(deduped)} duplicates")
        
        # Test 4: Last sentence extraction (Phase 3)
        print("\n🔍 Test 4: Last Sentence Extraction (Phase 3)")
        
        test_text = "This is the first sentence. This is the second sentence. This is the final sentence!"
        last_sentence = last_sentence_of(test_text)
        print(f"  - Text: {test_text}")
        print(f"  - Last sentence: {last_sentence}")
        
        # Test 5: Check for errors in the latest run
        print("\n🔍 Test 5: Error Check")
        
        # Check if there are any recent error logs
        log_path = "backend.log"
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Look for recent errors
            error_count = log_content.count('REFINE_BOUNDS_ERROR')
            settings_error_count = log_content.count('name \'settings\' is not defined')
            
            print(f"  - REFINE_BOUNDS_ERROR count: {error_count}")
            print(f"  - Settings error count: {settings_error_count}")
            
            if settings_error_count > 0:
                print("  ❌ Found settings errors in logs")
            else:
                print("  ✅ No settings errors found in logs")
        else:
            print("  ⚠️  Log file not found")
        
        print("\n🎯 Test Summary:")
        print(f"  - Sentence spans: {'✅' if len(sentence_spans) > 0 else '❌'}")
        print(f"  - Context captured: {'✅' if ever_watched_found else '❌'}")
        print(f"  - Soft start pad: {'✅' if result['start'] != original_clip['start'] else '❌'}")
        print(f"  - Deduplication: {'✅' if len(deduped) < len(test_clips) else '❌'}")
        print(f"  - Last sentence: {'✅' if last_sentence == 'This is the final sentence!' else '❌'}")
        print(f"  - No settings errors: {'✅' if settings_error_count == 0 else '❌'}")
        
        return (len(sentence_spans) > 0 and 
                ever_watched_found and 
                result['start'] != original_clip['start'] and 
                len(deduped) < len(test_clips) and 
                last_sentence == 'This is the final sentence!' and
                settings_error_count == 0)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_phase2_phase3_simple()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Phase 2 & 3 fixes are working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED - Issues detected")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
