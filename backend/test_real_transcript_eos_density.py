#!/usr/bin/env python3
"""
Test script for EOS density standardization on real transcript data.
Tests with the actual transcript file: c746f928-85c5-4541-8119-261e659291d8.json
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.secret_sauce_pkg.features import build_eos_index
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_real_transcript():
    """Load the real transcript file"""
    transcript_path = "uploads/transcripts/c746f928-85c5-4541-8119-261e659291d8.json"
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded transcript: {data.get('original_name', 'Unknown')}")
        logger.info(f"Duration: {data.get('duration', 0):.1f} seconds ({data.get('duration', 0)/60:.1f} minutes)")
        logger.info(f"Status: {data.get('status', 'Unknown')}")
        
        segments = data.get('transcript', [])
        logger.info(f"Number of segments: {len(segments)}")
        
        return segments, data
        
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return None, None

def test_real_transcript_eos_density():
    """Test EOS density calculation on real transcript data"""
    logger.info("Testing EOS density standardization on real transcript...")
    
    # Load real transcript
    segments, transcript_data = load_real_transcript()
    if not segments:
        logger.error("Failed to load transcript data")
        return False
    
    # Extract episode words if available
    episode_words = []
    for segment in segments:
        words = segment.get('words', [])
        episode_words.extend(words)
    
    logger.info(f"Total words in episode: {len(episode_words)}")
    
    # Test build_eos_index function with real data
    try:
        eos_times, word_end_times, eos_source = build_eos_index(segments, episode_words)
        
        logger.info(f"\nEOS index result for real transcript:")
        logger.info(f"  EOS markers: {len(eos_times)}")
        logger.info(f"  Word boundaries: {len(word_end_times)}")
        logger.info(f"  Source: {eos_source}")
        
        if eos_times and word_end_times:
            # Calculate density manually to verify
            duration_minutes = (word_end_times[-1] - word_end_times[0]) / 60.0
            density_per_min = len(eos_times) / duration_minutes
            
            logger.info(f"  Duration: {duration_minutes:.1f} minutes")
            logger.info(f"  Density: {density_per_min:.3f} markers per minute")
            
            # Show some sample EOS times
            logger.info(f"  Sample EOS times: {eos_times[:5]}")
            
            # Verify the density is reasonable
            if 0.5 <= density_per_min <= 15.0:
                logger.info("✓ EOS density is within reasonable range for real transcript")
                return True
            else:
                logger.warning(f"⚠ EOS density {density_per_min:.3f} seems unusual for real transcript")
                return False
        else:
            logger.warning("No EOS markers or word boundaries found in real transcript")
            return False
            
    except Exception as e:
        logger.error(f"EOS density test failed on real transcript: {e}")
        return False

def test_dynamic_discovery_with_real_data():
    """Test dynamic discovery with real transcript data"""
    logger.info("Testing dynamic discovery with real transcript data...")
    
    # Load real transcript
    segments, transcript_data = load_real_transcript()
    if not segments:
        logger.error("Failed to load transcript data")
        return False
    
    # Extract words and create EOS times
    words = []
    for segment in segments:
        segment_words = segment.get('words', [])
        words.extend(segment_words)
    
    # Create EOS times from segment boundaries
    eos_times = []
    for segment in segments:
        eos_times.append(segment.get('end', 0))
    
    duration_s = transcript_data.get('duration', 0)
    
    logger.info(f"Real transcript stats:")
    logger.info(f"  Duration: {duration_s:.1f} seconds ({duration_s/60:.1f} minutes)")
    logger.info(f"  Words: {len(words)}")
    logger.info(f"  EOS markers: {len(eos_times)}")
    
    # Test dynamic discovery
    try:
        from services.secret_sauce_pkg.features import discover_dynamic_length
        
        def mock_score_fn(text, start, end):
            # Simple mock scoring function
            duration = end - start
            return min(0.8, duration / 30.0)  # Score based on duration
        
        candidates = discover_dynamic_length(words, eos_times, mock_score_fn, duration_s)
        
        logger.info(f"Dynamic discovery result:")
        logger.info(f"  Candidates found: {len(candidates)}")
        
        if candidates:
            logger.info("  Sample candidates:")
            for i, candidate in enumerate(candidates[:3]):
                logger.info(f"    {i+1}: start={candidate['start']:.1f}s, end={candidate['end']:.1f}s, score={candidate['score']:.3f}")
        
        return len(candidates) > 0
        
    except Exception as e:
        logger.error(f"Dynamic discovery test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting EOS density standardization tests on real transcript data...")
    
    tests = [
        ("Real Transcript EOS Density", test_real_transcript_eos_density),
        ("Dynamic Discovery with Real Data", test_dynamic_discovery_with_real_data),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    if all_passed:
        logger.info("\n✓ EOS density standardization works correctly with real transcript data!")
        logger.info("  - Real transcript processing successful")
        logger.info("  - EOS density calculations consistent")
        logger.info("  - Dynamic discovery stable with real data")
