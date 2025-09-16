# backend/tests/test_transcript_utils.py
# Test suite for exact word timestamp slicing

import unittest
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.transcript_utils import (
    words_between,
    words_to_text,
    words_to_captions,
    captions_to_vtt,
    _normalize_text
)

class TestTranscriptUtils(unittest.TestCase):
    
    def test_exact_slice_basic(self):
        """Test basic word slicing functionality"""
        words = [
            {"word": "Hello", "start": 1.00, "end": 1.20},
            {"word": "world", "start": 1.21, "end": 1.45},
            {"word": "!", "start": 1.45, "end": 1.46},
            {"word": "Bye", "start": 2.00, "end": 2.20},
        ]
        w = words_between(words, 1.00, 1.46)
        assert [x["word"] for x in w] == ["Hello", "world", "!"]
        text = words_to_text(w)
        assert text == "Hello world!"
    
    def test_relative_caption_times(self):
        """Test relative caption timing calculation"""
        words = [
            {"word": "A", "start": 10.10, "end": 10.30},
            {"word": "B", "start": 10.40, "end": 10.55}
        ]
        caps = words_to_captions(words, clip_start=10.00)
        expected = [
            {"t": 0.1, "d": 0.2, "w": "A"},
            {"t": 0.4, "d": 0.15, "w": "B"}
        ]
        self.assertEqual(caps, expected)
    
    def test_boundary_tolerance(self):
        """Test boundary tolerance handling"""
        words = [
            {"word": "Start", "start": 1.00, "end": 1.20},
            {"word": "Middle", "start": 1.50, "end": 1.70},
            {"word": "End", "start": 1.90, "end": 2.00},
        ]
        # Test with tolerance - should include words slightly outside bounds
        w = words_between(words, 1.05, 1.95, tol=0.1)
        self.assertEqual(len(w), 3)  # All words should be included
        
        # Test without tolerance - should exclude boundary words
        w_strict = words_between(words, 1.05, 1.95, tol=0.0)
        self.assertEqual(len(w_strict), 1)  # Only middle word
    
    def test_filler_removal(self):
        """Test disfluency/filler word removal"""
        words = [
            {"word": "So", "start": 1.00, "end": 1.10},
            {"word": "um", "start": 1.11, "end": 1.15},
            {"word": "let's", "start": 1.16, "end": 1.25},
            {"word": "uh", "start": 1.26, "end": 1.30},
            {"word": "go", "start": 1.31, "end": 1.40},
        ]
        
        # With filler removal
        w_clean = words_between(words, 1.00, 1.40, drop_fillers=True)
        self.assertEqual([x["word"] for x in w_clean], ["So", "let's", "go"])
        
        # Without filler removal
        w_all = words_between(words, 1.00, 1.40, drop_fillers=False)
        self.assertEqual([x["word"] for x in w_all], ["So", "um", "let's", "uh", "go"])
    
    def test_text_normalization(self):
        """Test text normalization and spacing fixes"""
        words = [
            {"word": "Hello", "start": 1.00, "end": 1.20},
            {"word": ",", "start": 1.21, "end": 1.22},
            {"word": "world", "start": 1.23, "end": 1.40},
            {"word": "!", "start": 1.41, "end": 1.42},
        ]
        text = words_to_text(words)
        self.assertEqual(text, "Hello, world!")
    
    def test_vtt_generation(self):
        """Test WebVTT generation"""
        caps = [
            {"t": 0.0, "d": 0.5, "w": "Hello"},
            {"t": 0.5, "d": 0.3, "w": "world"},
            {"t": 0.8, "d": 0.2, "w": "!"},
        ]
        vtt = captions_to_vtt(caps)
        
        # Check VTT format
        self.assertIn("WEBVTT", vtt)
        self.assertIn("00:00:00.000 --> 00:00:01.000", vtt)
        self.assertIn("Hello world!", vtt)
    
    def test_vtt_word_wrapping(self):
        """Test VTT word wrapping with line length limits"""
        # Create a long sequence of words
        caps = []
        for i in range(20):
            caps.append({"t": i * 0.1, "d": 0.1, "w": f"word{i}"})
        
        vtt = captions_to_vtt(caps, line_chars=20)
        
        # Should have multiple cues due to line length limit
        cue_count = vtt.count("-->")
        self.assertGreater(cue_count, 1)
    
    def test_empty_input_handling(self):
        """Test handling of empty word lists"""
        w = words_between([], 1.0, 2.0)
        self.assertEqual(w, [])
        
        text = words_to_text([])
        self.assertEqual(text, "")
        
        caps = words_to_captions([], 0.0)
        self.assertEqual(caps, [])
        
        vtt = captions_to_vtt([])
        self.assertEqual(vtt, "WEBVTT\n\n")
    
    def test_negative_times(self):
        """Test handling of negative relative times"""
        words = [
            {"word": "A", "start": 0.5, "end": 0.7},
            {"word": "B", "start": 1.0, "end": 1.2},
        ]
        caps = words_to_captions(words, clip_start=1.0)
        
        # First word should have negative time, which gets clamped to 0
        self.assertEqual(caps[0]["t"], 0.0)
        self.assertEqual(caps[1]["t"], 0.0)
    
    def test_normalize_text_fixes(self):
        """Test text normalization fixes"""
        test_cases = [
            ("Hello , world !", "Hello, world!"),
            ("Hello  world", "Hello world"),
            ("Hello ' world", "Hello ' world"),
            ("  Hello world  ", "Hello world"),
        ]
        
        for input_text, expected in test_cases:
            result = _normalize_text(input_text)
            self.assertEqual(result, expected)

def run_performance_test():
    """Test performance with large word lists"""
    import time
    
    # Create a large word list (simulating a long episode)
    words = []
    for i in range(10000):
        words.append({
            "word": f"word{i}",
            "start": i * 0.1,
            "end": i * 0.1 + 0.05
        })
    
    start_time = time.time()
    
    # Test slicing performance
    w = words_between(words, 100.0, 200.0)
    text = words_to_text(w)
    caps = words_to_captions(w, 100.0)
    vtt = captions_to_vtt(caps)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nPerformance test: Processed {len(words)} words in {duration:.3f} seconds")
    print(f"Extracted {len(w)} words, generated {len(caps)} captions")
    print(f"VTT length: {len(vtt)} characters")
    
    return duration

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    print("PERFORMANCE TEST")
    print("="*50)
    run_performance_test()
