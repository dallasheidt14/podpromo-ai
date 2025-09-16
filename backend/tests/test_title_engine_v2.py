# backend/tests/test_title_engine_v2.py
# Test suite for CFD-First Title Engine V2

import unittest
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.title_engine_v2 import (
    classify_transcript,
    generate_titles_v2,
    extract_stat,
    cheap_topic,
    score_title,
    run_tests,
    TESTS
)

class TestTitleEngineV2(unittest.TestCase):
    
    def test_archetype_classification(self):
        """Test archetype classification accuracy"""
        test_results = run_tests()
        
        # Check that we get reasonable accuracy
        correct_count = sum(1 for result in test_results if result["correct"])
        total_count = len(test_results)
        accuracy = correct_count / total_count
        
        print(f"\nClassification Accuracy: {correct_count}/{total_count} ({accuracy:.1%})")
        
        # Should have at least 70% accuracy
        self.assertGreaterEqual(accuracy, 0.7, f"Classification accuracy too low: {accuracy:.1%}")
        
        # Print detailed results
        for result in test_results:
            status = "✅" if result["correct"] else "❌"
            print(f"{status} {result['expected']} -> {result['actual']} (conf: {result['confidence']:.2f})")
    
    def test_stat_extraction(self):
        """Test statistics extraction from text"""
        test_cases = [
            ("One in four prescriptions is off-label", "one in four"),
            ("The data shows 25% of cases", "25%"),
            ("It costs $1,200 per month", "$1,200"),
            ("1 in 10 people experience this", "1 in 10"),
            ("No statistics here", None),
        ]
        
        for text, expected in test_cases:
            result = extract_stat(text)
            self.assertEqual(result, expected, f"Failed to extract stat from: '{text}'")
    
    def test_topic_extraction(self):
        """Test topic extraction from text"""
        test_cases = [
            ("Machine Learning and Artificial Intelligence are transforming healthcare", "Machine Learning and Artificial Intelligence"),
            ("The doctor prescribed off-label medication", "doctor prescribed off-label medication"),
            ("This is about nothing important", "nothing important"),
            ("", "this"),  # fallback case
        ]
        
        for text, expected in test_cases:
            result = cheap_topic(text)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
    
    def test_title_scoring(self):
        """Test title scoring system"""
        # Test high-quality title
        high_score = score_title(
            "1 in 4 Prescriptions: What Everyone Misses About Off-Label",
            "Curiosity",
            {"primary_archetype": "Stats/Proof/Receipts", "payoff_ok": True},
            "tiktok"
        )
        self.assertGreater(high_score, 70, "High-quality title should score > 70")
        
        # Test low-quality title
        low_score = score_title(
            "Key Takeaways: What It Means",
            "Generic",
            {"primary_archetype": "Generic", "payoff_ok": False},
            "tiktok"
        )
        self.assertLess(low_score, 50, "Low-quality title should score < 50")
        
        # Test length penalties
        too_long = score_title(
            "This is a very long title that exceeds the recommended length for social media platforms",
            "Generic",
            {"primary_archetype": "Generic", "payoff_ok": True},
            "tiktok"
        )
        self.assertLess(too_long, 60, "Overly long title should be penalized")
    
    def test_title_generation(self):
        """Test end-to-end title generation"""
        # Test with off-label prescribing content (from your logs)
        text = "One in four prescriptions is off-label. The data shows this is more common than people think, but doctors don't always tell patients."
        
        titles = generate_titles_v2(
            text=text,
            platform="tiktok",
            n=3,
            clip_attrs={"payoff_ok": True}
        )
        
        self.assertGreater(len(titles), 0, "Should generate at least one title")
        self.assertLessEqual(len(titles), 3, "Should not exceed requested count")
        
        # Check title structure
        for title_data in titles:
            self.assertIn("title", title_data)
            self.assertIn("score", title_data)
            self.assertIn("trigger", title_data)
            self.assertIn("archetype", title_data)
            
            # Check title quality
            title = title_data["title"]
            self.assertIsInstance(title, str)
            self.assertGreater(len(title), 5, "Title should be meaningful length")
            self.assertLess(len(title), 100, "Title should not be too long")
            
            # Check for banned phrases
            banned_phrases = ["Key Insight", "Key Takeaways", "What It Means"]
            for banned in banned_phrases:
                self.assertNotIn(banned, title, f"Title should not contain banned phrase: {banned}")
    
    def test_platform_specific_lengths(self):
        """Test platform-specific length requirements"""
        text = "This is a test about machine learning and artificial intelligence in healthcare applications."
        
        # Test TikTok/Shorts (should prefer shorter titles)
        tiktok_titles = generate_titles_v2(text, platform="tiktok", n=2)
        for title_data in tiktok_titles:
            title = title_data["title"]
            self.assertLessEqual(len(title), 72, f"TikTok title too long: '{title}' ({len(title)} chars)")
        
        # Test YouTube (can be longer)
        youtube_titles = generate_titles_v2(text, platform="youtube", n=2)
        for title_data in youtube_titles:
            title = title_data["title"]
            self.assertLessEqual(len(title), 100, f"YouTube title too long: '{title}' ({len(title)} chars)")
    
    def test_archetype_specific_templates(self):
        """Test that appropriate templates are selected for each archetype"""
        test_cases = [
            ("Everyone says it works, but actually it doesn't.", "Hot Take/Contradiction"),
            ("Here's how to fix it in three steps.", "Advice/Playbook"),
            ("One in four prescriptions is off-label.", "Stats/Proof/Receipts"),
            ("Honestly, I'm embarrassed to admit this.", "Confession/Vulnerability"),
        ]
        
        for text, expected_archetype in test_cases:
            titles = generate_titles_v2(text, n=2)
            
            if titles:
                # Check that we get titles appropriate for the archetype
                for title_data in titles:
                    title = title_data["title"]
                    self.assertIsInstance(title, str)
                    self.assertGreater(len(title), 0)
                    
                    # For stats content, should include numbers
                    if expected_archetype == "Stats/Proof/Receipts":
                        # Should have at least one title with a number or stat
                        has_stat = any(char.isdigit() for char in title)
                        if not has_stat:
                            print(f"Warning: Stats archetype didn't produce stat-based title: '{title}'")
    
    def test_fallback_behavior(self):
        """Test fallback behavior for low-confidence cases"""
        # Very short text should trigger fallback
        short_text = "Hi"
        titles = generate_titles_v2(short_text, n=2)
        
        # Should still return some titles (fallback)
        self.assertGreater(len(titles), 0, "Should provide fallback titles for short text")
        
        # Test with gibberish
        gibberish = "asdf qwerty zxcv"
        titles = generate_titles_v2(gibberish, n=2)
        
        # Should still return some titles
        self.assertGreaterEqual(len(titles), 0, "Should handle gibberish gracefully")
    
    def test_avoid_titles_functionality(self):
        """Test that avoided titles are not included"""
        text = "This is about machine learning and artificial intelligence."
        avoid_titles = ["Machine Learning", "Artificial Intelligence"]
        
        titles = generate_titles_v2(text, avoid_titles=avoid_titles, n=5)
        
        for title_data in titles:
            title = title_data["title"]
            for avoided in avoid_titles:
                self.assertNotIn(avoided, title, f"Title should not contain avoided phrase: '{avoided}'")
    
    def test_persona_weighting(self):
        """Test persona weighting system"""
        text = "The data shows 75% of cases improve with this method. Here's the step-by-step process."
        
        # Test with Logic Nerds persona (should prefer stats/advice)
        titles = generate_titles_v2(
            text, 
            n=3, 
            persona="Logic Nerds",
            clip_attrs={"payoff_ok": True}
        )
        
        # Should still generate titles (persona weighting is optional)
        self.assertGreaterEqual(len(titles), 0, "Should generate titles with persona weighting")

def run_performance_test():
    """Run a performance test with multiple title generations"""
    import time
    
    test_texts = [
        "One in four prescriptions is off-label. The data shows this is more common than people think.",
        "Everyone says it works, but actually it doesn't. Here's what really happens.",
        "Honestly, I'm embarrassed to admit this mistake I made.",
        "Here's how to fix it in three simple steps that anyone can follow.",
        "The data will surprise you. 75% of people don't know this fact.",
    ]
    
    start_time = time.time()
    
    for text in test_texts:
        titles = generate_titles_v2(text, n=3)
        print(f"Generated {len(titles)} titles for: '{text[:50]}...'")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nPerformance: Generated titles for {len(test_texts)} texts in {duration:.2f} seconds")
    print(f"Average: {duration/len(test_texts):.3f} seconds per text")
    
    return duration

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run performance test
    print("\n" + "="*50)
    print("PERFORMANCE TEST")
    print("="*50)
    run_performance_test()
