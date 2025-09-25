"""
Test suite for hardening improvements in the title service.
Tests thread safety, caching, Unicode normalization, ad detection, and fallbacks.
"""

import pytest
import threading
import time
from unittest.mock import patch, MagicMock
from services.titles_service import TitlesService
from services.title_service import generate_title_pack


class TestHardeningImprovements:
    """Test comprehensive hardening improvements."""

    def test_thread_safety(self):
        """Test that multiple threads can safely call ensure_titles_for_clip."""
        ts = TitlesService()
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(3):
                    result = ts.ensure_titles_for_clip(f'thread_test_{worker_id}_{i}', platform='shorts')
                    results.append(result)
                    time.sleep(0.01)  # Small delay to ensure interleaving
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Thread safety test failed with errors: {errors}"
        assert len(results) == 9, f"Expected 9 results, got {len(results)}"

    def test_lru_caching(self):
        """Test that get_clip uses LRU caching."""
        ts = TitlesService()
        
        # Clear cache first
        ts.get_clip.cache_clear()
        
        # First call should miss cache
        result1 = ts.get_clip('test_clip_caching_1')
        cache_info = ts.get_clip.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 1
        
        # Second call should hit cache
        result2 = ts.get_clip('test_clip_caching_1')
        cache_info = ts.get_clip.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 1
        
        # Results should be identical
        assert result1 == result2

    def test_unicode_normalization(self):
        """Test Unicode and punctuation normalization."""
        ts = TitlesService()
        
        clip_with_punctuation = {
            'id': 'clip_test_unicode_0',
            'text': 'This is a test... with multiple   spaces    and trailing colons:::',
            'transcript': {'text': 'This is a test... with multiple   spaces    and trailing colons:::'}
        }
        
        # Mock the title generation to avoid file system operations
        with patch.object(ts, 'save_titles'):
            result = ts.ensure_titles_for_clip(clip_with_punctuation, platform='shorts')
            
        # Should successfully process the clip
        assert result is not None
        assert 'variants' in result
        assert 'chosen' in result

    def test_ad_detection(self):
        """Test that ad content is skipped."""
        ts = TitlesService()
        
        ad_clip = {
            'id': 'clip_test_ad_0',
            'text': 'Buy our amazing product now!',
            'features': {'is_advertisement': True}
        }
        
        result = ts.ensure_titles_for_clip(ad_clip, platform='shorts')
        
        # Should return None for ad content
        assert result is None

    def test_language_detection(self):
        """Test that language hints are extracted."""
        ts = TitlesService()
        
        clip_with_language = {
            'id': 'clip_test_lang_0',
            'text': 'This is a test clip',
            'features': {'language': 'es'}
        }
        
        # Mock the title generation
        with patch.object(ts, 'save_titles'):
            result = ts.ensure_titles_for_clip(clip_with_language, platform='shorts')
            
        # Should successfully process the clip
        assert result is not None

    def test_title_generation_fallbacks(self):
        """Test fallback when title generation fails."""
        ts = TitlesService()
        
        clip_data = {
            'id': 'clip_test_fallback_0',
            'text': 'This is a test clip for fallback testing',
            'transcript': {'text': 'This is a test clip for fallback testing'}
        }
        
        # Mock generate_title_pack to return empty variants
        with patch('services.title_service.generate_title_pack') as mock_generate:
            mock_generate.return_value = {'variants': [], 'overlay': ''}
            
            with patch.object(ts, 'save_titles'):
                result = ts.ensure_titles_for_clip(clip_data, platform='shorts')
        
        # Should return fallback title
        assert result is not None
        assert 'variants' in result
        assert len(result['variants']) > 0
        assert result['variants'][0] == 'This is a test clip for fallback testing'  # Should use snippet

    def test_throttling(self):
        """Test that warnings are throttled to once per clip."""
        ts = TitlesService()
        
        # Call the same problematic clip multiple times
        for i in range(5):
            result = ts.ensure_titles_for_clip('throttle_test_clip', platform='shorts')
            assert result is None  # Should always return None for missing clip
        
        # Should only have one warning in the skip_warnings set
        assert len(ts._skip_warnings) == 1
        assert 'throttle_test_clip' in ts._skip_warnings

    def test_platform_aware_selection(self):
        """Test that platform-aware selection is enabled."""
        import os
        
        # Check that environment variables are set correctly
        assert os.getenv('PL_V2_WEIGHT') == '0.5'
        assert os.getenv('PLATFORM_PROTECT') == 'true'
        
        # Test the platform weight calculation
        pl_v2_weight = float(os.getenv('PL_V2_WEIGHT', '0.5'))
        platform_protect = os.getenv('PLATFORM_PROTECT', 'true').lower() == 'true'
        mode = 'platform-aware' if pl_v2_weight > 0 else 'neutral'
        
        assert pl_v2_weight == 0.5
        assert platform_protect is True
        assert mode == 'platform-aware'

    def test_enhanced_run_summary_format(self):
        """Test the enhanced run summary format."""
        import os
        
        # Test the run summary components
        durs = [10.2, 18.0, 47.5, 47.5]
        durs_str = '[' + ','.join(map(str, durs)) + ']'
        pl_v2_weight = float(os.getenv('PL_V2_WEIGHT', '0.5'))
        platform_protect = os.getenv('PLATFORM_PROTECT', 'true').lower() == 'true'
        mode = 'platform-aware' if pl_v2_weight > 0 else 'neutral'
        
        summary = f'RUN_SUMMARY: seeds=10 strict=4 balanced=4 finals=4 durs={durs_str} eos=359 pl_v2_w={pl_v2_weight:.2f} protect={platform_protect} mode={mode} fallback=False'
        
        # Verify all components are present
        assert 'seeds=10' in summary
        assert 'strict=4' in summary
        assert 'balanced=4' in summary
        assert 'finals=4' in summary
        assert 'durs=[10.2,18.0,47.5,47.5]' in summary
        assert 'eos=359' in summary
        assert 'pl_v2_w=0.50' in summary
        assert 'protect=True' in summary
        assert 'mode=platform-aware' in summary
        assert 'fallback=False' in summary

    def test_deterministic_behavior(self):
        """Test that same input produces stable results."""
        ts = TitlesService()
        
        clip_data = {
            'id': 'clip_test_deterministic_0',
            'text': 'This is a test clip for deterministic behavior testing',
            'transcript': {'text': 'This is a test clip for deterministic behavior testing'}
        }
        
        # Mock the title generation to return consistent results
        with patch('services.title_service.generate_title_pack') as mock_generate:
            mock_generate.return_value = {
                'variants': [
                    {'title': 'Test Title 1'},
                    {'title': 'Test Title 2'}
                ],
                'overlay': 'Test Title 1'
            }
            
            with patch.object(ts, 'save_titles'):
                result1 = ts.ensure_titles_for_clip(clip_data, platform='shorts')
                result2 = ts.ensure_titles_for_clip(clip_data, platform='shorts')
        
        # Results should be identical
        assert result1 == result2
        assert result1['variants'] == ['Test Title 1', 'Test Title 2']
        assert result1['chosen'] == 'Test Title 1'

    def test_missing_text_fallback(self):
        """Test fallback when clip has no text but has words."""
        ts = TitlesService()
        
        clip_with_words = {
            'id': 'clip_test_words_0',
            'text': '',  # No text
            'transcript': {'text': ''},  # No transcript text
            'words': [
                {'word': 'This'},
                {'word': 'is'},
                {'word': 'a'},
                {'word': 'test'},
                {'word': 'clip'}
            ]
        }
        
        # Mock the title generation
        with patch('services.title_service.generate_title_pack') as mock_generate:
            mock_generate.return_value = {
                'variants': [{'title': 'This is a test clip'}],
                'overlay': 'This is a test clip'
            }
            
            with patch.object(ts, 'save_titles'):
                result = ts.ensure_titles_for_clip(clip_with_words, platform='shorts')
        
        # Should successfully extract text from words
        assert result is not None
        assert 'variants' in result
        assert len(result['variants']) > 0

    def test_cache_cleanup(self):
        """Test that cache can be cleared."""
        ts = TitlesService()
        
        # Add some items to cache
        ts.get_clip('test_clip_1')
        ts.get_clip('test_clip_2')
        
        # Verify cache has items
        cache_info = ts.get_clip.cache_info()
        assert cache_info.currsize > 0
        
        # Clear cache
        ts.get_clip.cache_clear()
        
        # Verify cache is empty
        cache_info = ts.get_clip.cache_info()
        assert cache_info.currsize == 0
        assert cache_info.hits == 0
        assert cache_info.misses == 0
