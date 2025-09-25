# backend/tests/test_titles_api_compat.py
# Tests for title service API compatibility

import pytest
from services.title_service import generate_title_pack, generate_title_pack_v2


class TestTitleServiceAPICompatibility:
    """Test that both v1 and v2 APIs work correctly"""
    
    def test_generate_title_pack_v1_signature(self):
        """Test that the old v1 signature (text, platform) still works"""
        pack = generate_title_pack("hello world", "shorts")
        
        assert isinstance(pack, dict)
        assert "variants" in pack
        assert isinstance(pack["variants"], list)
        assert len(pack["variants"]) > 0
        
        # Check that all expected fields are present
        assert "overlay" in pack
        assert "engine" in pack
        assert "generated_at" in pack
        assert "version" in pack
        assert "meta" in pack
        
        # Check that variants have the expected structure
        for variant in pack["variants"]:
            assert "title" in variant
            assert "style" in variant
            assert "length" in variant
            assert isinstance(variant["title"], str)
            assert isinstance(variant["style"], str)
            assert isinstance(variant["length"], int)
    
    def test_generate_title_pack_v1_with_episode_text(self):
        """Test that the old v1 signature with episode_text works"""
        pack = generate_title_pack("hello world", "shorts", episode_text="This is a longer episode about hello world")
        
        assert isinstance(pack, dict)
        assert "variants" in pack
        assert len(pack["variants"]) > 0
    
    def test_generate_title_pack_v2_signature(self):
        """Test that the new v2 signature (clip_id, platform, text) works"""
        pack = generate_title_pack_v2("clip_123", "shorts", "hello world")
        
        assert isinstance(pack, dict)
        assert "variants" in pack
        assert isinstance(pack["variants"], list)
        assert len(pack["variants"]) > 0
        
        # Check that all expected fields are present
        assert "overlay" in pack
        assert "engine" in pack
        assert "generated_at" in pack
        assert "version" in pack
        assert "meta" in pack
    
    def test_generate_title_pack_v2_with_episode_text(self):
        """Test that the new v2 signature with episode_text works"""
        pack = generate_title_pack_v2("clip_123", "shorts", "hello world", "This is a longer episode about hello world")
        
        assert isinstance(pack, dict)
        assert "variants" in pack
        assert len(pack["variants"]) > 0
    
    def test_v1_and_v2_produce_same_results(self):
        """Test that v1 and v2 produce equivalent results for the same input"""
        text = "Machine learning is amazing"
        platform = "shorts"
        
        # v1 call (with empty clip_id internally)
        pack_v1 = generate_title_pack(text, platform)
        
        # v2 call (with empty clip_id explicitly)
        pack_v2 = generate_title_pack_v2("", platform, text)
        
        # Should produce equivalent results (except for timestamps)
        assert pack_v1["variants"] == pack_v2["variants"]
        assert pack_v1["overlay"] == pack_v2["overlay"]
        assert pack_v1["engine"] == pack_v2["engine"]
        assert pack_v1["version"] == pack_v2["version"]
        assert pack_v1["meta"]["keywords"] == pack_v2["meta"]["keywords"]
    
    def test_v2_with_real_clip_id(self):
        """Test that v2 works with a real clip_id for better caching"""
        clip_id = "real_clip_123"
        platform = "shorts"
        text = "Machine learning is amazing"
        
        pack = generate_title_pack_v2(clip_id, platform, text)
        
        assert isinstance(pack, dict)
        assert "variants" in pack
        assert len(pack["variants"]) > 0
        
        # Should have analytics placeholders
        assert "analytics" in pack["meta"]
        analytics = pack["meta"]["analytics"]
        assert "impressions" in analytics
        assert "clicks" in analytics
        assert "ctr" in analytics
    
    def test_backward_compatibility_regression_guard(self):
        """Guard test to ensure we don't break existing callers"""
        # This test simulates how existing code calls the function
        # If this fails, we've broken backward compatibility
        
        # Simulate TitlesService.save_titles() call
        text = "This is a test title"
        platform = "shorts"
        
        # This should not raise TypeError
        try:
            pack = generate_title_pack(text, platform)
            assert isinstance(pack, dict)
            assert "variants" in pack
        except TypeError as e:
            pytest.fail(f"Backward compatibility broken: {e}")
    
    def test_error_handling(self):
        """Test that both APIs handle errors gracefully"""
        # Test with empty text
        pack = generate_title_pack("", "shorts")
        assert isinstance(pack, dict)
        
        pack = generate_title_pack_v2("clip_123", "shorts", "")
        assert isinstance(pack, dict)
        
        # Test with None text
        pack = generate_title_pack(None, "shorts")
        assert isinstance(pack, dict)
        
        pack = generate_title_pack_v2("clip_123", "shorts", None)
        assert isinstance(pack, dict)


if __name__ == "__main__":
    pytest.main([__file__])
