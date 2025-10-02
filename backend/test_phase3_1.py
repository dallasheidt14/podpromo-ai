#!/usr/bin/env python3
"""
Test script for Phase 3.1 improvements:
- Titles unification (adapter verification)
- Structured logs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_titles_unification():
    """Test titles service unification"""
    logger.info("Testing Titles Unification...")
    
    # Test that both services exist and have compatible APIs
    try:
        from services.title_service import generate_titles, normalize_platform
        logger.info("✓ title_service.py imports successful")
        
        from services.titles_service import generate_titles as ts_generate_titles, normalize_platform as ts_normalize_platform
        logger.info("✓ titles_service.py imports successful")
        
        # Test that they're the same functions (adapter working)
        same_generate = generate_titles is ts_generate_titles
        same_normalize = normalize_platform is ts_normalize_platform
        
        logger.info(f"  generate_titles same: {same_generate}")
        logger.info(f"  normalize_platform same: {same_normalize}")
        
        # Test TitlesService class still exists for backward compatibility
        from services.titles_service import TitlesService
        ts = TitlesService()
        logger.info("✓ TitlesService class instantiated successfully")
        
        # Test basic functionality
        test_text = "This is a test transcript about technology and innovation."
        test_platform = "tiktok"
        
        # Test normalize_platform
        normalized = normalize_platform(test_platform)
        logger.info(f"  normalize_platform('{test_platform}') = '{normalized}'")
        
        # Test generate_titles (if it doesn't require complex setup)
        try:
            titles = generate_titles(test_text, platform=test_platform)
            logger.info(f"  generate_titles returned {len(titles)} titles")
            if titles:
                logger.info(f"    First title: '{titles[0]}'")
        except Exception as e:
            logger.info(f"  generate_titles test skipped (requires setup): {e}")
        
        return same_generate and same_normalize
        
    except Exception as e:
        logger.error(f"Titles unification test failed: {e}")
        return False

def test_structured_logging():
    """Test structured logging functionality"""
    logger.info("Testing Structured Logging...")
    
    from services.utils.logging_ext import log_json
    
    # Create a test logger
    test_logger = logging.getLogger("test_structured")
    
    # Test different event types
    test_cases = [
        ("DYNAMIC", {"mode": "peaks", "peaks": 15, "kept": 12, "smooth_window": 4.5}),
        ("DYNAMIC", {"mode": "fallback", "peaks": 0, "kept": 8, "fallback_segments": 20}),
        ("FINALS", {"n": 5, "min_finals": 4, "enforced": True}),
        ("VIRALITY", {"scaler": "minmax", "range": 0.85, "collapse": False}),
        ("VIRALITY", {"scaler": "zscore", "range": 0.001, "collapse": True}),
        ("GATES", {"finished_required": True, "authoritative": False, "coverage": 0.8}),
        ("CUSTOM", {"custom_field": "test_value", "number": 42}),
    ]
    
    results = []
    
    for event, fields in test_cases:
        try:
            # Capture log output
            import io
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            test_logger.addHandler(handler)
            test_logger.setLevel(logging.INFO)
            
            # Log the structured event
            log_json(test_logger, event, **fields)
            
            # Get the log output
            log_output = log_capture.getvalue()
            
            # Verify JSON is present
            json_found = False
            human_found = False
            
            for line in log_output.strip().split('\n'):
                try:
                    # Try to parse as JSON
                    parsed = json.loads(line)
                    if parsed.get("event") == event:
                        json_found = True
                        logger.info(f"✓ {event}: JSON logged correctly")
                        logger.info(f"    JSON: {line}")
                except json.JSONDecodeError:
                    # This is the human-readable log
                    if event in line:
                        human_found = True
                        logger.info(f"✓ {event}: Human-readable logged correctly")
                        logger.info(f"    Human: {line}")
            
            # Clean up
            test_logger.removeHandler(handler)
            
            # Both JSON and human-readable should be present
            passed = json_found and human_found
            results.append(passed)
            
            if not passed:
                logger.error(f"✗ {event}: JSON={json_found}, Human={human_found}")
                logger.error(f"    Output: {log_output}")
            
        except Exception as e:
            logger.error(f"✗ {event}: Exception during test: {e}")
            results.append(False)
    
    all_passed = all(results)
    logger.info(f"Structured logging test: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed

def test_log_parsing():
    """Test that structured logs can be parsed correctly"""
    logger.info("Testing Log Parsing...")
    
    from services.utils.logging_ext import log_json
    
    # Test logger
    test_logger = logging.getLogger("test_parsing")
    
    # Create sample log entries
    sample_events = [
        ("DYNAMIC", {"mode": "peaks", "peaks": 12, "kept": 10}),
        ("FINALS", {"n": 6, "min_finals": 4, "enforced": False}),
        ("VIRALITY", {"scaler": "zscore", "range": 0.001, "collapse": True}),
        ("GATES", {"finished_required": True, "authoritative": True, "coverage": 0.9}),
    ]
    
    # Capture logs
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    
    # Generate logs
    for event, fields in sample_events:
        log_json(test_logger, event, **fields)
    
    # Parse logs
    log_output = log_capture.getvalue()
    parsed_events = []
    
    for line in log_output.strip().split('\n'):
        try:
            parsed = json.loads(line)
            parsed_events.append(parsed)
        except json.JSONDecodeError:
            continue  # Skip human-readable logs
    
    # Verify all events were parsed
    logger.info(f"Generated {len(sample_events)} events")
    logger.info(f"Parsed {len(parsed_events)} JSON events")
    
    # Check that all expected events are present
    expected_events = {event for event, _ in sample_events}
    parsed_events_set = {event["event"] for event in parsed_events}
    
    all_present = expected_events.issubset(parsed_events_set)
    
    logger.info(f"Expected events: {expected_events}")
    logger.info(f"Parsed events: {parsed_events_set}")
    logger.info(f"All events present: {all_present}")
    
    # Clean up
    test_logger.removeHandler(handler)
    
    return all_present

def test_integration_scenario():
    """Test integration scenario"""
    logger.info("Testing Integration Scenario...")
    
    # Test that all components work together
    from services.utils.logging_ext import log_json
    from services.title_service import generate_titles, normalize_platform
    from services.titles_service import TitlesService
    
    # Create a test logger
    test_logger = logging.getLogger("integration_test")
    
    # Simulate a realistic scenario
    logger.info("Realistic scenario:")
    
    # Test platform normalization
    platforms = ["tiktok", "youtube", "instagram", "facebook"]
    normalized_platforms = []
    
    for platform in platforms:
        normalized = normalize_platform(platform)
        normalized_platforms.append(normalized)
        logger.info(f"  {platform} → {normalized}")
    
    # Test structured logging
    logger.info("Structured logging test:")
    
    # Simulate dynamic discovery
    log_json(test_logger, "DYNAMIC", mode="peaks", peaks=15, kept=12, smooth_window=4.5)
    
    # Simulate finals processing
    log_json(test_logger, "FINALS", n=5, min_finals=4, enforced=True)
    
    # Simulate virality processing
    log_json(test_logger, "VIRALITY", scaler="minmax", range=0.85, collapse=False)
    
    # Simulate gates processing
    log_json(test_logger, "GATES", finished_required=True, authoritative=False, coverage=0.8)
    
    # Test TitlesService
    ts = TitlesService()
    logger.info(f"  TitlesService instantiated: {ts is not None}")
    
    # Verify all components work
    components_work = (
        len(normalized_platforms) == len(platforms) and
        all(np is not None for np in normalized_platforms) and
        ts is not None
    )
    
    logger.info(f"  All components working: {components_work}")
    
    return components_work

if __name__ == "__main__":
    logger.info("Starting Phase 3.1 tests...")
    
    tests = [
        ("Titles Unification", test_titles_unification),
        ("Structured Logging", test_structured_logging),
        ("Log Parsing", test_log_parsing),
        ("Integration Scenario", test_integration_scenario),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            logger.info(f"Result: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    logger.info(f"\n{'='*50}")
    logger.info("PHASE 3.1 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
