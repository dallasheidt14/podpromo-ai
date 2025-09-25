# Regression Tests for Critical Fixes

This directory contains regression tests that guard against the specific failure modes we fixed in the ASR and clip scoring pipeline.

## Test Files

### `test_util_words_and_eos.py`
Tests the word coercion and EOS proximity logic:
- **`_coerce_words_list()`**: Safely handles None, strings, and mixed inputs
- **`_nearest_eos_after()`**: Finds nearest EOS after a given time
- **`calculate_finish_confidence()`**: Robust finish confidence with proximity boost

### `test_clip_score_telemetry.py`
Tests the telemetry density calculation:
- **`_telemetry_density()`**: Safely calculates word/EOS density without NoneType crashes

### `test_tail_snap_and_gates.py`
Tests the tail snap functionality:
- **`extend_to_natural_end()`**: Respects max extension limits and snaps to natural boundaries

### `test_finish_thresholds.py`
Tests the adaptive threshold system:
- **`finish_threshold_for()`**: Accepts indicators dict and handles missing data gracefully

## Running the Tests

### Option 1: Using the test runner script
```bash
cd backend
python run_regression_tests.py
```

### Option 2: Using pytest directly
```bash
cd backend
pytest -v tests/test_util_words_and_eos.py tests/test_clip_score_telemetry.py tests/test_tail_snap_and_gates.py tests/test_finish_thresholds.py
```

### Option 3: Run all tests
```bash
cd backend
pytest -q
```

## What These Tests Guarantee

✅ **No NoneType crashes**: All telemetry calculations handle None inputs safely  
✅ **No 'str'.get errors**: Finish confidence never sees text, only normalized word lists  
✅ **EOS proximity boost**: Monotonic boost when EOS is close, stays within [0,1]  
✅ **Tail snap limits**: Never extends more than max_extend_sec, snaps when natural end is reachable  
✅ **Adaptive thresholds**: Only consumes dict-like indicators, stable with missing stats  
✅ **Safe density logging**: Never divides by None or zero in a way that crashes  

## Test Coverage

These tests specifically cover the critical paths that were causing crashes:

1. **Word processing pipeline**: From raw ASR output to normalized word lists
2. **EOS detection**: Finding sentence boundaries and calculating proximity
3. **Finish confidence**: Combining multiple signals safely
4. **Telemetry logging**: Density calculations without crashes
5. **Tail extension**: Smart boundary refinement within limits
6. **Adaptive thresholds**: Genre-based finish detection

## Adding New Tests

When adding new functionality that touches these critical paths, add corresponding tests to ensure:
- Input validation (None, wrong types, empty data)
- Boundary conditions (zero counts, single items, edge cases)
- Error handling (graceful degradation, not crashes)
- Output validation (correct types, reasonable ranges)

## Dependencies

- `pytest`: For test framework
- `pytest.approx`: For floating-point comparisons
- Standard library modules: `math`, `sys`, `os`, `subprocess`
