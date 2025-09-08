@echo off
echo PodPromo AI - Mode Switcher
echo ===========================
echo.
echo 1. FAST MODE (5-10x faster, good quality)
echo 2. BALANCED MODE (Current - 70-85% speed gain, minimal quality risk)
echo 3. HIGH QUALITY MODE (Slower, best transcription accuracy)
echo 4. CUSTOM MODE (Manual settings)
echo.
set /p choice="Choose mode (1-4): "

if "%choice%"=="1" (
    echo Setting FAST MODE...
    set SCORING_PROCESS_POOL=1
    set SPEED_PRESET=fast
    set TOP_K_CANDIDATES=15
    echo ✅ FAST MODE ENABLED - 5-10x faster processing
) else if "%choice%"=="2" (
    echo Setting BALANCED MODE...
    set SCORING_PROCESS_POOL=1
    set SPEED_PRESET=balanced
    set TOP_K_CANDIDATES=15
    echo ✅ BALANCED MODE ENABLED - 70-85% speed gain, minimal quality risk
) else if "%choice%"=="3" (
    echo Setting HIGH QUALITY MODE...
    set SCORING_PROCESS_POOL=1
    set SPEED_PRESET=quality
    set TOP_K_CANDIDATES=20
    echo ✅ HIGH QUALITY MODE ENABLED - Best transcription accuracy
) else if "%choice%"=="4" (
    echo Custom mode - edit environment variables manually
    echo Current settings:
    echo SCORING_PROCESS_POOL=%SCORING_PROCESS_POOL%
    echo SPEED_PRESET=%SPEED_PRESET%
    echo TOP_K_CANDIDATES=%TOP_K_CANDIDATES%
    echo.
    echo Available SPEED_PRESET values:
    echo - fast: beam=1, best_of=1, no_condition_prev, skip_laughter
    echo - balanced: beam=2, best_of=2, condition_prev=true, full_features
    echo - quality: beam=4, best_of=5, condition_prev=true, full_features
) else (
    echo Invalid choice. Please run again.
)

echo.
echo Restart the backend to apply changes:
echo cd backend ^&^& python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
