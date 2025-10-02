"""
Test script to debug score display issue in frontend
"""
import json

# Load the clips JSON for the latest episode
episode_id = "2349627e-b003-4b49-9a68-38d97cbc1f9f"
clips_path = f"uploads/{episode_id}/clips.json"

try:
    with open(clips_path, 'r') as f:
        clips = json.load(f)
    
    print(f"✅ Loaded {len(clips)} clips from {clips_path}\n")
    
    # Handle both list and dict formats
    if isinstance(clips, dict):
        clips_list = list(clips.values()) if clips else []
    else:
        clips_list = clips
    
    print(f"Clips data type: {type(clips)}")
    print(f"First clip type: {type(clips_list[0]) if clips_list else 'N/A'}\n")
    
    for i, clip in enumerate(clips_list, 1):
        print(f"=== Clip {i} ===")
        if isinstance(clip, str):
            print(f"⚠️  Clip is a string: {clip[:100]}...")
            continue
        print(f"ID: {clip.get('id')}")
        print(f"Title: {clip.get('title', 'N/A')}")
        print(f"\nScores (top-level):")
        print(f"  hook_score: {clip.get('hook_score', 'MISSING')}")
        print(f"  arousal_score: {clip.get('arousal_score', 'MISSING')}")
        print(f"  emotion_score: {clip.get('emotion_score', 'MISSING')}")
        print(f"  payoff_score: {clip.get('payoff_score', 'MISSING')}")
        print(f"  q_list_score: {clip.get('q_list_score', 'MISSING')}")
        
        if 'features' in clip:
            print(f"\nScores (nested features):")
            features = clip['features']
            print(f"  hook_score: {features.get('hook_score', 'MISSING')}")
            print(f"  arousal_score: {features.get('arousal_score', 'MISSING')}")
            print(f"  emotion_score: {features.get('emotion_score', 'MISSING')}")
            print(f"  payoff_score: {features.get('payoff_score', 'MISSING')}")
            print(f"  q_list_score: {features.get('q_list_score', 'MISSING')}")
        
        print(f"\nOther fields:")
        print(f"  virality: {clip.get('virality', 'MISSING')}")
        print(f"  display_score: {clip.get('display_score', 'MISSING')}")
        print(f"  final_score: {clip.get('final_score', 'MISSING')}")
        print()
        
except FileNotFoundError:
    print(f"❌ File not found: {clips_path}")
except Exception as e:
    print(f"❌ Error: {e}")

