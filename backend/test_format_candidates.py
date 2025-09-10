#!/usr/bin/env python3
"""Test format_candidates function"""

from services.secret_sauce_pkg.features import find_viral_clips_enhanced
from services.candidate_formatter import format_candidates
import os

# Create test segments
segments = [
    {
        'text': 'they don\'t disable And disarm you and remove your superpower as a leader and you lose your leadership impact So the go and do is this is to make sure that you have The systems in place to keep you',
        'start': 51.36,
        'end': 63.36
    }
]

# Get a real audio file
audio_files = [f for f in os.listdir('uploads') if f.lower().endswith('.mp3')]
if audio_files:
    test_audio = f'uploads/{audio_files[0]}'
    
    # Get clips from find_viral_clips_enhanced
    result = find_viral_clips_enhanced(segments, test_audio, genre='general', platform='tiktok')
    clips = result.get('clips', [])
    
    print(f'Input clips: {len(clips)}')
    if clips:
        first_clip = clips[0]
        hook_score = first_clip.get('hook_score', 'NOT FOUND')
        arousal_score = first_clip.get('arousal_score', 'NOT FOUND')
        print(f'Input clip hook_score: {hook_score}')
        print(f'Input clip arousal_score: {arousal_score}')
    
    # Test format_candidates
    candidates = format_candidates(clips, 'general', 'tiktok', 'test_episode')
    
    print(f'\nFormatted candidates: {len(candidates)}')
    if candidates:
        first_candidate = candidates[0]
        print(f'\nFormatted candidate keys: {list(first_candidate.keys())}')
        
        # Check individual scores
        for key in ['hook_score', 'arousal_score', 'emotion_score', 'payoff_score']:
            value = first_candidate.get(key, 'NOT FOUND')
            print(f'  {key}: {value}')
            
        # Check features object
        features = first_candidate.get('features', {})
        print(f'\nFeatures object: {features}')
        
        if features:
            print(f'\nFeatures hook_score: {features.get("hook_score", "NOT FOUND")}')
            print(f'Features arousal_score: {features.get("arousal_score", "NOT FOUND")}')
        else:
            print('Features object is empty!')
