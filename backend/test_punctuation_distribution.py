#!/usr/bin/env python3
"""Test punctuation distribution and context features"""

from services.transcript_builder import _distribute_punct_to_words
from services.secret_sauce_pkg.features import build_sentence_spans, _is_strong_opening, _has_complete_payoff

# Test with sample transcript data (similar to the actual format)
segments = [
    {
        'text': 'Ever watched a webinar or video that just seems to ramble on?',
        'start': 0.0,
        'end': 5.0,
        'words': [
            {'text': 'Ever', 'start': 0.0, 'end': 0.5, 'prob': 0.9},
            {'text': 'watched', 'start': 0.5, 'end': 1.0, 'prob': 0.9},
            {'text': 'a', 'start': 1.0, 'end': 1.2, 'prob': 0.9},
            {'text': 'webinar', 'start': 1.2, 'end': 1.8, 'prob': 0.9},
            {'text': 'or', 'start': 1.8, 'end': 2.0, 'prob': 0.9},
            {'text': 'video', 'start': 2.0, 'end': 2.5, 'prob': 0.9},
            {'text': 'that', 'start': 2.5, 'end': 2.8, 'prob': 0.9},
            {'text': 'just', 'start': 2.8, 'end': 3.1, 'prob': 0.9},
            {'text': 'seems', 'start': 3.1, 'end': 3.5, 'prob': 0.9},
            {'text': 'to', 'start': 3.5, 'end': 3.7, 'prob': 0.9},
            {'text': 'ramble', 'start': 3.7, 'end': 4.2, 'prob': 0.9},
            {'text': 'on', 'start': 4.2, 'end': 4.5, 'prob': 0.9}
        ]
    },
    {
        'text': 'You end up confused or zoning out, right?',
        'start': 5.0,
        'end': 8.0,
        'words': [
            {'text': 'You', 'start': 5.0, 'end': 5.3, 'prob': 0.9},
            {'text': 'end', 'start': 5.3, 'end': 5.6, 'prob': 0.9},
            {'text': 'up', 'start': 5.6, 'end': 5.8, 'prob': 0.9},
            {'text': 'confused', 'start': 5.8, 'end': 6.3, 'prob': 0.9},
            {'text': 'or', 'start': 6.3, 'end': 6.5, 'prob': 0.9},
            {'text': 'zoning', 'start': 6.5, 'end': 6.9, 'prob': 0.9},
            {'text': 'out', 'start': 6.9, 'end': 7.2, 'prob': 0.9},
            {'text': 'right', 'start': 7.2, 'end': 7.6, 'prob': 0.9}
        ]
    }
]

print("Testing punctuation distribution...")
words_with_punct = _distribute_punct_to_words(segments)
print(f"Distributed punctuation to {len(words_with_punct)} words")

print("\nWords with punctuation:")
for i, word in enumerate(words_with_punct):
    print(f"  {i}: '{word['text']}' (after: '{word['after']}')")

print("\nTesting sentence span building...")
spans = build_sentence_spans(words_with_punct)
print(f"Built {len(spans)} sentence spans:")

for i, span in enumerate(spans):
    print(f"  {i}: '{span.text}' ({span.start:.1f}s-{span.end:.1f}s)")
    print(f"    Strong opening: {_is_strong_opening(span.text)}")
    print(f"    Complete payoff: {_has_complete_payoff(span.text)}")
