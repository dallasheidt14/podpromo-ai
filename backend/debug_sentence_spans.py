#!/usr/bin/env python3
"""Debug sentence span building"""

import re
from services.transcript_builder import _distribute_punct_to_words

# Test with sample transcript data
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
    }
]

print("Testing punctuation distribution...")
words_with_punct = _distribute_punct_to_words(segments)

print("\nWords with punctuation:")
for i, word in enumerate(words_with_punct):
    print(f"  {i}: '{word['text']}' (after: '{word['after']}')")

# Test the regex directly
_SENT_END = re.compile(r'[.!?]["\']?$')
print(f"\nTesting regex on 'on?': {_SENT_END.search('on?')}")
print(f"Testing regex on '?': {_SENT_END.search('?')}")

# Test the sentence span building logic manually
print("\nTesting sentence span building logic...")
spans = []
buf = []
start_t = None
last_end = None
idx = 0

for w in words_with_punct:
    if start_t is None:
        start_t = w['start']
    buf.append(w.get('text', ''))
    after = (w.get('after') or '')
    flush = False

    # 1) Prefer punctuation
    if _SENT_END.search(after or w.get('text', '')):
        flush = True
        print(f"  Flushing due to punctuation: '{after}' in word '{w.get('text', '')}'")

    # 2) Fallback: long pause
    if not flush and last_end is not None:
        gap_ms = int((w['start'] - last_end) * 1000)
        if gap_ms >= 400:  # SENT_PAUSE_FALLBACK_MS
            flush = True
            print(f"  Flushing due to pause: {gap_ms}ms gap")

    if flush:
        text = " ".join(buf).strip()
        spans.append({'start': start_t, 'end': w['end'], 'text': text, 'idx': idx})
        idx += 1
        buf, start_t = [], None
        print(f"  Created span: '{text}'")

    last_end = w['end']

# Handle remaining buffer
if buf:
    text = " ".join(buf).strip()
    spans.append({'start': start_t or words_with_punct[0]['start'], 'end': words_with_punct[-1]['end'], 'text': text, 'idx': idx})

print(f"\nFinal spans: {len(spans)}")
for span in spans:
    print(f"  {span['idx']}: '{span['text']}' ({span['start']:.1f}s-{span['end']:.1f}s)")
