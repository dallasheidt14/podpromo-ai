"""
ViralMomentDetector - Intelligent content-based segmentation for viral clips
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class ViralMomentDetector:
    def __init__(self, genre='general'):
        self.genre = genre
        self.moment_patterns = self._get_genre_patterns(genre)
        
        # Performance limits - tightened significantly
        self.MAX_STORY_DURATION = 30  # Reduced from 60
        self.MAX_INSIGHT_DURATION = 25  # Reduced from 45
        self.MAX_LOOKAHEAD = 15  # Reduced from 20
        
    def _get_genre_patterns(self, genre: str) -> Dict:
        """Get genre-specific moment detection patterns"""
        base_patterns = {
            'story': {
                'starts': [
                    r"(?:so |and )?(?:one time|I remember when|there was this|back when)",
                    r"(?:let me tell you about|I'll never forget|the craziest thing)",
                    r"(?:true story|no joke|I kid you not)",
                    r"(?:this happened|I was|we were)",
                    r"(?:picture this|imagine|think about)"
                ],
                'ends': [
                    r"(?:and that's|so that's|which is) (?:how|why|when)",
                    r"(?:ever since|from that|after that)",
                    r"(?:learned|realized|understood) that",
                    r"(?:moral of the story|the lesson|the point)",
                    r"(?:that's why|this is why|which explains)"
                ],
                'min_duration': 15,  # Reduced from 20
                'max_duration': 30   # Reduced from 60
            },
            'insight': {
                'markers': [
                    r"here's (?:the thing|what people don't understand|the secret)",
                    r"the (?:truth|reality|fact) is",
                    r"what (?:nobody talks about|everyone misses|people forget)",
                    r"the (?:key|trick|secret) (?:is|to)",
                    r"here's what (?:I learned|changed everything|matters)",
                    r"the (?:biggest|most important) (?:lesson|takeaway|insight)",
                    r"if you want to (?:succeed|win|improve), you need to"
                ],
                'min_duration': 10,  # Reduced from 15
                'max_duration': 25   # Reduced from 45
            },
            'hot_take': {
                'markers': [
                    r"(?:unpopular opinion|hot take|controversial but)",
                    r"(?:everyone thinks|people say) .+ but (?:actually|really)",
                    r"I (?:don't care|disagree|think differently)",
                    r"(?:overrated|underrated|overhyped)",
                    r"(?:the real MVP|the actual problem|the truth about)",
                    r"(?:nobody wants to hear this|this might be controversial)",
                    r"(?:I'm going to get hate for this|prepare for the downvotes)"
                ],
                'min_duration': 8,   # Reduced from 10
                'max_duration': 20   # Reduced from 30
            }
        }
        
        # Genre-specific pattern additions
        genre_specific = {
            'comedy': {
                'story': {
                    'starts': base_patterns['story']['starts'] + [
                        r'so this one time', r'funniest thing', r"you won't believe",
                        r'this is hilarious', r'craziest story'
                    ]
                },
                'hot_take': {
                    'markers': base_patterns['hot_take']['markers'] + [
                        r'this is actually funny', r"people don't get the joke"
                    ]
                }
            },
            'sports': {
                'hot_take': {
                    'markers': base_patterns['hot_take']['markers'] + [
                        r'overrated', r'underrated', r'the real MVP',
                        r'this player is', r'this team is', r"everyone's sleeping on"
                    ]
                },
                'insight': {
                    'markers': base_patterns['insight']['markers'] + [
                        r'the key to winning', r'this is why they lost',
                        r"the stats don't lie", r"here's what the film shows"
                    ]
                }
            },
            'fantasy_sports': {
                'insight': {
                    'markers': base_patterns['insight']['markers'] + [
                        r'sleeper pick', r'league winner', r'waiver wire gem',
                        r"this week's", r'start him', r'sit him', r'buy low'
                    ]
                },
                'hot_take': {
                    'markers': base_patterns['hot_take']['markers'] + [
                        r"nobody's talking about", r"everyone's wrong about",
                        r'this is the week', r'trust me on this'
                    ]
                }
            },
            'business': {
                'insight': {
                    'markers': base_patterns['insight']['markers'] + [
                        r"here's what I learned", r'the biggest mistake',
                        r'this changed everything', r'the key to success',
                        r'if you want to scale', r'this is why companies fail'
                    ]
                },
                'story': {
                    'starts': base_patterns['story']['starts'] + [
                        r'when I started', r'back in the day', r'my first company',
                        r'this entrepreneur', r'this business'
                    ]
                }
            }
        }
        
        if genre in genre_specific:
            # Merge genre-specific with base patterns
            merged = {}
            for moment_type in base_patterns:
                if moment_type in genre_specific[genre]:
                    merged[moment_type] = {
                        **base_patterns[moment_type],
                        **genre_specific[genre][moment_type]
                    }
                else:
                    merged[moment_type] = base_patterns[moment_type]
            return merged
        
        return base_patterns
    
    def find_moments(self, transcript: List[Dict]) -> List[Dict]:
        """Find potential viral moments based on content patterns"""
        try:
            moments = []
            
            # Find stories - now much more selective
            stories = self._find_complete_stories(transcript)
            moments.extend(stories)
            
            # Find insights - tightened duration
            insights = self._find_insights(transcript)
            moments.extend(insights)
            
            # Find hot takes - tightened duration
            hot_takes = self._find_hot_takes(transcript)
            moments.extend(hot_takes)
            
            # Remove overlaps, keeping highest quality
            deduplicated = self._deduplicate_moments(moments)
            
            logger.info(f"Found {len(moments)} moments, {len(deduplicated)} after deduplication")
            return deduplicated
            
        except Exception as e:
            logger.error(f"Moment detection failed: {e}")
            return []
    
    def _find_complete_stories(self, transcript: List[Dict]) -> List[Dict]:
        """Find complete story arcs - now much more selective"""
        stories = []
        
        # Strong story start patterns - much more specific
        strong_story_starts = [
            r"^(?:so |and )?(?:one time|I'll never forget|craziest thing|funniest thing)",
            r"^(?:so |and )?one time\b",  # "one time" at start (most common)
            r"^(?:let me tell you about|here's what happened|this actually happened)",
            r"^(?:true story|no joke|I kid you not|this is hilarious)",
            r"^(?:picture this|imagine|think about|you won't believe)",
            r"^(?:back when|there was this|I remember when|this happened)",
            r"^(?:so this one time|the craziest story|this is wild|this is nuts)",
            r"^(?:so |and )?I was (?:watching|at|in|doing)",
            r"^(?:so |and )?there was (?:this|a|an)",
            r"^(?:so |and )?this (?:happened|was|is)"
        ]
        
        # Find all story beginnings with strong openings
        for i, segment in enumerate(transcript):
            if not self._has_strong_opening(segment.get('text', ''), strong_story_starts):
                continue
            
            # Look for story arc within max_duration=30 seconds
            story = self._extract_story_arc(transcript, i, max_duration=30)
            
            if story and self._is_self_contained(story['text']):
                stories.append(story)
        
        return stories
    
    def _has_strong_opening(self, text: str, strong_patterns: List[str]) -> bool:
        """Check if text has a strong opening hook"""
        text_lower = text.lower().strip()
        
        # Must start with a strong pattern
        for pattern in strong_patterns:
            if re.match(pattern, text_lower):
                return True
        
        # Additional quality checks for strong openings
        strong_indicators = [
            # Clear action/event starters
            r"^(?:I was|We were|He was|She was|They were)",
            # Direct statements
            r"^(?:this is|that was|here's)",
            # Time markers
            r"^(?:Last week|Yesterday|Earlier|This morning|Tonight)",
            # Emotional starters
            r"^(?:I'm telling you|Listen|Look|Check this out|Get this)"
        ]
        
        for pattern in strong_indicators:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _extract_story_arc(self, transcript: List[Dict], start_idx: int, max_duration: int = 30) -> Optional[Dict]:
        """Extract a complete story arc within max_duration"""
        if start_idx >= len(transcript):
            return None
            
        story_start = transcript[start_idx]['start']
        story_end_idx = start_idx
        
        logger.debug(f"Extracting story arc from index {start_idx}, start time {story_start}, max_duration {max_duration}")
        
        # Look ahead for story completion or time cutoff
        for j in range(start_idx + 1, min(len(transcript), start_idx + self.MAX_LOOKAHEAD)):
            duration = transcript[j]['end'] - story_start
            logger.debug(f"  Checking segment {j}: duration {duration:.1f}s, text: '{transcript[j]['text'][:50]}...'")
            
            if duration > max_duration:
                # Time cutoff - use previous segment
                story_end_idx = j - 1
                logger.debug(f"  Time cutoff reached, using segment {story_end_idx}")
                break
            
            # Check for ending patterns
            if self._has_ending_pattern(transcript[j]['text']):
                story_end_idx = j
                logger.debug(f"  Found ending pattern in segment {j}")
                break
            
            # Check if we've reached a natural break
            if self._is_natural_break(transcript[j]['text']):
                story_end_idx = j
                print(f"DEBUG:   Found natural break in segment {j}")
                break
        
        # If we didn't find an ending pattern or natural break, 
        # capture a reasonable chunk of the story (at least 3 segments if available)
        if story_end_idx == start_idx and len(transcript) > start_idx + 2:
            # Try to capture at least 3 segments for a good story chunk
            story_end_idx = min(start_idx + 2, len(transcript) - 1)
            logger.debug(f"  No ending found, capturing {story_end_idx - start_idx + 1} segments")
        
        logger.debug(f"Story end index: {story_end_idx}")
        
        # Ensure we have a valid story BEFORE creating the object
        if story_end_idx <= start_idx:
            logger.debug(f"Invalid story: end_idx ({story_end_idx}) <= start_idx ({start_idx})")
            return None
        
        # Additional validation: ensure we have at least 2 segments for a meaningful story
        if story_end_idx - start_idx < 1:
            logger.debug(f"Story too short: only {story_end_idx - start_idx + 1} segments")
            return None
            
        # Create story segment
        story = {
            'start': story_start,
            'end': transcript[story_end_idx]['end'],
            'text': self._extract_text(transcript, start_idx, story_end_idx),
            'type': 'story',
            'confidence': 0.8
        }
        
        duration = story['end'] - story['start']
        logger.debug(f"Story duration: {duration:.1f}s (min: {self.moment_patterns['story']['min_duration']}, max: {self.moment_patterns['story']['max_duration']})")
        
        # For viral clips, be more flexible about duration
        # If we have a strong opening and reasonable content, accept it
        if duration >= self.moment_patterns['story']['min_duration']:
            if duration <= self.moment_patterns['story']['max_duration']:
                logger.debug(f"Story duration valid, returning story")
                return story
            else:
                # Duration exceeded, but if it's not too long, still consider it
                if duration <= max_duration + 5:  # Allow 5 second buffer
                    logger.debug(f"Story duration exceeded but within buffer, returning story")
                    return story
        
        logger.debug(f"Story duration invalid, returning None")
        return None
    
    def _is_natural_break(self, text: str) -> bool:
        """Check if text represents a natural story break"""
        text_lower = text.lower().strip()
        
        # Natural break indicators
        break_patterns = [
            r"^(?:So|And|But|Then|Finally|Anyway|Moving on)",
            r"^(?:That's|This is|Here's|There's) (?:the thing|what happened|the story)",
            r"^(?:Long story short|Bottom line|Point is|Moral of the story)",
            r"^(?:Anyway|So yeah|That's it|That's all|End of story)"
        ]
        
        for pattern in break_patterns:
            if re.match(pattern, text_lower):
                return True
        
        return False
    
    def _is_self_contained(self, text: str) -> bool:
        """Check if text is self-contained and doesn't require prior context"""
        text_lower = text.lower().strip()
        
        # Only reject obvious context dependencies
        severe_markers = [
            r'as i (?:said|mentioned) (?:before|earlier)',
            r'going back to what',
            r'like we discussed',
            r'as i told you (?:before|earlier)',
            r'returning to (?:what|the topic)',
            r'remember when i (?:said|mentioned)'
        ]
        
        # Check for severe context dependency
        for pattern in severe_markers:
            if re.search(pattern, text_lower):
                return False
        
        # Be more permissive - only require basic readability
        # Must have some content and not be completely incomprehensible
        if len(text_lower.strip()) < 10:  # Too short to be meaningful
            return False
        
        return True
    
    def _find_insights(self, transcript: List[Dict]) -> List[Dict]:
        """Find insight moments - tightened duration"""
        insights = []
        
        for i, segment in enumerate(transcript):
            text = segment.get('text', '').lower()
            for pattern in self.moment_patterns['insight']['markers']:
                if re.search(pattern, text):
                    # Include previous and next segments for context, but keep tight
                    start_idx = max(0, i - 1)  # Reduced from -2
                    end_idx = min(len(transcript) - 1, i + 1)  # Reduced from +2
                    
                    moment = {
                        'start': transcript[start_idx]['start'],
                        'end': transcript[end_idx]['end'],
                        'text': self._extract_text(transcript, start_idx, end_idx),
                        'type': 'insight',
                        'confidence': 0.7
                    }
                    
                    duration = moment['end'] - moment['start']
                    if (self.moment_patterns['insight']['min_duration'] <= 
                        duration <= self.moment_patterns['insight']['max_duration']):
                        # Only add if self-contained
                        if self._is_self_contained(moment['text']):
                            insights.append(moment)
                    break
        
        return insights
    
    def _find_hot_takes(self, transcript: List[Dict]) -> List[Dict]:
        """Find hot take moments - tightened duration"""
        hot_takes = []
        
        for i, segment in enumerate(transcript):
            text = segment.get('text', '').lower()
            for pattern in self.moment_patterns['hot_take']['markers']:
                if re.search(pattern, text):
                    # Include more context for hot takes to meet duration requirements
                    start_idx = max(0, i - 1)  # Include previous segment
                    end_idx = min(len(transcript) - 1, i + 1)  # Include next segment
                    
                    moment = {
                        'start': transcript[start_idx]['start'],
                        'end': transcript[end_idx]['end'],
                        'text': self._extract_text(transcript, start_idx, end_idx),
                        'type': 'hot_take',
                        'confidence': 0.6
                    }
                    
                    duration = moment['end'] - moment['start']
                    if (self.moment_patterns['hot_take']['min_duration'] <= 
                        duration <= self.moment_patterns['hot_take']['max_duration']):
                        # Only add if self-contained
                        if self._is_self_contained(moment['text']):
                            hot_takes.append(moment)
                    break
        
        return hot_takes
    
    def _has_ending_pattern(self, text: str) -> bool:
        """Check if text contains a story ending pattern"""
        text_lower = text.lower()
        for pattern in self.moment_patterns['story']['ends']:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _extract_text(self, transcript: List[Dict], start_idx: int, end_idx: int) -> str:
        """Extract text from transcript segments"""
        texts = []
        for i in range(start_idx, end_idx + 1):
            if i < len(transcript):
                texts.append(transcript[i].get('text', ''))
        return ' '.join(texts)
    
    def _deduplicate_moments(self, moments: List[Dict]) -> List[Dict]:
        """Remove overlapping moments, keeping highest quality"""
        if not moments:
            return moments
        
        # Sort by priority: stories > insights > hot_takes, then by confidence
        type_priority = {'story': 3, 'insight': 2, 'hot_take': 1}
        
        moments.sort(key=lambda m: (
            type_priority.get(m.get('type', 'general'), 0),
            m.get('confidence', 0)
        ), reverse=True)
        
        kept = []
        for moment in moments:
            overlap = False
            for existing in kept:
                if (moment['start'] < existing['end'] and 
                    moment['end'] > existing['start']):
                    overlap = True
                    break
            
            if not overlap:
                kept.append(moment)
        
        return kept
    
    def _requires_context(self, text: str) -> bool:
        """Check if text requires prior context to understand"""
        # More lenient for stories, stricter for insights
        context_markers = [
            r'^(this|that|these|those|it|they)',  # Unclear references
            r'^(he|she|them)(?! who)',  # Pronouns without antecedents
            r'as I (said|mentioned) (before|earlier)',  # Back references
            r'like I said',  # Back references
            r'going back to',  # Back references
        ]
        
        text_lower = text.lower().strip()
        return any(re.match(pattern, text_lower) for pattern in context_markers)
    
    def _has_quality(self, moment: Dict, quality: str) -> bool:
        """Check if moment has specific quality"""
        quality_checks = {
            'complete_arc': lambda m: any(word in m['text'].lower() for word in ['start', 'beginning', 'end', 'finally']),
            'emotional_peak': lambda m: any(word in m['text'].lower() for word in ['amazing', 'incredible', 'shocked', 'unbelievable', 'crazy']),
            'unexpected_twist': lambda m: any(phrase in m['text'].lower() for phrase in ['but then', 'suddenly', 'out of nowhere', 'surprisingly']),
            'counterintuitive': lambda m: any(phrase in m['text'].lower() for phrase in ['but actually', 'people think', 'the truth is', 'contrary to']),
            'actionable': lambda m: any(word in m['text'].lower() for word in ['should', 'need to', 'have to', 'must', 'always', 'never']),
            'memorable': lambda m: any(phrase in m['text'].lower() for phrase in ['remember this', 'key point', 'takeaway', 'bottom line']),
            'controversial': lambda m: any(phrase in m['text'].lower() for phrase in ['unpopular', 'controversial', 'might get hate', 'downvotes']),
            'well_argued': lambda m: any(phrase in m['text'].lower() for phrase in ['because', 'since', 'the reason', 'evidence', 'proof']),
            'quotable': lambda m: any(phrase in m['text'].lower() for phrase in ['quote', 'saying', 'phrase', 'line'])
        }
        
        check = quality_checks.get(quality)
        return check(moment) if check else False
