"""
ViralMomentDetector - Intelligent content-based segmentation for viral clips
"""

import re
import logging
from typing import List, Dict, Optional
from functools import lru_cache
from config_loader import load_config

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
    
    # --- Insight V2 helpers (lightweight, no heavy deps) ---
    CLAUSE_END = re.compile(r"[.!?](?:\s+|$)")
    CONTRAST = re.compile(r"(most (people|folks)|everyone|nobody).{0,40}\b(actually|but|instead)\b", re.I)
    CAUSAL = re.compile(r"\b(because|therefore|so|which means)\b", re.I)
    HAS_NUM = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b)?\b", re.I)
    COMPAR = re.compile(r"\b(vs\.?|versus|more than|less than|bigger than|smaller than)\b", re.I)
    IMPER = re.compile(r"\b(try|avoid|do|don['']t|stop|start|focus|use|measure|swap|choose|should|need|must)\b", re.I)
    HEDGE = re.compile(r"\b(maybe|probably|i think|i guess|kinda|sort of)\b", re.I)
    COORD = re.compile(r"^(and|but|so|or|yet|nor)\b", re.I)
    EVIDENCE_TOKEN = re.compile(r"\b\d+(?:\.\d+)?(?:%|k|m|b|\$)?\b|\bvs\.?|versus\b|more than|less than", re.I)
    
    def _span_duration(self, transcript, i0, i1) -> float:
        """Calculate duration between two transcript indices"""
        return float(transcript[i1]['end'] - transcript[i0]['start'])
    
    def _token_count(self, s: str) -> int:
        """Count tokens in a string"""
        return len(re.findall(r"[A-Za-z']+", s or ""))
    
    def _looks_evidential(self, s: str) -> bool:
        """Check if text contains evidence tokens or causal markers"""
        return bool(self.EVIDENCE_TOKEN.search(s or "")) or bool(self.CAUSAL.search(s or ""))
    
    def _expand_insight_window(self, transcript, i, max_s=25.0):
        """
        Start at segment i; expand forward with 'soft boundary' rules:
          - If current seg ends with a clause mark + NEXT starts capitalized, we usually stop…
          - …UNLESS the current clause is a short opener (≤8 tokens), OR the next clause
            starts with a coordinator (but/and/so) OR contains clear evidence.
        Never exceed max_s from the seed start.
        Optionally borrow the previous very-short setup (≤3s) if it's not low-info.
        """
        start = i
        end = i
        t0 = float(transcript[i]['start'])

        # forward growth
        while end + 1 < len(transcript):
            nxt_end = float(transcript[end + 1]['end'])
            if (nxt_end - t0) > max_s:
                break

            cur_txt = (transcript[end].get('text') or "").replace("'", "'")
            nxt_txt = (transcript[end + 1].get('text') or "").replace("'", "'").lstrip()

            # Hard boundary heuristic: current ends with clause mark AND next starts with capital
            ends_clause = bool(self.CLAUSE_END.search(cur_txt))
            next_cap    = bool(nxt_txt[:1].isupper())

            # Soft-openers: very short first/preceding clause (<= 8 tokens)
            short_opener = (self._token_count(cur_txt) <= 8)

            # Continuation signals: coordinating conjunction at start OR next clause has evidence
            continuation = bool(self.COORD.match(nxt_txt)) or self._looks_evidential(nxt_txt)

            # Decision:
            # - If hard boundary detected AND NOT (short opener OR continuation), stop.
            if ends_clause and next_cap and not (short_opener or continuation):
                break

            end += 1

        # Borrow a short setup segment before i if close and not low-info
        if start > 0:
            prev = transcript[start - 1]
            if (float(transcript[i]['start']) - float(prev['start'])) <= 3.0:
                txt_prev = prev.get('text') or ""
                if not self._is_low_info(txt_prev):
                    start -= 1

        return start, end
    
    def _is_low_info(self, text: str) -> bool:
        """Check if text is mostly filler/low-information content"""
        toks = re.findall(r"[A-Za-z']+", text.lower())
        if not toks:
            return True
        fillers = set("like you know kinda sort of basically literally actually really just i think maybe probably".split())
        f = sum(1 for t in toks if t in fillers)
        return (f / max(1, len(toks))) > 0.5
    
    @lru_cache(maxsize=128)
    def _sentences(self, text: str) -> List[str]:
        """Split text into sentences (cached)"""
        return re.split(r'(?<=[.!?])\s+', text.strip())
    
    @lru_cache(maxsize=128)
    def _tokens(self, text: str) -> List[str]:
        """Extract tokens from text (cached)"""
        return re.findall(r"[A-Za-z]+", text)
    
    def _noun_burst(self, text: str) -> bool:
        """Detect concrete noun burst: 2+ proper-looking capitalized tokens not in sentence-initial position"""
        sents = self._sentences(text)
        for s in sents:
            toks = self._tokens(s)
            caps = 0
            for idx, t in enumerate(toks):
                if idx == 0:  # ignore sentence-initial capital
                    continue
                if t[0].isupper():
                    caps += 1
            if caps >= 2:
                return True
        return False
    
    def _evidence_and_confidence(self, text: str) -> tuple[bool, float]:
        """
        Returns (has_evidence, confidence 0.5..0.9).
        Evidence = any of {contrast, number, comparison}. Imperatives add confidence;
        hedges reduce confidence slightly. Start at 0.6.
        """
        t = (text or "").strip()
        conf = 0.6
        hard_hits = 0
        
        if self.CONTRAST.search(t): 
            conf += 0.10
            hard_hits += 1
        if self.HAS_NUM.search(t):  
            conf += 0.10
            hard_hits += 1
        if self.COMPAR.search(t):   
            conf += 0.05
            hard_hits += 1
        if self.CAUSAL.search(t):   
            conf += 0.05
        if self.IMPER.search(t):    
            conf += 0.10
        if self.HEDGE.search(t):    
            conf -= 0.10
        if self._noun_burst(t):     
            conf += 0.05
        
        conf = max(0.5, min(0.9, conf))
        has_ev = (hard_hits >= 1)
        return has_ev, conf
    
    def _avg_info_density(self, transcript, i0, i1) -> float | None:
        """
        Optional: if segments have 'info_density', return the average as a tiny nudge.
        Safe to return None if unavailable.
        """
        vals = []
        for j in range(i0, i1+1):
            v = transcript[j].get('info_density')
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return (sum(vals)/len(vals)) if vals else None
    
    def find_moments(self, transcript: List[Dict]) -> List[Dict]:
        """Find potential viral moments based on content patterns"""
        try:
            moments = []
            
            # Find stories - now much more selective
            stories = self._find_complete_stories(transcript)
            moments.extend(stories)
            
            # Find insights - use V2 if enabled
            config = load_config()
            if config.get("insight_v2", {}).get("enabled", False):
                insights = self._find_insights_v2(transcript)
            else:
                insights = self._find_insights_v1(transcript)
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
    
    def _find_insights_v1(self, transcript: List[Dict]) -> List[Dict]:
        """Find insight moments - V1 implementation (tightened duration)"""
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
    
    def _find_insights_v2(self, transcript: List[Dict]) -> List[Dict]:
        """
        Insight V2:
        - Trigger on your existing markers.
        - Expand window by clauses (≤ max_duration).
        - Require evidence (numbers/comparison/contrast) OR be short & punchy (≤12s).
        - Confidence reflects structure; small nudge for high info_density.
        - Keep min/max duration + self-contained checks.
        """
        cfg = self.moment_patterns.get('insight', {})
        min_d = float(cfg.get('min_duration', 10.0))
        max_d = float(cfg.get('max_duration', 25.0))
        insights = []

        for i, seg in enumerate(transcript):
            raw = (seg.get('text') or "")
            text = raw.replace("'", "'")  # normalize smart quotes
            low = text.lower()
            if not any(re.search(p, low, flags=re.I) for p in cfg.get('markers', [])):
                continue

            i0, i1 = self._expand_insight_window(transcript, i, max_s=max_d)
            start = float(transcript[i0]['start']); end = float(transcript[i1]['end'])
            dur = end - start
            window = self._extract_text(transcript, i0, i1)

            if not (min_d <= dur <= max_d):
                continue

            if not self._is_self_contained(window):
                continue

            has_ev, conf = self._evidence_and_confidence(window)

            if not has_ev:
                if dur > 12.0:
                    continue
                if not (self.IMPER.search(window) or self.CAUSAL.search(window)):
                    continue
                conf = max(0.55, conf - 0.05)

            # optional info_density nudge
            idv = self._avg_info_density(transcript, i0, i1)
            if isinstance(idv, float) and idv >= 0.60:
                conf = min(0.90, conf + 0.05)

            insights.append({"start":start, "end":end, "text":window, "type":"insight", "confidence":conf})

        deduped = self._deduplicate_moments(insights)
        return deduped
    
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
