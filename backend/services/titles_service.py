import re, random, time, hashlib, json, os
from datetime import datetime
from typing import List, Tuple, Dict, Any
from pathlib import Path

MAX = {"shorts":80, "tiktok":80, "reels":80, "youtube":100}

HOOK_PATTERNS = [
    r"^how to\b", r"^the (secret|truth) about\b", r"^(stop|never) doing\b",
    r"^\d+\b (ways|reasons|mistakes)\b", r"^this (one|simple) thing\b", r"^why\b"
]

EMOJI = ["🚀","🔥","🤯","✅","⚠️","💡","🎯"]
BANNED = ["subscribe","like and subscribe","click the link"]

def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    for b in BANNED: s = re.sub(b, "", s, flags=re.I)
    return s.strip(" -–—·|")

def titlecase(s: str) -> str:
    return re.sub(r"\b(\w+)", lambda m: m.group(1).capitalize(), s.lower())

def clip_snippet(text: str, max_chars: int) -> str:
    s = clean(text)
    if len(s) <= max_chars: return s
    # prefer sentence boundary
    s = s[:max_chars-1]
    cut = max(s.rfind("."), s.rfind("!"), s.rfind("?"))
    return s[:cut].strip() if cut > 35 else s[:max_chars-1].rstrip() + "…"

def cheap_arousal(text: str) -> float:
    bonus = 0
    bonus += text.count("!") * 0.1
    bonus += len(re.findall(r"\b(huge|wild|insane|shocking|secret|mistake|risk)\b", text, re.I)) * 0.08
    return min(1.0, bonus)

def fits_limit(s: str, limit: int) -> str:
    return s if len(s) <= limit else (s[:limit-1].rstrip() + "…")

def make_variants(text: str, platform: str, allow_emoji: bool, seed: int | None) -> List[str]:
    rnd = random.Random(seed or hashlib.sha1(text.encode()).digest())
    limit = MAX[platform]
    base = clean(text)

    # pattern-driven hooks
    starters = [
        "The Secret To", "Stop Doing", "3 Mistakes In", "Why", "How To", "Nobody Tells You About",
        "This One Thing About", "Before You", "What No One Explains About"
    ]
    nouns = ["Startups","Hiring","AI","Sales","Founders","Marketing","Product","Habits","Money"]

    # build candidates
    cands = [
        f"{rnd.choice(starters)} {clip_snippet(base, 40)}",
        f"{clip_snippet(base, 60)}",
        f"{re.sub(r'^\W+','', base.split('.')[0])}",
        f"{rnd.choice(['Big mistake:','Truth:','Hot take:','Pro tip:'])} {clip_snippet(base, 55)}",
        f"{rnd.choice(['Why','How'])} {clip_snippet(base, 62).lower()}",
        f"{rnd.choice(nouns)}: {clip_snippet(base, 60)}",
    ]

    # light scoring to pick "best"
    scored = []
    for t in cands:
        score = len(re.findall("|".join(HOOK_PATTERNS), t, re.I)) > 0
        score = (1.0 if score else 0) + cheap_arousal(t) + (0.15 if len(t) < limit*0.85 else 0)
        txt = titlecase(t)
        if allow_emoji and rnd.random() < 0.55:
            txt = (rnd.choice(EMOJI) + " " + txt) if rnd.random() < 0.5 else (txt + " " + rnd.choice(EMOJI))
        scored.append((score, fits_limit(txt, limit)))

    scored.sort(key=lambda x: x[0], reverse=True)
    unique = []
    seen = set()
    for _, t in scored:
        k = t.lower()
        if k not in seen:
            unique.append(t); seen.add(k)
        if len(unique) == 6: break
    return unique

class TitlesService:
    GEN_VERSION = 1

    def __init__(self):
        from config.settings import UPLOAD_DIR
        self.upload_dir = Path(UPLOAD_DIR)

    def get_clip(self, clip_id: str) -> Dict[str, Any] | None:
        """Load clip data from episode clips.json file"""
        # Search through all episode directories for the clip
        for episode_dir in self.upload_dir.iterdir():
            if not episode_dir.is_dir():
                continue
                
            clips_file = episode_dir / "clips.json"
            if not clips_file.exists():
                continue
                
            try:
                with open(clips_file, 'r', encoding='utf-8') as f:
                    clips_data = json.load(f)
                    
                # Look for the clip in this episode's clips
                if isinstance(clips_data, list):
                    for clip in clips_data:
                        if clip.get("id") == clip_id:
                            return clip
                elif isinstance(clips_data, dict) and clips_data.get("id") == clip_id:
                    return clips_data
                    
            except Exception as e:
                print(f"Error reading clips from {clips_file}: {e}")
                continue
                
        return None

    def save_titles(self, clip_id: str, platform: str, variants: List[str], chosen: str, meta: dict):
        """Save generated titles back to episode clips.json file"""
        # Find the episode directory containing this clip
        episode_dir = None
        clips_file = None
        
        for ep_dir in self.upload_dir.iterdir():
            if not ep_dir.is_dir():
                continue
                
            clips_f = ep_dir / "clips.json"
            if not clips_f.exists():
                continue
                
            try:
                with open(clips_f, 'r', encoding='utf-8') as f:
                    clips_data = json.load(f)
                    
                # Check if this episode contains our clip
                if isinstance(clips_data, list):
                    for clip in clips_data:
                        if clip.get("id") == clip_id:
                            episode_dir = ep_dir
                            clips_file = clips_f
                            break
                elif isinstance(clips_data, dict) and clips_data.get("id") == clip_id:
                    episode_dir = ep_dir
                    clips_file = clips_f
                    break
                    
            except Exception as e:
                print(f"Error reading clips from {clips_f}: {e}")
                continue
                
        if not episode_dir or not clips_file:
            print(f"Could not find episode directory for clip {clip_id}")
            return False
        
        # Load the full clips data
        try:
            with open(clips_file, 'r', encoding='utf-8') as f:
                clips_data = json.load(f)
        except Exception as e:
            print(f"Error loading clips data: {e}")
            return False
        
        # Find and update the specific clip
        if isinstance(clips_data, list):
            for i, clip in enumerate(clips_data):
                if clip.get("id") == clip_id:
                    # Initialize title structure if not exists
                    if "title_variants" not in clip:
                        clip["title_variants"] = {}
                    if "title_meta" not in clip:
                        clip["title_meta"] = {}
                    
                    # Update with new data
                    clip["title_variants"][platform] = variants
                    clip["title"] = chosen
                    clip["title_meta"][platform] = meta
                    
                    clips_data[i] = clip
                    break
        elif isinstance(clips_data, dict) and clips_data.get("id") == clip_id:
            # Single clip file
            clip = clips_data
            if "title_variants" not in clip:
                clip["title_variants"] = {}
            if "title_meta" not in clip:
                clip["title_meta"] = {}
            
            clip["title_variants"][platform] = variants
            clip["title"] = chosen
            clip["title_meta"][platform] = meta
        
        # Save back to file
        try:
            with open(clips_file, 'w', encoding='utf-8') as f:
                json.dump(clips_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving titles for clip {clip_id}: {e}")
            return False

    def set_chosen_title(self, clip_id: str, platform: str, title: str) -> bool:
        """Set the chosen title for a clip"""
        # Find the episode directory containing this clip
        episode_dir = None
        clips_file = None
        
        for ep_dir in self.upload_dir.iterdir():
            if not ep_dir.is_dir():
                continue
                
            clips_f = ep_dir / "clips.json"
            if not clips_f.exists():
                continue
                
            try:
                with open(clips_f, 'r', encoding='utf-8') as f:
                    clips_data = json.load(f)
                    
                # Check if this episode contains our clip
                if isinstance(clips_data, list):
                    for clip in clips_data:
                        if clip.get("id") == clip_id:
                            episode_dir = ep_dir
                            clips_file = clips_f
                            break
                elif isinstance(clips_data, dict) and clips_data.get("id") == clip_id:
                    episode_dir = ep_dir
                    clips_file = clips_f
                    break
                    
            except Exception as e:
                print(f"Error reading clips from {clips_f}: {e}")
                continue
                
        if not episode_dir or not clips_file:
            print(f"Could not find episode directory for clip {clip_id}")
            return False
        
        # Load the full clips data
        try:
            with open(clips_file, 'r', encoding='utf-8') as f:
                clips_data = json.load(f)
        except Exception as e:
            print(f"Error loading clips data: {e}")
            return False
        
        # Find and update the specific clip
        if isinstance(clips_data, list):
            for i, clip in enumerate(clips_data):
                if clip.get("id") == clip_id:
                    clip["title"] = title
                    
                    # Update the platform's chosen title in variants
                    if "title_variants" not in clip:
                        clip["title_variants"] = {}
                    if platform not in clip["title_variants"]:
                        clip["title_variants"][platform] = []
                    
                    # Move chosen title to front of variants list
                    variants = clip["title_variants"][platform]
                    if title in variants:
                        variants.remove(title)
                    variants.insert(0, title)
                    
                    clips_data[i] = clip
                    break
        elif isinstance(clips_data, dict) and clips_data.get("id") == clip_id:
            # Single clip file
            clip = clips_data
            clip["title"] = title
            
            # Update the platform's chosen title in variants
            if "title_variants" not in clip:
                clip["title_variants"] = {}
            if platform not in clip["title_variants"]:
                clip["title_variants"][platform] = []
            
            # Move chosen title to front of variants list
            variants = clip["title_variants"][platform]
            if title in variants:
                variants.remove(title)
            variants.insert(0, title)
        
        # Save back to file
        try:
            with open(clips_file, 'w', encoding='utf-8') as f:
                json.dump(clips_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error setting title for clip {clip_id}: {e}")
            return False

    def generate_variants(self, clip: Dict[str, Any], req) -> Tuple[List[str], str, dict]:
        """Generate title variants for a clip"""
        text = clip.get("transcript") or clip.get("text") or ""
        if not text.strip():
            # Fallback if no transcript
            text = f"Clip from {clip.get('startTime', 0)}s to {clip.get('endTime', 0)}s"
        
        variants = make_variants(text, req.platform, req.allow_emoji, req.seed)
        chosen = variants[0] if variants else "Untitled Clip"
        
        meta = {
            "generator": "heuristic",
            "version": self.GEN_VERSION,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        return variants, chosen, meta

    def ensure_titles_exist(self, clip_id: str, platform: str = "shorts") -> bool:
        """Ensure a clip has titles generated (lazy generation)"""
        clip = self.get_clip(clip_id)
        if not clip:
            return False
        
        # Check if titles already exist for this platform
        if (clip.get("title_variants", {}).get(platform) and 
            clip.get("title")):
            return True
        
        # Generate titles if missing
        from models_titles import TitleGenRequest
        req = TitleGenRequest(platform=platform)
        variants, chosen, meta = self.generate_variants(clip, req)
        return self.save_titles(clip_id, platform, variants, chosen, meta)
