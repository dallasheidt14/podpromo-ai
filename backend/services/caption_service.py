"""
Caption Service - Generates adaptive captions with word highlighting and proper line breaks.
"""

import re
from typing import List, Dict, Tuple
from datetime import timedelta

class CaptionService:
    """Service for generating adaptive captions with word highlighting"""
    
    def __init__(self):
        self.max_words_per_line = 10
        self.min_words_per_line = 6
    
    def generate_ass_captions(self, transcript: List[Dict], style: str = "bold") -> str:
        """
        Generate .ass subtitle file with adaptive line breaks and word highlighting.
        
        Args:
            transcript: List of word objects with text, start, end
            style: Caption style (bold, clean, caption-heavy)
        
        Returns:
            ASS subtitle content as string
        """
        if not transcript:
            return ""
        
        # Get style configuration
        style_config = self._get_style_config(style)
        
        # Generate ASS header
        ass_content = self._generate_ass_header(style_config)
        
        # Process transcript into caption lines
        caption_lines = self._process_transcript(transcript)
        
        # Generate ASS dialogue entries
        for i, line in enumerate(caption_lines):
            start_time = self._format_time(line["start"])
            end_time = self._format_time(line["end"])
            
            # Create highlighted text with bold tags around spoken words
            highlighted_text = self._highlight_spoken_words(line["text"], line["spoken_words"])
            
            # Format as ASS dialogue
            dialogue = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{highlighted_text}\n"
            ass_content += dialogue
        
        return ass_content
    
    def _get_style_config(self, style: str) -> Dict:
        """Get caption style configuration"""
        styles = {
            "bold": {
                "fontsize": 54,
                "outline": 2,
                "shadow": 2,
                "primary_color": "&HFFFFFF&",
                "outline_color": "&H000000&",
                "shadow_color": "&H000000&"
            },
            "clean": {
                "fontsize": 42,
                "outline": 0,
                "shadow": 0,
                "primary_color": "&HFFFFFF&",
                "outline_color": "&H000000&",
                "shadow_color": "&H000000&"
            },
            "caption-heavy": {
                "fontsize": 60,
                "outline": 1,
                "shadow": 1,
                "primary_color": "&HFFFFFF&",
                "outline_color": "&H000000&",
                "shadow_color": "&H000000&",
                "line_spacing": -5
            }
        }
        return styles.get(style, styles["bold"])
    
    def _generate_ass_header(self, style_config: Dict) -> str:
        """Generate ASS file header with style configuration"""
        header = """[Script Info]
Title: PodPromo AI Generated Captions
ScriptType: v4.00+
WrapStyle: 1
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{fontsize},{primary_color},&H000000&,{outline_color},{shadow_color},1,0,0,0,100,100,0,0,1,{outline},2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
            fontsize=style_config["fontsize"],
            primary_color=style_config["primary_color"],
            outline_color=style_config["outline_color"],
            shadow_color=style_config["shadow_color"],
            outline=style_config["outline"]
        )
        return header
    
    def _process_transcript(self, transcript: List[Dict]) -> List[Dict]:
        """
        Process transcript into optimal caption lines with proper word counts.
        Each line should have 6-10 words for optimal readability.
        """
        lines = []
        current_line = {
            "text": "",
            "words": [],
            "start": transcript[0]["start"] if transcript else 0,
            "end": 0,
            "spoken_words": []
        }
        
        for word in transcript:
            word_text = word.get("text", "").strip()
            if not word_text:
                continue
            
            # Check if adding this word would exceed max words per line
            if len(current_line["words"]) >= self.max_words_per_line:
                # Finalize current line
                current_line["end"] = word["start"]
                current_line["text"] = " ".join(current_line["words"])
                lines.append(current_line)
                
                # Start new line
                current_line = {
                    "text": "",
                    "words": [word_text],
                    "start": word["start"],
                    "end": 0,
                    "spoken_words": [word]
                }
            else:
                # Add word to current line
                current_line["words"].append(word_text)
                current_line["spoken_words"].append(word)
        
        # Add final line
        if current_line["words"]:
            current_line["end"] = transcript[-1]["end"] if transcript else 0
            current_line["text"] = " ".join(current_line["words"])
            lines.append(current_line)
        
        return lines
    
    def _highlight_spoken_words(self, full_text: str, spoken_words: List[Dict]) -> str:
        """
        Add ASS bold tags around words that are currently being spoken.
        This creates a karaoke-style effect where words light up as they're spoken.
        """
        if not spoken_words:
            return full_text
        
        # Create a mapping of word positions in the full text
        highlighted_text = full_text
        
        # For now, we'll bold the entire line since ASS doesn't support per-word timing easily
        # In a more advanced version, we could split into multiple dialogue entries
        return f"{{\\b1}}{full_text}{{\\b0}}"
    
    def _format_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.cc)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = td.total_seconds() % 60
        centiseconds = int((secs % 1) * 100)
        
        return f"{hours}:{minutes:02d}:{secs:05.2f}".replace(".", ",")
    
    def get_ffmpeg_subtitle_filter(self, ass_file_path: str, style: str = "bold") -> str:
        """
        Generate ffmpeg subtitle filter command for the specified style.
        
        Args:
            ass_file_path: Path to the generated .ass file
            style: Caption style (bold, clean, caption-heavy)
        
        Returns:
            ffmpeg subtitle filter string
        """
        style_config = self._get_style_config(style)
        
        # Create force_style string for ffmpeg
        force_style = (
            f"Fontsize={style_config['fontsize']},"
            f"PrimaryColour={style_config['primary_color']},"
            f"Outline={style_config['outline']},"
            f"Shadow={style_config['shadow']},"
            f"Alignment=2"  # Center alignment
        )
        
        return f"subtitles='{ass_file_path}':force_style='{force_style}'"
