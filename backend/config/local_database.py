"""
Local database configuration for development when Supabase is not available
Uses SQLite as a fallback for development
"""

import os
import sqlite3
import json
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Local SQLite database path
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "local_dev.db")

def get_local_db():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    return conn

def init_local_tables():
    """Initialize local database tables"""
    conn = get_local_db()
    cursor = conn.cursor()
    
    # Episodes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            filename TEXT,
            title TEXT,
            duration REAL,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Clips table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            episode_id TEXT,
            start_time REAL,
            end_time REAL,
            title TEXT,
            score REAL,
            platform TEXT,
            preview_url TEXT,
            transcript TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (episode_id) REFERENCES episodes (id)
        )
    """)
    
    # Users table (for future use)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            subscription_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Local database tables initialized")

def save_episode(episode_data: Dict[str, Any]) -> bool:
    """Save episode data to local database"""
    try:
        conn = get_local_db()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO episodes 
            (id, filename, title, duration, raw_text)
            VALUES (?, ?, ?, ?, ?)
        """, (
            episode_data.get('id'),
            episode_data.get('filename'),
            episode_data.get('title'),
            episode_data.get('duration'),
            episode_data.get('raw_text')
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save episode: {e}")
        return False

def save_clips(clips_data: list[Dict[str, Any]]) -> bool:
    """Save clips data to local database"""
    try:
        conn = get_local_db()
        cursor = conn.cursor()
        
        for clip in clips_data:
            cursor.execute("""
                INSERT OR REPLACE INTO clips 
                (id, episode_id, start_time, end_time, title, score, platform, preview_url, transcript)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                clip.get('id'),
                clip.get('episode_id'),
                clip.get('start_time'),
                clip.get('end_time'),
                clip.get('title'),
                clip.get('score'),
                clip.get('platform'),
                clip.get('preview_url'),
                clip.get('transcript')
            ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save clips: {e}")
        return False

def get_episodes() -> list[Dict[str, Any]]:
    """Get all episodes from local database"""
    try:
        conn = get_local_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM episodes ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get episodes: {e}")
        return []

def get_clips(episode_id: Optional[str] = None) -> list[Dict[str, Any]]:
    """Get clips from local database, optionally filtered by episode_id"""
    try:
        conn = get_local_db()
        cursor = conn.cursor()
        
        if episode_id:
            cursor.execute("SELECT * FROM clips WHERE episode_id = ? ORDER BY score DESC", (episode_id,))
        else:
            cursor.execute("SELECT * FROM clips ORDER BY created_at DESC")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get clips: {e}")
        return []

def check_local_db_connection():
    """Check if local database is working"""
    try:
        conn = get_local_db()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Local database connection failed: {e}")
        return False
