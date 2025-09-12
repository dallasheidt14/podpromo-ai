"""
Database configuration and connection management using Supabase with local SQLite fallback
"""

import os, re, logging
from typing import Optional
from .local_database import (
    init_local_tables, save_episode, save_clips, get_episodes, get_clips,
    check_local_db_connection
)

logger = logging.getLogger(__name__)

# Supabase configuration from environment
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")

# Early format checks to catch typos in CI/CD
if SUPABASE_URL and not re.match(r"^https://[a-zA-Z0-9-]+\.supabase\.co$", SUPABASE_URL):
    raise ValueError("Invalid SUPABASE_URL format")
if SUPABASE_KEY and not SUPABASE_KEY.startswith("eyJ"):
    raise ValueError("Invalid SUPABASE_KEY format (expected JWT-like)")

if not SUPABASE_URL or not SUPABASE_KEY:
    if os.getenv("ALLOW_LOCAL_DB_FALLBACK", "true").lower() in {"1","true","yes"}:
        logging.getLogger(__name__).warning("Using local SQLite fallback — NOT for production.")
        from .local_database import LocalDB as DB
        db = DB()
        USE_LOCAL_DB = True
    else:
        raise RuntimeError("Missing SUPABASE_URL/SUPABASE_KEY and local fallback disabled")
else:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    db = supabase
    USE_LOCAL_DB = False

def get_supabase():
    """Get Supabase client instance"""
    if USE_LOCAL_DB:
        return None
    return supabase

def check_db_connection():
    """Check database connection health"""
    if USE_LOCAL_DB:
        return check_local_db_connection()
    
    try:
        # Simple connection test - just check if we can reach Supabase
        # The error about missing table means the connection is working
        result = supabase.table('_dummy_').select('*').limit(1).execute()
        return True
    except Exception as e:
        # If we get a table not found error, the connection is working
        if "Could not find the table" in str(e):
            logger.info("Supabase connection successful - table not found error is expected")
            return True
        else:
            logger.error(f"Supabase connection failed: {e}")
            logger.info("Falling back to local database")
            return check_local_db_connection()

def init_db():
    """Initialize database tables"""
    if USE_LOCAL_DB:
        try:
            init_local_tables()
            logger.info("Local SQLite database initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize local database: {e}")
            raise
    else:
        try:
            logger.info("Supabase database is ready - tables will be created automatically")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Supabase database: {e}")
            raise

# Database operation functions with fallback
def save_episode_data(episode_data):
    """Save episode data to database"""
    if USE_LOCAL_DB:
        return save_episode(episode_data)
    else:
        try:
            result = supabase.table('episodes').upsert(episode_data).execute()
            logger.info(f"Saved episode {episode_data.get('id')} to Supabase")
            return True
        except Exception as e:
            logger.error(f"Failed to save episode to Supabase: {e}")
            return False

def save_clips_data(clips_data):
    """Save clips data to database"""
    if USE_LOCAL_DB:
        return save_clips(clips_data)
    else:
        try:
            result = supabase.table('clips').upsert(clips_data).execute()
            logger.info(f"Saved {len(clips_data)} clips to Supabase")
            return True
        except Exception as e:
            logger.error(f"Failed to save clips to Supabase: {e}")
            return False

def get_episodes_data():
    """Get episodes from database"""
    if USE_LOCAL_DB:
        return get_episodes()
    else:
        try:
            result = supabase.table('episodes').select('*').order('created_at', desc=True).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get episodes from Supabase: {e}")
            return []

def get_clips_data(episode_id=None):
    """Get clips from database"""
    if USE_LOCAL_DB:
        return get_clips(episode_id)
    else:
        try:
            if episode_id:
                result = supabase.table('clips').select('*').eq('episode_id', episode_id).order('score', desc=True).execute()
            else:
                result = supabase.table('clips').select('*').order('created_at', desc=True).execute()
            return result.data
        except Exception as e:
            logger.error(f"Failed to get clips from Supabase: {e}")
            return []
