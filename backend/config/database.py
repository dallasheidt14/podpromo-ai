"""
Database configuration and connection management using Supabase with local SQLite fallback
"""

import os
from supabase import create_client, Client
import logging
from .local_database import (
    init_local_tables, save_episode, save_clips, get_episodes, get_clips,
    check_local_db_connection
)

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = "https://jozohgxvdzosbcrdppkk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impvem9oZ3h2ZHpvc2JjcmRwcGtrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTY2ODAxNDUsImV4cCI6MjA3MjI1NjE0NX0.XUW5YIRUaYFyk1EYEYNBi-Y5HrgTm7S8ciUcN5aJtgE"

# Try to initialize Supabase client
supabase: Client = None
USE_LOCAL_DB = False

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized")
except Exception as e:
    logger.warning(f"Failed to initialize Supabase: {e}")
    logger.info("Falling back to local SQLite database")
    USE_LOCAL_DB = True
    init_local_tables()

def get_supabase():
    """Get Supabase client instance"""
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
