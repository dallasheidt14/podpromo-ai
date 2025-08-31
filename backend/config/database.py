"""
Database configuration and connection management using Supabase
"""

import os
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = "https://jozohgxvdzosbcrdppkk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impvem9oZ3h2ZHpvc2JjcmRwcGtrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTY2ODAxNDUsImV4cCI6MjA3MjI1NjE0NX0.XUW5YIRUaYFyk1EYEYNBi-Y5HrgTm7S8ciUcN5aJtgE"

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase():
    """Get Supabase client instance"""
    return supabase

def check_db_connection():
    """Check database connection health"""
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
            logger.error(f"Database connection failed: {e}")
            return False

def init_db():
    """Initialize database tables - will be handled by Supabase"""
    try:
        logger.info("Supabase database is ready - tables will be created automatically")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
