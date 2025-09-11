-- Supabase database schema for podpromo application
-- Run this in the Supabase SQL editor

-- Episodes table
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    filename TEXT,
    title TEXT,
    duration REAL,
    raw_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Clips table
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    episode_id TEXT REFERENCES episodes(id) ON DELETE CASCADE,
    start_time REAL,
    end_time REAL,
    title TEXT,
    score REAL,
    platform TEXT,
    preview_url TEXT,
    transcript TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE,
    subscription_status TEXT DEFAULT 'free',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_clips_episode_id ON clips(episode_id);
CREATE INDEX IF NOT EXISTS idx_clips_score ON clips(score DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at DESC);

-- Enable Row Level Security (RLS) for better security
ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE clips ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (you can restrict these later)
CREATE POLICY "Allow public read access to episodes" ON episodes FOR SELECT USING (true);
CREATE POLICY "Allow public read access to clips" ON clips FOR SELECT USING (true);
CREATE POLICY "Allow public insert access to episodes" ON episodes FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public insert access to clips" ON clips FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access to episodes" ON episodes FOR UPDATE USING (true);
CREATE POLICY "Allow public update access to clips" ON clips FOR UPDATE USING (true);
CREATE POLICY "Allow public delete access to episodes" ON episodes FOR DELETE USING (true);
CREATE POLICY "Allow public delete access to clips" ON clips FOR DELETE USING (true);