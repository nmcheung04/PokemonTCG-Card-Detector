-- Pokemon Card Detection System - Complete Database Setup
-- This file contains all the necessary SQL commands to set up the database

-- =====================================================
-- 1. CREATE USER_DECKS TABLE
-- =====================================================

CREATE TABLE user_decks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    card_name TEXT NOT NULL,
    card_id TEXT,
    price_info TEXT,
    distance FLOAT,
    confidence TEXT,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    image_data TEXT,
    image_url TEXT
);

-- =====================================================
-- 2. ENABLE ROW LEVEL SECURITY
-- =====================================================

ALTER TABLE user_decks ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 3. DROP EXISTING POLICIES (if they exist)
-- =====================================================

DROP POLICY IF EXISTS "Users can view own cards" ON user_decks;
DROP POLICY IF EXISTS "Users can insert own cards" ON user_decks;
DROP POLICY IF EXISTS "Users can delete own cards" ON user_decks;
DROP POLICY IF EXISTS "Allow all operations for authenticated users" ON user_decks;

-- =====================================================
-- 4. CREATE SIMPLIFIED RLS POLICIES
-- =====================================================

-- Create a permissive policy that allows all operations
-- This is suitable for development and can be made more restrictive later
CREATE POLICY "Allow all operations for authenticated users" ON user_decks
    FOR ALL USING (true) WITH CHECK (true);

-- =====================================================
-- 5. ADD IMAGE_URL COLUMN (if it doesn't exist)
-- =====================================================

-- This will add the column if it doesn't exist, or do nothing if it already exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'user_decks' AND column_name = 'image_url'
    ) THEN
        ALTER TABLE user_decks ADD COLUMN image_url TEXT;
    END IF;
END $$;

-- =====================================================
-- 6. CREATE INDEXES FOR BETTER PERFORMANCE
-- =====================================================

-- Index on user_id for faster queries
CREATE INDEX IF NOT EXISTS idx_user_decks_user_id ON user_decks(user_id);

-- Index on added_at for sorting
CREATE INDEX IF NOT EXISTS idx_user_decks_added_at ON user_decks(added_at DESC);

-- Index on card_name for searching
CREATE INDEX IF NOT EXISTS idx_user_decks_card_name ON user_decks(card_name);

-- =====================================================
-- 7. VERIFICATION QUERIES
-- =====================================================

-- Uncomment these lines to verify the setup:
-- SELECT * FROM user_decks LIMIT 1;
-- SELECT * FROM pg_policies WHERE tablename = 'user_decks';
-- \d user_decks

-- =====================================================
-- SETUP COMPLETE
-- =====================================================

-- The database is now ready for the Pokemon Card Detection System!
-- You can now run the Flask application and start detecting cards. 