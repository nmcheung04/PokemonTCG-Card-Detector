#!/usr/bin/env python3
"""
Setup script for Pokemon Card Manager Web Application
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("🎴 Pokemon Card Manager Web Application Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_card_database():
    """Check if card database exists"""
    db_path = Path("../card_database.pkl")
    if db_path.exists():
        print("✅ Card database found")
        return True
    else:
        print("⚠️  Warning: card_database.pkl not found in parent directory")
        print("   The application will work but card detection may be limited")
        return False

def create_env_file():
    """Create .env file with user input"""
    print("\n🔧 Configuration Setup")
    print("-" * 30)
    
    env_content = []
    
    # Flask secret key
    flask_secret = input("Enter Flask secret key (or press Enter for default): ").strip()
    if not flask_secret:
        flask_secret = "dev-secret-key-change-in-production"
    env_content.append(f"FLASK_SECRET_KEY={flask_secret}")
    
    # Supabase configuration
    print("\n📊 Supabase Configuration:")
    print("1. Go to https://supabase.com and create a new project")
    print("2. Go to Settings > API to get your project URL and anon key")
    
    supabase_url = input("Enter Supabase URL: ").strip()
    if not supabase_url:
        print("⚠️  Supabase URL is required for the application to work")
        return False
    
    supabase_key = input("Enter Supabase anon key: ").strip()
    if not supabase_key:
        print("⚠️  Supabase anon key is required for the application to work")
        return False
    
    env_content.append(f"SUPABASE_URL={supabase_url}")
    env_content.append(f"SUPABASE_ANON_KEY={supabase_key}")
    
    # Pokemon TCG API (optional)
    pokemon_api_key = input("Enter Pokemon TCG API key (optional, press Enter to skip): ").strip()
    if pokemon_api_key:
        env_content.append(f"POKEMON_API_KEY={pokemon_api_key}")
    
    # Write .env file
    try:
        with open(".env", "w") as f:
            f.write("\n".join(env_content))
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def create_supabase_schema():
    """Provide SQL schema for Supabase setup"""
    print("\n🗄️  Supabase Database Setup")
    print("-" * 30)
    print("Run the following SQL in your Supabase SQL editor:")
    print()
    
    sql_schema = """
-- Create user_decks table
CREATE TABLE user_decks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    card_name TEXT NOT NULL,
    card_id TEXT,
    price_info TEXT,
    distance FLOAT,
    confidence TEXT,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    image_data TEXT
);

-- Enable Row Level Security
ALTER TABLE user_decks ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to see only their own cards
CREATE POLICY "Users can view own cards" ON user_decks
    FOR SELECT USING (auth.uid() = user_id);

-- Create policy to allow users to insert their own cards
CREATE POLICY "Users can insert own cards" ON user_decks
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Create policy to allow users to delete their own cards
CREATE POLICY "Users can delete own cards" ON user_decks
    FOR DELETE USING (auth.uid() = user_id);
"""
    
    print(sql_schema)
    
    # Save to file
    try:
        with open("supabase_schema.sql", "w") as f:
            f.write(sql_schema)
        print("✅ SQL schema saved to supabase_schema.sql")
    except Exception as e:
        print(f"⚠️  Could not save SQL schema: {e}")

def test_setup():
    """Test if the setup is working"""
    print("\n🧪 Testing Setup")
    print("-" * 30)
    
    # Test imports
    try:
        import flask
        import supabase
        import cv2
        import numpy as np
        import imagehash
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = ["SUPABASE_URL", "SUPABASE_ANON_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            return False
        else:
            print("✅ Environment variables loaded successfully")
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False
    
    return True

def main():
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check card database
    check_card_database()
    
    # Create environment file
    if not create_env_file():
        print("❌ Setup failed. Please check your configuration.")
        sys.exit(1)
    
    # Create Supabase schema
    create_supabase_schema()
    
    # Test setup
    if test_setup():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the SQL schema in your Supabase project")
        print("2. Start the application: python app.py")
        print("3. Open http://localhost:5000 in your browser")
        print("\nHappy collecting! 🎴")
    else:
        print("\n❌ Setup test failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 