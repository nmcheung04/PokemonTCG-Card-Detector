#!/usr/bin/env python3
"""
Test script to verify Supabase connection and authentication
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

def test_supabase_connection():
    """Test basic Supabase connection"""
    print("🔍 Testing Supabase connection...")
    
    try:
        # Get environment variables
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase environment variables")
            return False
        
        print(f"✅ Supabase URL: {supabase_url}")
        print(f"✅ Supabase Key: {supabase_key[:20]}...")
        
        # Create client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test basic connection
        print("🔄 Testing connection...")
        
        # Try to get auth settings
        auth_settings = supabase.auth.get_session()
        print("✅ Supabase client created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_auth_signup():
    """Test user registration"""
    print("\n🔐 Testing user registration...")
    
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test email and password
        test_email = "test@example.com"
        test_password = "testpassword123"
        
        print(f"🔄 Attempting to register: {test_email}")
        
        response = supabase.auth.sign_up({
            "email": test_email,
            "password": test_password,
            "options": {
                "email_confirm": False
            }
        })
        
        if response.user:
            print("✅ Registration test successful")
            print(f"   User ID: {response.user.id}")
            return True
        else:
            print("❌ Registration failed - no user returned")
            return False
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False

def test_database_connection():
    """Test database table access"""
    print("\n🗄️ Testing database connection...")
    
    try:
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Try to query the user_decks table
        print("🔄 Testing user_decks table access...")
        
        response = supabase.table('user_decks').select('*').limit(1).execute()
        print("✅ Database connection successful")
        print(f"   Table accessible: user_decks")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    print("🧪 Supabase Connection Test")
    print("=" * 40)
    
    # Test connection
    if not test_supabase_connection():
        print("\n❌ Basic connection failed. Check your environment variables.")
        return
    
    # Test database
    if not test_database_connection():
        print("\n❌ Database connection failed. Make sure you've run the SQL schema.")
        return
    
    # Test auth (optional - might fail if email already exists)
    print("\n⚠️  Auth test might fail if test email already exists")
    test_auth_signup()
    
    print("\n✅ All tests completed!")
    print("\nIf you see any errors above, please:")
    print("1. Check your .env file has correct Supabase credentials")
    print("2. Make sure you've run the SQL schema in Supabase")
    print("3. Check Supabase project settings")

if __name__ == "__main__":
    main() 