#!/usr/bin/env python3
"""
Helper script to get a Firebase ID token for testing.

Usage:
    python get_firebase_token.py email@example.com password
"""

import sys
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Get your Firebase Web API Key from Firebase Console > Project Settings > General > Web API Key
FIREBASE_WEB_API_KEY = os.getenv("FIREBASE_WEB_API_KEY", "")

def get_firebase_token(email: str, password: str) -> dict:
    """Sign in with email/password and get ID token."""
    
    if not FIREBASE_WEB_API_KEY:
        print("ERROR: FIREBASE_WEB_API_KEY not set in .env file")
        print("Get it from: Firebase Console > Project Settings > General > Web API Key")
        sys.exit(1)
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        print(f"ERROR: {response.status_code}")
        print(response.json())
        sys.exit(1)
    
    data = response.json()
    return {
        "idToken": data["idToken"],
        "refreshToken": data["refreshToken"],
        "expiresIn": data["expiresIn"],
        "localId": data["localId"]
    }

def create_test_user(email: str, password: str) -> dict:
    """Create a new Firebase user for testing."""
    
    if not FIREBASE_WEB_API_KEY:
        print("ERROR: FIREBASE_WEB_API_KEY not set")
        sys.exit(1)
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        print(f"ERROR creating user: {response.status_code}")
        print(response.json())
        sys.exit(1)
    
    data = response.json()
    return {
        "idToken": data["idToken"],
        "refreshToken": data["refreshToken"],
        "expiresIn": data["expiresIn"],
        "localId": data["localId"]
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Login:  python get_firebase_token.py email@example.com password")
        print("  Signup: python get_firebase_token.py --create email@example.com password")
        sys.exit(1)
    
    if sys.argv[1] == "--create":
        if len(sys.argv) < 4:
            print("Usage: python get_firebase_token.py --create email@example.com password")
            sys.exit(1)
        email = sys.argv[2]
        password = sys.argv[3]
        print(f"Creating user: {email}")
        result = create_test_user(email, password)
        print(f"\n✅ User created!")
    else:
        email = sys.argv[1]
        password = sys.argv[2]
        print(f"Signing in: {email}")
        result = get_firebase_token(email, password)
        print(f"\n✅ Signed in!")
    
    print(f"\nUID: {result['localId']}")
    print(f"Expires in: {result['expiresIn']} seconds")
    print(f"\n🔑 ID Token (use in Authorization header):")
    print(f"Bearer {result['idToken']}")
    print(f"\n📋 Copy this for curl:")
    print(f"-H \"Authorization: Bearer {result['idToken']}\"")
    print(f"\n🔄 Refresh Token (save this to get new tokens later):")
    print(result['refreshToken'])
