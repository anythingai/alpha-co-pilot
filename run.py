#!/usr/bin/env python3
"""
Alpha Co-Pilot Application Runner
Simple script to run the Flask app with proper configuration
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_environment():
    """Check if required environment variables are set
    Accepts either:
      - Gemini: GEMINI_API_KEY (preferred)
      - Azure: AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
    """
    has_gemini = bool(os.getenv('GEMINI_API_KEY'))
    has_azure = bool(os.getenv('AZURE_OPENAI_API_KEY')) and bool(os.getenv('AZURE_OPENAI_ENDPOINT'))

    if not (has_gemini or has_azure):
        print("❌ Missing required environment variables:")
        print("   - Provide GEMINI_API_KEY (recommended), or")
        print("   - Provide AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        print("\n💡 Please update your .env file with the correct values")
        return False

    return True

def main():
    """Main application runner"""
    print("Starting Alpha Co-Pilot...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Import and run the Flask app
    try:
        from app import app
        
        print("Environment configured")
        print("Flask app loaded")
        print("\nServer starting at: http://localhost:5000")
        print("")
        print("Dashboard: http://localhost:5000")
        print("\nAPI Endpoints:")
        print("   - POST /api/generate (Generate alpha content)")
        print("   - POST /api/share (Share to Discord)")
        print("   - GET /health (Health check)")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Run the Flask app
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import Flask app: {e}")
        print("💡 Make sure you've installed requirements: pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Alpha Co-Pilot stopped!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()