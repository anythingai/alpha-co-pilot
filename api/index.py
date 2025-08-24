#!/usr/bin/env python3
"""
Alpha Co-Pilot - Vercel Serverless Entry Point
"""

import os
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Flask app
from app import app

# Vercel expects the app to be available as 'app'
# This allows Vercel to detect and run the Flask application
application = app

# For development
if __name__ == "__main__":
    app.run(debug=True)
