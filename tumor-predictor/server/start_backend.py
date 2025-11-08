#!/usr/bin/env python3
"""
Simple backend startup script - run from server directory
"""

import subprocess
import sys
import os

def start_backend():
    """Start the backend server"""
    print("Starting Tumor Predictor Backend...")
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Start the server using uvicorn
        print("Starting FastAPI server on http://0.0.0.0:8000...")
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\nBackend stopped by user")
    except Exception as e:
        print(f"Error starting backend: {e}")
        # Fallback: try running app.py directly
        try:
            print("Trying alternative startup method...")
            subprocess.run([sys.executable, "app.py"], check=True)
        except Exception as e2:
            print(f"Alternative startup also failed: {e2}")

if __name__ == "__main__":
    start_backend()

