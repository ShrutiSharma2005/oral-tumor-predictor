#!/usr/bin/env python3
"""
Simple backend startup script
"""

import subprocess
import sys
import os
import time

def start_backend():
    """Start the backend server"""
    print("Starting Tumor Predictor Backend...")
    
    # Change to server directory
    server_dir = os.path.join(os.path.dirname(__file__), 'server')
    os.chdir(server_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    try:
        # Start the server
        print("Starting FastAPI server...")
        subprocess.run([sys.executable, "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting backend: {e}")
    except KeyboardInterrupt:
        print("\nBackend stopped by user")

if __name__ == "__main__":
    start_backend()
