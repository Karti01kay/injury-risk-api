"""
start.py — run from repo root on Render.
Sets sys.path so api/ imports work, trains model if needed,
then launches uvicorn.
"""

import sys
import os

# Add api/ to path so all imports inside main.py resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "api"))

# Also run from api/ directory so relative file paths work
os.chdir(os.path.join(os.path.dirname(__file__), "api"))

# Now import and run the app
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")