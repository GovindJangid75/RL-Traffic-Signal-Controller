"""
server/app.py — OpenEnv multi-mode deployment entry point.
The validator requires a callable main() function with if __name__ == '__main__'.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.server import app  # noqa: F401 — re-exported for openenv runner
import uvicorn


def main():
    """Start the FastAPI server — called by the OpenEnv runner."""
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()
