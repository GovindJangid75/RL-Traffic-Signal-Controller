"""
server/app.py — OpenEnv multi-mode deployment entry point.
Re-exports the FastAPI app from api/server.py so the OpenEnv
validator can locate and launch the server via this standard path.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.server import app  # noqa: F401 — re-exported for openenv runner

__all__ = ["app"]
