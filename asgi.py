"""ASGI deployment entry point for Vercel and other ASGI hosts."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"

# The Vercel Python runtime installs project dependencies, but does not install
# this src-layout project itself. Keep the bundled package importable before
# Streamlit starts executing app.py.
if not SOURCE_ROOT.is_dir():
    raise RuntimeError("Bundled Genome Firewall source directory is missing")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# Vercel imports this module and serves the top-level ASGI application. The
# Streamlit script remains app.py so local and Docker launch commands stay the
# same.
app = st.App(str(PROJECT_ROOT / "app.py"))
