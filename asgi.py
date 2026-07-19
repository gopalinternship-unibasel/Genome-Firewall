"""ASGI deployment entry point for Vercel and other ASGI hosts."""

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent

# Vercel imports this module and serves the top-level ASGI application. The
# Streamlit script remains app.py so local and Docker launch commands stay the
# same.
app = st.App(str(PROJECT_ROOT / "app.py"))
