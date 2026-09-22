"""Application entry point. Run: uvicorn app.api:app"""
from .server import create_app

app = create_app()
