#!/usr/bin/env python3
"""
Run the FREE RAG Support KB System
"""
import os
import sys
from src.api.main import app
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", 8000))
    print("=" * 60)
    print("🚀 FREE Intelligent Support KB RAG System")
    print("=" * 60)
    print(f"Mode: Free-only (Local LLM + ChromaDB)")
    print(f"Port: http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=True  # Auto-reload on code changes
    )