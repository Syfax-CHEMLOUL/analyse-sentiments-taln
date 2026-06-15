"""
Entry point — run with:
    python main.py
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn
from app.main import app  # noqa: F401 — re-exported for uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
