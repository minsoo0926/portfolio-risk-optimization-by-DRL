"""
Portfolio Optimization Web Server

This is the main entry point for the portfolio optimization web interface.
It starts a FastAPI server that provides a user interface for training and
evaluating portfolio optimization models.
"""
import logging
import uvicorn
from app.server import setup_app
from app.utils import setup_logger

# Create FastAPI app
app = setup_app()
logger = setup_logger()

def main(host="127.0.0.1", port=8000):
    """Start the portfolio optimization web interface"""
    logger.info(f"Web interface running at http://{host}:{port}")
    uvicorn.run("main:app", host=host, port=port, log_level="info", reload=True)

if __name__ == "__main__":
    main()