"""
Portfolio Optimization Web Server

This is the main entry point for the portfolio optimization web interface.
It starts a FastAPI server that provides a user interface for training and
evaluating portfolio optimization models.
"""
from app.server import start_server

def main():
    """Start the portfolio optimization web interface"""
    print("Starting Portfolio Optimization Web Interface...")
    start_server(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()