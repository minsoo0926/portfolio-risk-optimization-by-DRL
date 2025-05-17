"""
Utility functions for the portfolio optimization web application.
"""
import os
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for thread safety
import matplotlib.pyplot as plt
import numpy as np

def generate_sample_images():
    """
    Generate sample portfolio chart images for initial display.
    """
    print("Generating sample chart images...")
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created static directory")
    
    # Generate sample data
    days = 252
    np.random.seed(42)
    
    # Portfolio value data
    initial_capital = 10000
    daily_returns = np.random.normal(0.0005, 0.01, days)
    portfolio_values = [initial_capital]
    
    for r in daily_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    # Portfolio performance chart
    plt.figure(figsize=(14, 10))
    
    # Portfolio value chart
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values[1:], 'b-', linewidth=2)
    plt.title('Portfolio Value Over Time', fontsize=15)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True)
    
    # Daily returns chart
    plt.subplot(2, 1, 2)
    plt.plot(daily_returns * 100, 'g-', linewidth=1)
    plt.title('Daily Returns', fontsize=15)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save portfolio performance image
    performance_path = os.path.join('static', 'portfolio_performance.png')
    plt.savefig(performance_path)
    print(f"Saved portfolio performance chart: {performance_path}")
    
    # Portfolio weights chart
    plt.figure(figsize=(10, 6))
    labels = [f'Stock {i+1}' for i in range(10)]
    weights = np.array([0.1] * 10)
    plt.bar(labels, weights)
    plt.title('Portfolio Weights', fontsize=15)
    plt.ylabel('Weight', fontsize=12)
    plt.grid(True, axis='y')
    
    # Save portfolio weights image
    weights_path = os.path.join('static', 'portfolio_weights.png')
    plt.savefig(weights_path)
    print(f"Saved portfolio weights chart: {weights_path}")
    
    return {
        'performance': performance_path,
        'weights': weights_path
    }

def ensure_directories():
    """
    Create necessary directories for the application.
    """
    directories = ['static', 'logs', 'save']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    return True


import logging

def setup_logger(name='portfolio_optimization',log_file='logs/server.log'):
    # 로깅 설정
    stream_handler = logging.StreamHandler()
    stream_handler.setStream(open(os.devnull, 'w', encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            stream_handler
        ]
    )
    logger = logging.getLogger(name)
    
    return logger