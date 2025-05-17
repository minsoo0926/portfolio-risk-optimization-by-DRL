# Portfolio Risk Optimization Using Deep Reinforcement Learning

A Deep Reinforcement Learning (DRL) model for portfolio risk optimization. This project utilizes stock market data to learn optimal portfolio weights, maximizing risk-adjusted returns by considering returns, volatility, and transaction costs.

## Key Features

- **Proximal Policy Optimization (PPO)** algorithm for deep reinforcement learning
- Web interface for training and evaluation control
- Real-time learning progress monitoring
- Robustness evaluation across various market conditions
- Portfolio performance and weight visualization

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/portfolio-risk-optimization-by-DRL.git
cd portfolio-risk-optimization-by-DRL

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Launch Web Interface

```bash
python app/server.py
```

Access the following features via web browser at http://localhost:8000:

1. **Train Model**: Click "Train Model" to start DRL model training
2. **Evaluate Model**: Click "Evaluate Model" to assess trained model performance
3. **Stop Task**: Terminate currently running tasks
4. **Real-time Logs**: Monitor training and evaluation processes in real-time
5. **Result Visualization**: View portfolio performance and weight visualizations after evaluation

## Model Architecture

- **State Space**: 52-dimensional (daily returns, moving average returns, volatility, relative volume, macro indicators, and previous actions for 10 stocks)
- **Action Space**: 10-dimensional continuous (portfolio weights for each stock)
- **Reward Function**: Portfolio return - volatility penalty - transaction cost penalty

## Experimental Results

The model demonstrates stable performance across various market conditions, especially excelling in risk-adjusted returns (Sharpe Ratio).

## Project Structure

- `app/` - Web application and server code
  - `server.py` - FastAPI server implementation
  - `templates.py` - Web interface HTML templates
  - `utils.py` - Web application utility functions
- `data/` - Financial time series data
  - `derived/` - Processed data files
  - `price/` - Stock price data
  - `train_set/` - Training datasets
- `debug/` - Debugging utilities and logs
- `logs/` - Application logs
- `save/` - Saved model checkpoints
- `static/` - Static files for web interface (images, etc.)
- `train.py` - Model training script
- `evaluation.py` - Model evaluation script
- `main.py` - Core functions and model implementation
- `requirements.txt` - Python dependencies