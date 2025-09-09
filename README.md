# Portfolio Risk Optimization using Deep Reinforcement Learning

A portfolio optimization system using Deep Reinforcement Learning (DRL) with PPO algorithm, featuring a web interface for training and evaluation.

## Architecture

- **`train.py`** - Pure training script (no server conflicts)
- **`evaluation.py`** - Standalone evaluation module
- **`env.py`** - Portfolio environment with robust error handling
- **`models.py`** - Custom PPO policy models
- **`generate_scenario.py`** - Market data generation with fallbacks
- **`app/`** - Web application components
  - **`server.py`** - Main web server implementation
  - **`templates.py`** - HTML templates
  - **`utils.py`** - Logging and utility functions

## Quick Start

### Prerequisites

```bash
python=3.11
pip install -r requirements.txt
```

### 1. Start the Web Server
```bash
python main.py
```
Then open http://localhost:8080 in your browser.

### 2. Use the Web Interface
- **Start Infinite Training**: Begins continuous training cycles
- **Evaluate Model**: Tests trained model and generates charts
- **Stop Task**: Halts current operation

### 3. Manual Training (Optional)
```bash
python train.py
```

### 4. Manual Evaluation (Optional)
```bash
python evaluation.py
```

## Training Features

- **Infinite Loop Training**: Continuous training cycles via web interface
- **Multiple Environments**: Trains on diverse market scenarios
- **Model Checkpointing**: Saves best performing models
- **Real-time Monitoring**: Live logs and metrics
- **Early Stopping**: Prevents overfitting (when evaluations enabled)

## Evaluation Features

- **Safe Execution**: Runs independently without conflicting with training
- **Performance Metrics**: Sharpe ratio, max drawdown, returns
- **Visualizations**: Portfolio value and weights charts
- **Transaction Costs**: Realistic cost modeling

## File Structure

```
├── app/
│   ├── server.py          # Web server implementation  
│   ├── templates.py       # HTML templates
│   └── utils.py           # Logging and utilities
├── train.py               # Pure training script
├── evaluation.py          # Standalone evaluation
├── env.py                 # Robust portfolio environment
├── models.py              # Custom PPO models
├── generate_scenario.py   # Data generation with fallbacks
├── data/                  # Market data
├── logs/                  # Log files
├── static/                # Web assets
└── temp/                  # Temporary model files
```

## Usage Notes

- **Start Server First**: Always start the web server before using the web interface
- **Independent Operations**: Training and evaluation can run independently
- **Error Recovery**: System automatically handles data and dimension errors
- **Model Persistence**: Models are saved regularly during training
- **Safe Stopping**: Tasks can be safely stopped without corruption
