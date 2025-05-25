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

## Key Improvements

### **✅ Server Conflict Resolution**
- **No Dual Servers**: Clean separation between web server and training
- **Pure Training Script**: `train.py` no longer starts its own server
- **Subprocess Execution**: Web interface runs training as isolated subprocess

### **✅ Dimension Error Fixes**
- **Robust State Construction**: Auto-padding/truncating to ensure 52-dimensional state
- **Safe Data Access**: Bounds checking prevents index out of range errors
- **NaN/Inf Handling**: Automatic cleaning of invalid data values
- **Fallback Data**: Dummy data generation when real data fails

### **✅ Evaluation Issues Fixed**
- **Removed Intermediate Evaluations**: No more evaluation during training cycles
- **Optional Final Evaluation**: Can be disabled to prevent dimension conflicts
- **Conflict-Free Evaluation**: Standalone evaluation runs safely alongside training
- **Robust Error Handling**: Graceful failure recovery

### **✅ Performance Optimizations**
- **Reduced Evaluation Frequency**: Less frequent callbacks for stability
- **Memory Management**: Better cleanup and resource management
- **Safe Model Loading**: File locking prevents concurrent access issues
- **Improved Logging**: Cleaner output with appropriate log levels

## Architecture Benefits

1. **No Server Conflicts**: Web server and training run independently
2. **Stable Training**: Removed problematic intermediate evaluations
3. **Robust Data Handling**: Automatic error recovery and fallbacks
4. **Clean Separation**: Each module has single responsibility
5. **Production Ready**: Proper error handling and logging

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