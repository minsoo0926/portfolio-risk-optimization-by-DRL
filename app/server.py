"""
Portfolio Optimization Server Module

This module provides a FastAPI server for the portfolio optimization application.
It handles API routes, static file serving, and real-time log streaming.
"""
import logging
import queue
import json
from datetime import datetime
import os
from pathlib import Path
import asyncio
import subprocess
import threading
import time

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for thread safety
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, APIRouter, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.utils import setup_logger

logger = setup_logger()

# Log message queue
log_queue = queue.Queue(maxsize=1000)

# Process management 
global current_process, is_training, is_evaluating
current_process = None
is_training = False
is_evaluating = False

# Queue logging handler
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    
    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = {
                "level": record.levelname,
                "message": msg,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            }
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            pass  # Ignore if queue is full

# Add queue handler
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(queue_handler)

# HTML template
from app.templates import HTML_TEMPLATE

# Create Router
router = APIRouter()

# Model training function
def run_training():
    global current_process, is_training, is_evaluating
    
    if is_training or is_evaluating:
        return {"status": "error", "message": "Another task is already running."}
    
    is_training = True
    logger.info("="*50)
    logger.info("Starting infinite portfolio optimization training")
    logger.info("="*50)
    
    try:
        # Infinite loop training function
        def continuous_training():
            global current_process, is_training
            cycle_count = 0
            
            try:
                while is_training:
                    cycle_count += 1
                    logger.info(f"Starting training cycle #{cycle_count}")
                    
                    # Run training process
                    process = subprocess.Popen(
                        ["python", "train.py"],  # Direct execution of train.py
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        env=os.environ.copy()
                    )
                    
                    # Update current process
                    current_process = process
                    
                    # Stream output in real-time
                    try:
                        for line in iter(process.stdout.readline, ''):
                            if not line:
                                break
                            if is_training:  # Check if still training
                                logger.info(line.strip())
                            else:
                                break
                    except Exception as e:
                        logger.error(f"Error reading process output: {e}")
                    
                    # Wait for process completion
                    try:
                        return_code = process.wait(timeout=30)  # 30 second timeout for cleanup
                        
                        if return_code == 0:
                            logger.info(f"Training cycle #{cycle_count} completed successfully")
                        else:
                            logger.error(f"Training cycle #{cycle_count} failed with exit code {return_code}")
                            
                    except subprocess.TimeoutExpired:
                        logger.warning("Process cleanup timeout, forcing termination")
                        try:
                            process.kill()
                            process.wait()
                        except:
                            pass
                    
                    # Check if training should continue
                    if not is_training:
                        logger.info("Training stopped by user")
                        break
                    
                    # Wait before next cycle
                    logger.info("Waiting 30 seconds before starting next training cycle...")
                    for i in range(30):
                        if not is_training:
                            logger.info("Training stopped during wait period")
                            break
                        time.sleep(1)
                    
                    if not is_training:
                        break
                
                logger.info(f"Infinite training completed after {cycle_count} cycles")
                
            except Exception as e:
                logger.error(f"Error in continuous training: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                # Clean up
                current_process = None
                is_training = False
                logger.info("Training process cleanup completed")
        
        # Start infinite training thread
        training_thread = threading.Thread(target=continuous_training, daemon=True)
        training_thread.start()
        
        # Return immediately to allow background execution
        return {"status": "success", "message": "Infinite training started in background."}
        
    except Exception as e:
        logger.error(f"Error starting infinite training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        current_process = None
        is_training = False
        return {"status": "error", "message": f"Error starting training: {str(e)}"}

# Model evaluation function
def run_evaluation():
    global current_process, is_training, is_evaluating
    # Check if another task is already running
    if is_training or is_evaluating:
        logger.warning("Evaluation request rejected: Another task is already running")
        return {"status": "error", "message": "Another task is already running."}
        
    try:
        # Update state
        is_evaluating = True
        logger.info("="*50)
        logger.info("Starting portfolio optimization model evaluation")
        logger.info("="*50)
        
        # Start evaluation process
        logger.info("Portfolio simulation starting")
        logger.info("UI update: Evaluation started")
        
        # Run evaluation process in background
        def run_evaluation_process():
            global current_process, is_evaluating
            try:
                # Run evaluation as subprocess
                process = subprocess.Popen(
                    ["python", "-c", "from evaluation import evaluate_model; evaluate_model('./ppo_portfolio', initial_capital=10000, log_weights=True)"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy()  # 현재 환경 변수 유지
                )
                
                # Update current process reference
                current_process = process
                
                # Log output
                for line in iter(process.stdout.readline, ''):
                    if line and is_evaluating:
                        logger.info(line.strip())
                
                # Check process completion
                if process.poll() is not None:
                    if process.returncode == 0:
                        logger.info("Model evaluation completed successfully")
                    else:
                        logger.error(f"Model evaluation failed: exit code {process.returncode}")
                
                # Copy image files to static folder
                import shutil
                for img_file in ['portfolio_performance.png', 'portfolio_weights.png']:
                    if os.path.exists(img_file):
                        shutil.copy(img_file, os.path.join("static", img_file))
                        logger.info(f"Image file copied: {img_file} -> static/{img_file}")
                
                # Complete message
                logger.info("Model evaluation completed successfully")
                logger.info("UI update: Chart images generated")
                
                # Update state
                current_process = None
                is_evaluating = False
                
            except Exception as e:
                logger.error(f"Error during evaluation process: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                current_process = None
                is_evaluating = False
        
        # Run evaluation process in background
        evaluation_thread = threading.Thread(target=run_evaluation_process)
        evaluation_thread.daemon = True
        evaluation_thread.start()
        
        # Return immediately
        return {"status": "success", "message": "Model evaluation started."}
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        current_process = None
        is_evaluating = False
        return {"status": "error", "message": f"Error during evaluation: {str(e)}"}

def generate_portfolio_charts():
    """Generate portfolio performance and weights charts"""
    logger.info("Generating portfolio charts")
    
    # Generate sample data
    days = 252  # Trading days in a year
    initial_capital = 10000
    np.random.seed(int(time.time()) % 1000)  # Different seed each time
    daily_returns = np.random.normal(0.0005, 0.01, days)  # Mean 0.05%, std 1%
    portfolio_values = [initial_capital]
    
    # Accumulate data
    for r in daily_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + r))
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    annualized_return = ((1 + total_return/100) ** (252/days) - 1) * 100
    sharpe_ratio = annualized_return / (np.std(daily_returns) * np.sqrt(252) * 100)
    
    # Create charts using Agg backend
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
    plt.savefig('portfolio_performance.png')
    
    # Result dictionary
    results = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio
    }
    
    # Log results
    logger.info(f"Simulation completed: Total return {results['total_return']:.2f}%")
    logger.info(f"Annualized return: {results['annualized_return']:.2f}%, Sharpe ratio: {results['sharpe_ratio']:.4f}")
    
    # Generate weights chart
    plt.figure(figsize=(10, 6))
    np.random.seed(int(time.time()) % 1000)  # Different seed 
    labels = [f'Stock {i+1}' for i in range(10)]
    weights = np.random.dirichlet(np.ones(10), size=1)[0]  # Random weights
    plt.bar(labels, weights)
    plt.title('Portfolio Weights', fontsize=15)
    plt.ylabel('Weight', fontsize=12)
    plt.grid(True, axis='y')
    plt.savefig('portfolio_weights.png')
    
    # Copy result images to static folder
    try:
        import shutil
        if not os.path.exists("static"):
            os.makedirs("static")
            logger.info("Static folder created")
            
        # Copy portfolio performance image
        if os.path.exists('portfolio_performance.png'):
            shutil.copy('portfolio_performance.png', os.path.join("static", "portfolio_performance.png"))
            logger.info(f"Image file copied: portfolio_performance.png -> static/portfolio_performance.png")
        
        # Copy weights image
        if os.path.exists('portfolio_weights.png'):
            shutil.copy('portfolio_weights.png', os.path.join("static", "portfolio_weights.png"))
            logger.info(f"Image file copied: portfolio_weights.png -> static/portfolio_weights.png")
            
    except Exception as e:
        logger.warning(f"Error copying image files: {str(e)}")
        import traceback
        logger.warning(traceback.format_exc())
    
    return results

# Stop current process
def stop_current_process():
    global current_process, is_training, is_evaluating
    
    if not is_training and not is_evaluating:
        logger.warning("No tasks are currently running.")
        return {"status": "success", "message": "No tasks are currently running."}
    
    try:
        logger.info("Stopping current task...")
        
        # Set stop flags first
        was_training = is_training
        was_evaluating = is_evaluating
        is_training = False
        is_evaluating = False
        
        # Get current process reference
        proc = current_process
        
        if proc is not None:
            try:
                # First, try graceful termination
                logger.info("Sending termination signal to process...")
                proc.terminate()
                
                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=5)
                    logger.info("Process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning("Process did not terminate gracefully, forcing kill...")
                    proc.kill()
                    try:
                        proc.wait(timeout=3)
                        logger.info("Process force killed")
                    except subprocess.TimeoutExpired:
                        logger.error("Could not kill process")
                            
            except Exception as e:
                logger.error(f"Error terminating process: {e}")
        
        # Clean up process reference
        current_process = None
        
        # Log what was stopped
        if was_training:
            logger.info("Infinite training loop stopped successfully")
            return {"status": "success", "message": "Infinite training stopped."}
        elif was_evaluating:
            logger.info("Model evaluation stopped successfully")
            return {"status": "success", "message": "Model evaluation stopped."}
        else:
            return {"status": "success", "message": "Task stopped."}
        
    except Exception as e:
        logger.error(f"Error stopping task: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Ensure cleanup even on error
        current_process = None
        is_training = False
        is_evaluating = False
        
        return {"status": "error", "message": f"Error stopping task: {str(e)}"}

# API endpoints
@router.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTML_TEMPLATE

@router.get("/api/status")
async def get_status():
    return {
        "is_running": is_training or is_evaluating,
        "operation": "Model training" if is_training else "Model evaluation" if is_evaluating else "-"
    }

@router.post("/api/train")
async def train_model(background_tasks: BackgroundTasks):
    global is_training, is_evaluating
    
    if is_training or is_evaluating:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Another task is already running."}
        )
    
    logger.info("Received request to start infinite training")
    
    try:
        # Start infinite training
        result = run_training()
        
        if result["status"] == "success":
            logger.info("Infinite training started successfully")
            return {"status": "success", "message": "Infinite training started. Use STOP button to halt."}
        else:
            logger.error(f"Failed to start training: {result['message']}")
            return JSONResponse(
                status_code=500,
                content=result
            )
            
    except Exception as e:
        logger.error(f"Error in train API: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

@router.post("/api/evaluate")
async def evaluate_model(background_tasks: BackgroundTasks):
    if is_training or is_evaluating:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Another task is already running."}
        )
    
    # Run evaluation as background task
    background_tasks.add_task(run_evaluation)
    return {"status": "success", "message": "Model evaluation started."}

@router.post("/api/stop")
async def stop_process():
    logger.info("Stop request received")
    result = stop_current_process()
    
    if result["status"] == "error":
        # Consider already stopped as success to avoid frontend errors
        if "No tasks are currently running" in result["message"]:
            return {"status": "success", "message": "Task already stopped."}
        
        return JSONResponse(
            status_code=400,
            content=result
        )
    
    return result

@router.get("/api/stream")
async def stream_logs():
    async def generate():
        try:
            while True:
                try:
                    # Get log message from queue (non-blocking)
                    try:
                        msg = log_queue.get_nowait()
                        yield f"data: {json.dumps(msg)}\n\n"
                    except queue.Empty:
                        # Send heartbeat if queue is empty
                        heartbeat = {
                            "level": "INFO",
                            "message": "heartbeat",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        yield f"data: {json.dumps(heartbeat)}\n\n"
                    
                    # Short delay for socket stability
                    await asyncio.sleep(0.0)
                except Exception as e:
                    logger.error(f"Stream processing error: {str(e)}")
                    # Continue after short delay
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Handle client disconnection
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Stream connection error: {str(e)}")

    return StreamingResponse(generate(), media_type="text/event-stream")

def setup_app():
    """Set up and configure the FastAPI application"""
    # Create required directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Generate sample images on startup
    try:
        pass
        # generate_portfolio_charts()
        # logger.info("Sample chart images generated")
    except Exception as e:
        logger.warning(f"Error generating sample images: {str(e)}")
        # Try to copy existing images to static folder
        try:
            import shutil
            for img_file in ['portfolio_performance.png', 'portfolio_weights.png']:
                if os.path.exists(img_file):
                    shutil.copy(img_file, os.path.join("static", img_file))
        except Exception as e:
            logger.warning(f"Error copying image files: {str(e)}")
    
    # Create FastAPI app
    app = FastAPI(title="Portfolio Optimization DRL")
    
    # Register router
    app.include_router(router)
    
    # Setup static file serving
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    return app

import uvicorn
# Create FastAPI app
app = setup_app()

def start_server(host="127.0.0.1", port=8000):
    """Start the web server"""
        
    logger.info(f"Web interface running at http://{host}:{port}")
    uvicorn.run("app.server:app", host=host, port=port, log_level="info", reload=True)

if __name__ == "__main__":
    start_server()