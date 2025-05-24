"""
HTML templates for the portfolio optimization web interface.
"""

# Main dashboard HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Portfolio Optimization DRL</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .container {
            display: flex;
            flex-direction: column;
            max-width: 1200px;
            margin: 0 auto;
        }
        .control-panel {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        button {
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
        }
        .train-btn {
            background-color: #4CAF50;
            color: white;
        }
        .evaluate-btn {
            background-color: #2196F3;
            color: white;
        }
        .stop-btn {
            background-color: #f44336;
            color: white;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        #status {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .idle {
            background-color: #e0e0e0;
            color: #333;
        }
        .running {
            background-color: #d4edda;
            color: #155724;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
        #log-container {
            background: #222;
            color: #eee;
            padding: 15px;
            border-radius: 5px;
            height: 600px;
            overflow-y: auto;
            margin-bottom: 20px;
            font-family: monospace;
        }
        #metrics-container {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: #f9f9f9;
            border-radius: 4px;
            padding: 10px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
        }
        .metric-label {
            font-weight: bold;
        }
        .metric-value {
            font-family: monospace;
        }
        .log-INFO {
            color: #87CEFA;
        }
        .log-WARNING {
            color: #FFD700;
        }
        .log-ERROR {
            color: #FF6347;
        }
        .log-SUCCESS {
            color: #90EE90;
        }
        .log-timestamp {
            color: #aaa;
            font-size: 0.8em;
            margin-right: 10px;
        }
        .results-panel {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .results-title {
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .results-content {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 15px;
        }
        .result-image {
            max-width: 48%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .loader {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 2s linear infinite;
            display: inline-block;
            margin-left: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Risk Optimization (DRL)</h1>
        
        <div class="control-panel">
            <h2>Control Panel</h2>
            
            <div class="button-group">
                <button id="trainButton" class="train-btn">Start Infinite Training</button>
                <button id="evaluateButton" class="evaluate-btn">Evaluate Model</button>
                <button id="stopButton" class="stop-btn" disabled>Stop Task</button>
            </div>
            
            <div id="status" class="idle">Status: Ready</div>
            
            <div class="training-info" style="background: #e8f4f8; padding: 10px; border-radius: 4px; margin: 10px 0; border-left: 4px solid #2196F3;">
                <strong>ℹ️ Training Mode:</strong> Training runs in continuous cycles. Each cycle completes a full training session, then automatically starts the next cycle after a brief pause. Use the <strong>STOP</strong> button to halt the infinite loop.
            </div>
            
            <div id="metrics-container">
                <h3>Metrics <span id="metricsLoader" class="loader hidden"></span></h3>
                <div class="metric-card">
                    <span class="metric-label">Current Operation:</span>
                    <span class="metric-value" id="currentOperation">-</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Current Step:</span>
                    <span class="metric-value" id="currentStep">-</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Best Return:</span>
                    <span class="metric-value" id="bestReturn">-</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Latest Sharpe Ratio:</span>
                    <span class="metric-value" id="lastSharpe">-</span>
                </div>
            </div>
        </div>
        
        <h3>Real-time Logs</h3>
        <div id="log-container"></div>
        
        <div id="results-panel" class="results-panel">
            <h2 class="results-title">Evaluation Results</h2>
            <div class="results-content">
                <img id="performance-img" class="result-image" alt="Portfolio Performance" />
                <img id="weights-img" class="result-image" alt="Portfolio Weights" />
            </div>
        </div>
    </div>

    <script>
        const trainButton = document.getElementById('trainButton');
        const evaluateButton = document.getElementById('evaluateButton');
        const stopButton = document.getElementById('stopButton');
        const statusDiv = document.getElementById('status');
        const logContainer = document.getElementById('log-container');
        const currentOperation = document.getElementById('currentOperation');
        const currentStep = document.getElementById('currentStep');
        const bestReturn = document.getElementById('bestReturn');
        const lastSharpe = document.getElementById('lastSharpe');
        const metricsLoader = document.getElementById('metricsLoader');
        const resultsPanel = document.getElementById('results-panel');
        const performanceImg = document.getElementById('performance-img');
        const weightsImg = document.getElementById('weights-img');
        
        let isProcessRunning = false;
        
        // Update button state function
        function updateButtonState(running) {
            isProcessRunning = running;
            trainButton.disabled = running;
            evaluateButton.disabled = running;
            stopButton.disabled = !running;
            
            if (running) {
                statusDiv.className = 'running';
                if (currentOperation.textContent === "Model Training") {
                    statusDiv.textContent = 'Status: Infinite Training Running...';
                } else {
                    statusDiv.textContent = 'Status: Running...';
                }
                metricsLoader.classList.remove('hidden');
            } else {
                statusDiv.className = 'idle';
                statusDiv.textContent = 'Status: Ready';
                metricsLoader.classList.add('hidden');
                currentOperation.textContent = '-';
            }
        }
        
        // Start training
        trainButton.addEventListener('click', async () => {
            if (isProcessRunning) return;
            
            // Clear log container and hide results panel
            logContainer.innerHTML = '';
            resultsPanel.classList.add('hidden');
            
            // Add initial message about infinite training
            const initMsg = document.createElement('div');
            initMsg.classList.add('log-INFO');
            initMsg.innerHTML = '<span class="log-timestamp">' + new Date().toLocaleTimeString() + '</span>Starting infinite training mode...';
            logContainer.appendChild(initMsg);
            
            try {
                updateButtonState(true);
                currentOperation.textContent = "Model Training (Infinite Loop)";
                
                const response = await fetch('/api/train', {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log('Infinite training started');
                    const successMsg = document.createElement('div');
                    successMsg.classList.add('log-SUCCESS');
                    successMsg.innerHTML = '<span class="log-timestamp">' + new Date().toLocaleTimeString() + '</span>Infinite training started successfully. Training will run continuously until stopped.';
                    logContainer.appendChild(successMsg);
                } else {
                    showError('Failed to start training: ' + data.message);
                    updateButtonState(false);
                }
                
            } catch (error) {
                showError('Error starting training: ' + error.message);
                updateButtonState(false);
            }
        });
        
        // Start evaluation
        evaluateButton.addEventListener('click', async () => {
            if (isProcessRunning) return;
            
            // Clear log container
            logContainer.innerHTML = '';
            
            // Hide results panel
            resultsPanel.classList.add('hidden');
            
            try {
                updateButtonState(true);
                currentOperation.textContent = "Model Evaluation";
                
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log('Evaluation started');
                    
                    // If direct response with timestamp, update images immediately
                    if (data.timestamp) {
                        // Refresh images with timestamp to prevent caching
                        const timestamp = new Date().getTime();
                        performanceImg.src = `/static/portfolio_performance.png?t=${timestamp}`;
                        weightsImg.src = `/static/portfolio_weights.png?t=${timestamp}`;
                        
                        // Show results panel after a short delay
                        setTimeout(() => {
                            resultsPanel.classList.remove('hidden');
                            updateButtonState(false);
                        }, 500);
                    }
                } else {
                    showError('Failed to start evaluation: ' + data.message);
                    updateButtonState(false);
                }
                
            } catch (error) {
                showError('Error starting evaluation: ' + error.message);
                updateButtonState(false);
            }
        });
        
        // Stop task
        stopButton.addEventListener('click', async () => {
            if (!isProcessRunning) return;
            
            try {
                const response = await fetch('/api/stop', {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log('Task stopped');
                    updateButtonState(false);
                } else {
                    showError('Failed to stop task: ' + data.message);
                }
                
            } catch (error) {
                showError('Error stopping task: ' + error.message);
            }
        });
        
        // Show error function
        function showError(message) {
            statusDiv.className = 'error';
            statusDiv.textContent = 'Error: ' + message;
            console.error(message);
            
            const errorLog = document.createElement('div');
            errorLog.classList.add('log-ERROR');
            errorLog.textContent = message;
            logContainer.appendChild(errorLog);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        // Set up log stream
        const eventSource = new EventSource('/api/stream');
        
        eventSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            
            // Ignore heartbeat messages
            if (data.message === "heartbeat") return;
            
            // Update logs
            const logEntry = document.createElement('div');
            logEntry.classList.add(`log-${data.level}`);
            
            // Add timestamp
            const timestamp = document.createElement('span');
            timestamp.classList.add('log-timestamp');
            timestamp.textContent = data.timestamp;
            
            // Message text node
            const messageText = document.createTextNode(data.message);
            
            // Combine
            logEntry.appendChild(timestamp);
            logEntry.appendChild(messageText);
            logContainer.appendChild(logEntry);
            
            // Keep only the most recent 1000 log entries
            while (logContainer.children.length > 1000) {
                logContainer.removeChild(logContainer.firstChild);
            }
            
            // Auto-scroll
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // Update metrics
            if (data.message.includes('Timestep')) {
                const stepMatch = data.message.match(/Timestep (\d+)/);
                if (stepMatch) {
                    currentStep.textContent = stepMatch[1];
                }
            } else if (data.message.includes('Training cycle #')) {
                const cycleMatch = data.message.match(/Training cycle #(\d+)/);
                if (cycleMatch) {
                    currentStep.textContent = `Cycle ${cycleMatch[1]}`;
                }
            } else if (data.message.includes('Total Return:') || data.message.includes('Total return:')) {
                const returnMatch = data.message.match(/Total [Rr]eturn: ([\d.-]+)%/);
                if (returnMatch) {
                    bestReturn.textContent = `${returnMatch[1]}%`;
                }
            } else if (data.message.includes('Sharpe Ratio:') || data.message.includes('Sharpe ratio:')) {
                const sharpeMatch = data.message.match(/[Ss]harpe [Rr]atio: ([\d.-]+)/);
                if (sharpeMatch) {
                    lastSharpe.textContent = sharpeMatch[1];
                }
            } else if (data.message.includes('cycle completed successfully')) {
                // Highlight successful cycle completion
                logEntry.style.backgroundColor = '#d4edda';
                logEntry.style.color = '#155724';
                logEntry.style.fontWeight = 'bold';
            } else if (data.message.includes('Waiting') && data.message.includes('next training cycle')) {
                // Highlight wait periods
                logEntry.style.backgroundColor = '#fff3cd';
                logEntry.style.color = '#856404';
            } else if (data.message.includes('chart') || 
                       data.message.includes('image') || 
                       data.message.includes('UI update')) {
                // Add timestamp to image URLs to prevent caching
                const timestamp = new Date().getTime();
                console.log('Loading images:', timestamp);
                // Wait briefly to ensure files are created
                setTimeout(() => {
                    performanceImg.src = `/static/portfolio_performance.png?t=${timestamp}`;
                    weightsImg.src = `/static/portfolio_weights.png?t=${timestamp}`;
                    resultsPanel.classList.remove('hidden');
                    console.log('Results panel shown');
                }, 1000);
            }
            
            // Update status for infinite training
            if (data.message.includes('Training completed') || 
                data.message.includes('Model evaluation completed') ||
                data.message.includes('stopped successfully') ||
                data.message.includes('Infinite training completed')) {
                updateButtonState(false);
            }
        };
        
        eventSource.onerror = function() {
            const errorMsg = document.createElement('div');
            errorMsg.classList.add('log-ERROR');
            errorMsg.textContent = 'Server connection lost. Reconnecting...';
            logContainer.appendChild(errorMsg);
        };
        
        // Check initial status
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                updateButtonState(data.is_running);
                if (data.is_running) {
                    currentOperation.textContent = data.operation;
                }
            })
            .catch(error => {
                showError('Error checking status: ' + error.message);
            });
    </script>
</body>
</html>
"""