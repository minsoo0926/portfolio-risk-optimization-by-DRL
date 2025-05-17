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
from fastapi import FastAPI, APIRouter, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("portfolio_optimization")

# 로그 메시지를 저장할 큐
log_queue = queue.Queue(maxsize=1000)

# 현재 실행 중인 프로세스 관리
current_process = None
is_training = False
is_evaluating = False

# 큐 로깅 핸들러
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
            pass  # 큐가 가득 차면 메시지 무시

# 큐 핸들러 추가
queue_handler = QueueHandler(log_queue)
queue_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(queue_handler)

# 모델 훈련 함수
def run_training():
    global current_process, is_training, is_evaluating
    
    if is_training or is_evaluating:
        return {"status": "error", "message": "다른 작업이 이미 실행 중입니다."}
    
    is_training = True
    logger.info("="*50)
    logger.info("포트폴리오 최적화 훈련 시작")
    logger.info("="*50)
    
    try:
        # 훈련 프로세스 실행
        current_process = subprocess.Popen(
            ["python", "-c", "from train import main; main()"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 훈련 출력 로깅
        for line in iter(current_process.stdout.readline, ''):
            logger.info(line.strip())
        
        # 프로세스 완료 대기
        current_process.wait()
        
        if current_process.returncode == 0:
            logger.info("훈련 완료: 성공")
        else:
            logger.error(f"훈련 실패: 종료 코드 {current_process.returncode}")
        
        current_process = None
        is_training = False
        return {"status": "success", "message": "훈련이 완료되었습니다."}
        
    except Exception as e:
        logger.error(f"훈련 실행 중 오류: {str(e)}")
        current_process = None
        is_training = False
        return {"status": "error", "message": f"훈련 실행 중 오류: {str(e)}"}

# 모델 평가 함수
def run_evaluation():
    global current_process, is_training, is_evaluating
    
    if is_training or is_evaluating:
        return {"status": "error", "message": "다른 작업이 이미 실행 중입니다."}
    
    is_evaluating = True
    logger.info("="*50)
    logger.info("포트폴리오 최적화 모델 평가 시작")
    logger.info("="*50)
    
    try:
        # 평가 프로세스 실행
        current_process = subprocess.Popen(
            ["python", "-c", "from evaluation import evaluate_model; evaluate_model('ppo_portfolio_best')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 평가 출력 로깅
        for line in iter(current_process.stdout.readline, ''):
            logger.info(line.strip())
        
        # 프로세스 완료 대기
        current_process.wait()
        
        if current_process.returncode == 0:
            logger.info("모델 평가 완료: 성공")
        else:
            logger.error(f"모델 평가 실패: 종료 코드 {current_process.returncode}")
        
        current_process = None
        is_evaluating = False
        return {"status": "success", "message": "모델 평가가 완료되었습니다."}
        
    except Exception as e:
        logger.error(f"모델 평가 실행 중 오류: {str(e)}")
        current_process = None
        is_evaluating = False
        return {"status": "error", "message": f"모델 평가 실행 중 오류: {str(e)}"}

# 훈련 작업 취소
def stop_current_process():
    global current_process, is_training, is_evaluating
    
    if current_process is None:
        return {"status": "error", "message": "실행 중인 작업이 없습니다."}
    
    try:
        current_process.terminate()
        time.sleep(1)
        
        if current_process.poll() is None:  # 아직 종료되지 않았으면 강제 종료
            current_process.kill()
        
        logger.info("작업이 중지되었습니다.")
        current_process = None
        is_training = False
        is_evaluating = False
        return {"status": "success", "message": "작업이 중지되었습니다."}
        
    except Exception as e:
        logger.error(f"작업 중지 중 오류: {str(e)}")
        return {"status": "error", "message": f"작업 중지 중 오류: {str(e)}"}

# FastAPI 관련 설정
router = APIRouter()

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>포트폴리오 최적화 DRL</title>
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
        <h1>포트폴리오 위험 최적화 (DRL)</h1>
        
        <div class="control-panel">
            <h2>제어 패널</h2>
            
            <div class="button-group">
                <button id="trainButton" class="train-btn">모델 훈련</button>
                <button id="evaluateButton" class="evaluate-btn">모델 평가</button>
                <button id="stopButton" class="stop-btn" disabled>작업 중지</button>
            </div>
            
            <div id="status" class="idle">상태: 대기 중</div>
            
            <div id="metrics-container">
                <h3>메트릭 <span id="metricsLoader" class="loader hidden"></span></h3>
                <div class="metric-card">
                    <span class="metric-label">현재 작업:</span>
                    <span class="metric-value" id="currentOperation">-</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">현재 스텝:</span>
                    <span class="metric-value" id="currentStep">-</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">최고 수익률:</span>
                    <span class="metric-value" id="bestReturn">-</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">최근 Sharpe 비율:</span>
                    <span class="metric-value" id="lastSharpe">-</span>
                </div>
            </div>
        </div>
        
        <h3>실시간 로그</h3>
        <div id="log-container"></div>
        
        <div id="results-panel" class="results-panel hidden">
            <h2 class="results-title">평가 결과</h2>
            <div class="results-content">
                <img id="performance-img" class="result-image" alt="포트폴리오 성과" />
                <img id="weights-img" class="result-image" alt="포트폴리오 가중치" />
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
        
        // 버튼 상태 업데이트 함수
        function updateButtonState(running) {
            isProcessRunning = running;
            trainButton.disabled = running;
            evaluateButton.disabled = running;
            stopButton.disabled = !running;
            
            if (running) {
                statusDiv.className = 'running';
                statusDiv.textContent = '상태: 실행 중...';
                metricsLoader.classList.remove('hidden');
            } else {
                statusDiv.className = 'idle';
                statusDiv.textContent = '상태: 대기 중';
                metricsLoader.classList.add('hidden');
            }
        }
        
        // 훈련 시작
        trainButton.addEventListener('click', async () => {
            if (isProcessRunning) return;
            
            // 로그 컨테이너 초기화 및 결과 패널 숨기기
            logContainer.innerHTML = '';
            resultsPanel.classList.add('hidden');
            
            try {
                updateButtonState(true);
                currentOperation.textContent = "모델 훈련";
                
                const response = await fetch('/api/train', {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log('훈련 시작됨');
                } else {
                    showError('훈련 시작 실패: ' + data.message);
                    updateButtonState(false);
                }
                
            } catch (error) {
                showError('훈련 시작 중 오류: ' + error.message);
                updateButtonState(false);
            }
        });
        
        // 평가 시작
        evaluateButton.addEventListener('click', async () => {
            if (isProcessRunning) return;
            
            // 로그 컨테이너 초기화
            logContainer.innerHTML = '';
            
            try {
                updateButtonState(true);
                currentOperation.textContent = "모델 평가";
                
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log('평가 시작됨');
                } else {
                    showError('평가 시작 실패: ' + data.message);
                    updateButtonState(false);
                }
                
            } catch (error) {
                showError('평가 시작 중 오류: ' + error.message);
                updateButtonState(false);
            }
        });
        
        // 작업 중지
        stopButton.addEventListener('click', async () => {
            if (!isProcessRunning) return;
            
            try {
                const response = await fetch('/api/stop', {
                    method: 'POST',
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log('작업 중지됨');
                    updateButtonState(false);
                } else {
                    showError('작업 중지 실패: ' + data.message);
                }
                
            } catch (error) {
                showError('작업 중지 중 오류: ' + error.message);
            }
        });
        
        // 오류 표시 함수
        function showError(message) {
            statusDiv.className = 'error';
            statusDiv.textContent = '오류: ' + message;
            console.error(message);
            
            const errorLog = document.createElement('div');
            errorLog.classList.add('log-ERROR');
            errorLog.textContent = message;
            logContainer.appendChild(errorLog);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
        
        // 로그 스트림 설정
        const eventSource = new EventSource('/api/stream');
        
        eventSource.onmessage = function(e) {
            const data = JSON.parse(e.data);
            
            // 하트비트 메시지는 무시
            if (data.message === "heartbeat") return;
            
            // 로그 업데이트
            const logEntry = document.createElement('div');
            logEntry.classList.add(`log-${data.level}`);
            
            // 타임스탬프 추가
            const timestamp = document.createElement('span');
            timestamp.classList.add('log-timestamp');
            timestamp.textContent = data.timestamp;
            
            // 메시지 텍스트 노드
            const messageText = document.createTextNode(data.message);
            
            // 조합
            logEntry.appendChild(timestamp);
            logEntry.appendChild(messageText);
            logContainer.appendChild(logEntry);
            
            // 스크롤 자동 이동
            logContainer.scrollTop = logContainer.scrollHeight;
            
            // 메트릭 업데이트
            if (data.message.includes('Timestep')) {
                const stepMatch = data.message.match(/Timestep (\d+)/);
                if (stepMatch) {
                    currentStep.textContent = stepMatch[1];
                }
            } else if (data.message.includes('총 수익률:')) {
                const returnMatch = data.message.match(/총 수익률: ([\d.-]+)/);
                if (returnMatch) {
                    bestReturn.textContent = `${returnMatch[1]}%`;
                }
            } else if (data.message.includes('Sharpe Ratio:')) {
                const sharpeMatch = data.message.match(/Sharpe Ratio: ([\d.-]+)/);
                if (sharpeMatch) {
                    lastSharpe.textContent = sharpeMatch[1];
                }
            } else if (data.message.includes('포트폴리오 성과 차트가')) {
                // 이미지 업데이트를 위한 타임스탬프 추가 (캐시 방지)
                const timestamp = new Date().getTime();
                performanceImg.src = `/static/portfolio_performance.png?t=${timestamp}`;
                weightsImg.src = `/static/portfolio_weights.png?t=${timestamp}`;
                resultsPanel.classList.remove('hidden');
            }
            
            // 상태 업데이트
            if (data.message.includes('훈련 완료') || 
                data.message.includes('모델 평가 완료') ||
                data.message.includes('실행 실패')) {
                updateButtonState(false);
            }
        };
        
        eventSource.onerror = function() {
            const errorMsg = document.createElement('div');
            errorMsg.classList.add('log-ERROR');
            errorMsg.textContent = '서버 연결이 끊어졌습니다. 재연결 중...';
            logContainer.appendChild(errorMsg);
        };
        
        // 초기 상태 확인
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                updateButtonState(data.is_running);
                if (data.is_running) {
                    currentOperation.textContent = data.operation;
                }
            })
            .catch(error => {
                showError('상태 확인 중 오류: ' + error.message);
            });
    </script>
</body>
</html>
"""

# API 엔드포인트
@router.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTML_TEMPLATE

@router.get("/api/status")
async def get_status():
    return {
        "is_running": is_training or is_evaluating,
        "operation": "모델 훈련" if is_training else "모델 평가" if is_evaluating else "-"
    }

@router.post("/api/train")
async def train_model(background_tasks: BackgroundTasks):
    if is_training or is_evaluating:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "다른 작업이 이미 실행 중입니다."}
        )
    
    # 백그라운드 작업으로 훈련 실행
    background_tasks.add_task(run_training)
    return {"status": "success", "message": "훈련이 시작되었습니다."}

@router.post("/api/evaluate")
async def evaluate_model(background_tasks: BackgroundTasks):
    if is_training or is_evaluating:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "다른 작업이 이미 실행 중입니다."}
        )
    
    # 백그라운드 작업으로 평가 실행
    background_tasks.add_task(run_evaluation)
    return {"status": "success", "message": "모델 평가가 시작되었습니다."}

@router.post("/api/stop")
async def stop_process():
    result = stop_current_process()
    
    if result["status"] == "error":
        return JSONResponse(
            status_code=400,
            content=result
        )
    
    return result

@router.get("/api/stream")
async def stream_logs():
    async def generate():
        while True:
            try:
                # 큐에서 로그 메시지 가져오기 (블로킹 없이)
                try:
                    msg = log_queue.get_nowait()
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    # 큐가 비어있으면 하트비트 메시지 전송
                    heartbeat = {
                        "level": "INFO",
                        "message": "heartbeat",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                
                # 짧은 딜레이
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"스트림 에러: {str(e)}")
                break

    return StreamingResponse(generate(), media_type="text/event-stream")

def setup_app():
    # 필요한 디렉토리 생성
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # 정적 파일 관리 (평가 결과 그래프 등)
    # 결과 이미지를 static 폴더로 복사
    try:
        # 평가 이미지 파일을 static 폴더로 관리
        import shutil
        for img_file in ['portfolio_performance.png', 'portfolio_weights.png']:
            if os.path.exists(img_file):
                shutil.copy(img_file, os.path.join("static", img_file))
    except Exception as e:
        logger.warning(f"이미지 파일 복사 중 오류: {str(e)}")
    
    # FastAPI 앱 생성
    app = FastAPI(title="Portfolio Optimization DRL")
    
    # 라우터 등록
    app.include_router(router)
    
    # 정적 파일 서빙 설정 (이미지, CSS 등을 위한 설정)
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    return app

def get_logger():
    return logger

# FastAPI 앱 시작을 위한 함수
def start_server():
    import uvicorn
    
    # 필요한 import가 함수 내부에 있어야 원활한 실행이 가능
    app = setup_app()
    
    logger.info("웹 인터페이스가 http://localhost:8000 에서 실행 중입니다")
    uvicorn.run(app, host="0.0.0.0", port=8000)