import logging
from utils import start_server

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("포트폴리오 최적화 웹 인터페이스 시작")
    
    # FastAPI 웹 서버 시작
    start_server()