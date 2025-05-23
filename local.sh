trap "exit 1" INT

source ~/.bashrc

conda activate rl

uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload --reload-delay 0.5