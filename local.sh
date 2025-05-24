trap "exit 1" INT

uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload --reload-delay 0.5