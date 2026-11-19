#!/bin/zsh
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.
lsof -ti:8000 | xargs kill -9 2>/dev/null
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
