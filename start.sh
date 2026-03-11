#!/bin/bash
cd "$(dirname "$0")/backend"
source venv/bin/activate
ANALYSIS_ENABLE_CLASSIFIERS=1 uvicorn main:app --reload --port 8000
