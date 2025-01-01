.PHONY: install run clean test lint venv dev stop status

SHELL := /bin/bash
PYTHON = python3

install:
	. ./venv/bin/activate &&  pip install -r requirements.txt

run:
	@echo "Starting LLM server..."
	. ./venv/bin/activate && $(PYTHON) -m llama_cpp.server --port 5333 --n-gpu-layers 16 \
		--model ./models/ggml.gguf \
		--clip_model_path ./models/clip.gguf --chat_format minicpm-v-2.6 > llm.log 2>&1 & \
	echo $$! > llm.pid
	@echo "Starting FastAPI server..."
	. ./venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev:
	. ./venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 1122 --log-level debug

test:
	@echo "Running tests..."
	. ./venv/bin/activate && pytest ./tests/ -v --capture=no --log-cli-level=INFO

