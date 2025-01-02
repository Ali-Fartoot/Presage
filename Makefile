PYTHON = python3
USE_GPU ?= false

.PHONY: install run test


install:
        . ./venv/bin/activate &&  pip install -r requirements.txt

run:
        @echo "Starting LLM server..."
        . ./venv/bin/activate && $(PYTHON) -m llama_cpp.server --port 5333  \
                $(if $(filter true,$(USE_GPU)),--n_gpu_layers 16,--n_gpu_layers 0) \
                --model ./models/ggml.gguf \
                --clip_model_path ./models/clip.gguf --chat_format minicpm-v-2.6 > llm.log 2>&1 & \
        echo $$! > llm.pid
        @echo "Starting FastAPI server..."
        . ./venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
        @echo "Running tests..."
        . ./venv/bin/activate && pytest ./tests/ -v --capture=no --log-cli-level=INFO
