#!/bin/bash
# scripts/start_triton.sh
set -e

echo "Starting Triton Inference Server with NAFNet & DnCNN..."

docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$(pwd)/models:/models" \
  nvcr.io/nvidia/tritonserver:24.04-py3 \
  tritonserver \
    --model-repository=/models \
    --log-verbose=1 \
    --strict-model-config=false

echo "Triton server stopped."
