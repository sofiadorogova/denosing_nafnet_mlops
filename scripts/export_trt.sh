#!/bin/bash
# scripts/export_trt.sh <input.onnx> <output.plan>
set -e

ONNX_PATH="$1"
PLAN_PATH="$2"

if [ -z "$ONNX_PATH" ] || [ -z "$PLAN_PATH" ]; then
    echo "Usage: $0 <input.onnx> <output.plan>"
    exit 1
fi

echo "Converting $ONNX_PATH → $PLAN_PATH (FP16 + CUDA cores optimization)..."

docker run --gpus all --rm -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:24.04-py3 \
  bash -c "
    cd /workspace &&
    trtexec \
      --onnx='$ONNX_PATH' \
      --saveEngine='$PLAN_PATH' \
      --fp16 \
      --shapes="noisy:1x3x512x512"\
      --workspace=4096 \
      --verbose=0 \
      --best \
      --timingCacheFile=tensorrt.cache
  "

echo "TensorRT engine saved: $PLAN_PATH"
echo "To run inference: use tritonserver or onnxruntime with TensorRT execution provider"
