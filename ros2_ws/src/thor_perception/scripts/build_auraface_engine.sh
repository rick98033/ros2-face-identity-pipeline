#!/bin/bash
# Build TensorRT engine for AuraFace face embedding model.
#
# Usage:
#   ./build_auraface_engine.sh
#
# Environment variables:
#   OPT_LEVEL: TensorRT optimization level (default: 3, use 0 if build fails)
#
# This script:
#   1. Downloads AuraFace model from HuggingFace (fal/AuraFace-v1)
#   2. Builds TensorRT engine with FP16 precision
#   3. Saves engine to /opt/models/face_id/auraface.engine

set -euo pipefail

MODEL_DIR="/opt/models/face_id"
ONNX_MODEL="$MODEL_DIR/auraface.onnx"
TRT_ENGINE="$MODEL_DIR/auraface.engine"
HF_REPO="fal/AuraFace-v1"
HF_FILENAME="glintr100.onnx"
AURAFACE_REVISION="af6d057c9b0ec4071d4c49c80e3539258798b609"
AURAFACE_SHA256="a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60"
OPT_LEVEL="${OPT_LEVEL:-3}"

echo "============================================"
echo "AuraFace TensorRT Engine Builder"
echo "============================================"
echo "Model dir:  $MODEL_DIR"
echo "HF repo:    $HF_REPO"
echo "HF file:    $HF_FILENAME"
echo "HF revision: $AURAFACE_REVISION"
echo "Opt level:  $OPT_LEVEL"
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Download from HuggingFace if not present
if [ ! -f "$ONNX_MODEL" ]; then
    echo "Downloading AuraFace model from HuggingFace..."
    pip3 install -q huggingface_hub

    python3 << 'EOF'
import shutil
from huggingface_hub import hf_hub_download

repo_id = "fal/AuraFace-v1"
revision = "af6d057c9b0ec4071d4c49c80e3539258798b609"
src = hf_hub_download(repo_id=repo_id, filename="glintr100.onnx", revision=revision)
shutil.copy(src, "/opt/models/face_id/auraface.onnx")
EOF

    # Verify download succeeded
    if [ ! -f "$ONNX_MODEL" ]; then
        echo "ERROR: ONNX model not found after download"
        echo "Check repo structure at https://huggingface.co/$HF_REPO"
        exit 1
    fi

fi

echo "${AURAFACE_SHA256}  ${ONNX_MODEL}" | sha256sum -c -

echo ""
echo "ONNX model: $(ls -lh "$ONNX_MODEL")"
echo ""

# Build TensorRT engine
TRTEXEC=/usr/src/tensorrt/bin/trtexec

if [ ! -f "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    echo "Ensure TensorRT is installed correctly"
    exit 1
fi

echo "Building TensorRT engine (optimization level $OPT_LEVEL)..."
echo "This may take several minutes..."
echo ""

$TRTEXEC \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_ENGINE" \
    --fp16 \
    --builderOptimizationLevel="$OPT_LEVEL" \
    --verbose 2>&1 | tee /tmp/trtexec_auraface.log

if [ -f "$TRT_ENGINE" ] && [ -s "$TRT_ENGINE" ]; then
    echo ""
    echo "============================================"
    echo "SUCCESS: Engine built!"
    echo "============================================"
    echo "Engine: $(ls -lh "$TRT_ENGINE")"
    echo ""
    echo "The auraface_id_node will now use GPU inference."
else
    echo ""
    echo "============================================"
    echo "ERROR: Engine build failed"
    echo "============================================"
    echo "Build log: /tmp/trtexec_auraface.log"
    echo ""
    echo "Try with lower optimization level:"
    echo "  OPT_LEVEL=0 $0"
    echo ""
    rm -f "$TRT_ENGINE"
    exit 1
fi
