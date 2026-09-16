#!/bin/bash
# Build YuNet TensorRT engine (run once after deployment to hardware)
#
# Usage: ./build_face_engine.sh
#
# This script converts the YuNet ONNX model to a TensorRT FP16 engine.
# Run this ONCE after deploying to the Jetson Thor hardware.
# The engine is hardware-specific and must be rebuilt if moving to different GPU.

set -euo pipefail

ONNX_MODEL="/opt/models/face/yunet.onnx"
TRT_ENGINE="/opt/models/face/yunet.engine"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

echo "=============================================="
echo "YuNet TensorRT Engine Builder"
echo "=============================================="

# Check if engine already exists
if [ -f "$TRT_ENGINE" ]; then
    echo "Engine already exists at $TRT_ENGINE"
    echo "Delete it first if you want to rebuild: rm $TRT_ENGINE"
    exit 0
fi

# Check ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX model not found at $ONNX_MODEL"
    echo "Download it first: wget -O $ONNX_MODEL https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    exit 1
fi

echo "${YUNET_SHA256}  ${ONNX_MODEL}" | sha256sum -c -

# Check trtexec exists
if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    exit 1
fi

echo "Building TensorRT FP16 engine..."
echo "Input:  $ONNX_MODEL"
echo "Output: $TRT_ENGINE"
echo ""
echo "This may take 2-5 minutes on first run..."
echo ""

$TRTEXEC \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_ENGINE" \
    --fp16 \
    --builderOptimizationLevel=0 \
    --verbose 2>&1 | tee /tmp/trtexec_yunet.log

# Check engine was created AND has non-zero size
if [ -f "$TRT_ENGINE" ] && [ -s "$TRT_ENGINE" ]; then
    ENGINE_SIZE=$(du -h "$TRT_ENGINE" | cut -f1)
    echo ""
    echo "=============================================="
    echo "SUCCESS: TensorRT engine built"
    echo "Engine: $TRT_ENGINE ($ENGINE_SIZE)"
    echo "=============================================="
    echo ""
    echo "Face detection will now use GPU-accelerated TensorRT inference."
else
    echo ""
    echo "ERROR: Engine build failed (empty or missing). Check /tmp/trtexec_yunet.log"
    rm -f "$TRT_ENGINE"  # Clean up empty file
    exit 1
fi
