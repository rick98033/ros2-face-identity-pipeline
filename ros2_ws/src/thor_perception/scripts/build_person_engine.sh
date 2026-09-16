#!/bin/bash
# Build PeopleNet TensorRT engine (run once after deployment to hardware)
#
# Usage: ./build_person_engine.sh
#
# This script converts the PeopleNet ONNX model to a TensorRT FP16 engine.
# Run this ONCE after deploying to the Jetson Thor hardware.
# The engine is hardware-specific and must be rebuilt if moving to different GPU.

set -euo pipefail

ONNX_MODEL="/opt/models/perception/resnet34_peoplenet.onnx"
# Use same naming convention as existing DeepStream-built engine
TRT_ENGINE="/opt/models/perception/resnet34_peoplenet.onnx_b1_gpu0_fp16.engine"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

echo "=============================================="
echo "PeopleNet TensorRT Engine Builder"
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
    echo ""
    echo "Download it with:"
    echo "  mkdir -p /opt/models/perception && cd /opt/models/perception"
    echo "  curl -L -o resnet34_peoplenet.onnx \\"
    echo "    'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_onnx_v2.6.3/files/resnet34_peoplenet.onnx'"
    exit 1
fi

# Check trtexec exists
if [ ! -x "$TRTEXEC" ]; then
    echo "ERROR: trtexec not found at $TRTEXEC"
    exit 1
fi

echo "Building TensorRT FP16 engine..."
echo "Input:  $ONNX_MODEL"
echo "Output: $TRT_ENGINE"
echo ""
echo "This may take 3-8 minutes on first run (PeopleNet is larger than YuNet)..."
echo ""

$TRTEXEC \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_ENGINE" \
    --fp16 \
    --builderOptimizationLevel=0 \
    --verbose 2>&1 | tee /tmp/trtexec_peoplenet.log

# Check engine was created AND has non-zero size
if [ -f "$TRT_ENGINE" ] && [ -s "$TRT_ENGINE" ]; then
    ENGINE_SIZE=$(du -h "$TRT_ENGINE" | cut -f1)
    ENGINE_HASH=$(sha256sum "$TRT_ENGINE" | cut -c1-8)
    echo ""
    echo "=============================================="
    echo "SUCCESS: TensorRT engine built"
    echo "Engine: $TRT_ENGINE ($ENGINE_SIZE)"
    echo "Hash:   $ENGINE_HASH (for provenance tracking)"
    echo "=============================================="
    echo ""
    echo "Person detection will now use GPU-accelerated TensorRT inference."
else
    echo ""
    echo "ERROR: Engine build failed (empty or missing). Check /tmp/trtexec_peoplenet.log"
    rm -f "$TRT_ENGINE"  # Clean up empty file
    exit 1
fi
