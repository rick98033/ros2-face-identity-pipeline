# Model sources

No model weights or TensorRT engines are included.

## YuNet face detector

- Upstream: `opencv/opencv_zoo`
- File used by the historical implementation:
  `face_detection_yunet_2023mar.onnx`
- Recorded SHA-256:
  `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Model-directory license: MIT

The download URL in `scripts/build_face_engine.sh` is documentation only until
the resulting artifact is independently verified.

## AuraFace embedding model

- Upstream: `fal/AuraFace-v1`
- Embedding file: `glintr100.onnx`
- Pinned upstream revision:
  `af6d057c9b0ec4071d4c49c80e3539258798b609`
- Recorded SHA-256:
  `a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60`
- Model license: Apache-2.0

The repository contains several ONNX files. The embedding adapter requires
`glintr100.onnx`; detector and attribute models are not substitutes.

## NVIDIA PeopleNet

- Upstream: NVIDIA NGC TAO PeopleNet
- Historical variant: deployable ONNX v2.6.3
- Redistribution: not included here
- Terms: NVIDIA model terms apply independently of this repository

PeopleNet is optional. Another detector may publish compatible
`vision_msgs/Detection2DArray` messages for the included person tracker.
