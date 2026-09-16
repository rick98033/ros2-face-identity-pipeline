"""Face crop and alignment for AuraFace embedding.

This module provides functions to extract and align face crops from images
using 5-point landmarks. The alignment follows the InsightFace/ArcFace
standard template for optimal face recognition performance.

Input landmark convention (subject-perspective from YuNet):
- "Left eye" = subject's left eye (appears on VIEWER'S RIGHT for frontal face)
- Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth

ArcFace template convention (viewer-perspective):
- Index 0 = viewer's left (lower x), Index 1 = viewer's right (higher x)
- This module converts from subject to viewer convention internally.
"""

import numpy as np
import cv2

from thor_telemetry import get_logger as _get_structured_logger, ErrorCode
_structured_logger = _get_structured_logger("alignment")

# InsightFace standard 5-point landmarks destination template
# These are the target positions for a 112x112 aligned face crop
ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],  # left eye (subject's left)
    [73.5318, 51.5014],  # right eye (subject's right)
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)

# Scaled for 192x192 output (AuraFace uses this size)
ARCFACE_DST_192 = ARCFACE_DST_112 * (192.0 / 112.0)

# Default template (192x192 for AuraFace)
ARCFACE_DST = ARCFACE_DST_192


def estimate_transform(src_landmarks: np.ndarray) -> np.ndarray | None:
    """Estimate similarity transform from 5-point landmarks to ArcFace template.

    Uses cv2.estimateAffinePartial2D which computes a similarity transform
    (rotation, uniform scale, translation) that best maps source landmarks
    to the ArcFace destination template.

    Args:
        src_landmarks: Shape (5, 2) array of landmark coordinates.
                       Order: left_eye, right_eye, nose_tip, left_mouth, right_mouth

    Returns:
        2×3 affine matrix for use with cv2.warpAffine, or None if estimation fails
        (e.g., insufficient valid landmarks or degenerate configuration).
    """
    if src_landmarks.shape != (5, 2):
        return None

    M, inliers = cv2.estimateAffinePartial2D(src_landmarks, ARCFACE_DST)

    if M is None:
        return None

    # Require at least 2 inlier points for a valid transform
    # Note: Reduced from 3 to 2 to handle small faces and slight angles
    if inliers is None or inliers.sum() < 2:
        return None

    return M


def align_face(
    frame: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 192
) -> np.ndarray | None:
    """Crop and align face for AuraFace embedding.

    Extracts an aligned face crop from the input frame using a similarity
    transform computed from 5-point landmarks. The output is suitable for
    input to AuraFace or other InsightFace-compatible models.

    Args:
        frame: BGR image as numpy array, shape (H, W, 3)
        landmarks: 5-point landmarks from YuNet detector.
                   Can be shape (10,) flat array or (5, 2) reshaped.
                   Order: left_eye_x, left_eye_y, right_eye_x, right_eye_y,
                          nose_x, nose_y, left_mouth_x, left_mouth_y,
                          right_mouth_x, right_mouth_y
        output_size: Output crop size in pixels (default 112 for InsightFace)

    Returns:
        Aligned face crop as BGR image, shape (output_size, output_size, 3),
        or None if alignment fails (bad landmarks or transform estimation error).

    Note:
        Landmarks must be in the same coordinate system as the frame (source
        frame coordinates). If frame has been resized, landmarks must be scaled
        accordingly.
    """
    if frame is None or frame.size == 0:
        _structured_logger.emit_failure(
            operation="align_face",
            error_code=ErrorCode.BAD_REQUEST,
            error_detail="Frame is None or empty",
            trigger="invalid_input",
        )
        return None

    # Handle flat array input (10 elements -> 5x2)
    if landmarks.shape == (10,):
        landmarks = landmarks.reshape(5, 2)
    elif landmarks.shape != (5, 2):
        _structured_logger.emit_failure(
            operation="align_face",
            error_code=ErrorCode.BAD_REQUEST,
            error_detail=f"Invalid landmark shape: {landmarks.shape}",
            trigger="invalid_landmarks",
        )
        return None

    # Convert from subject-perspective to viewer-perspective for ArcFace template
    # Input: [subj_left_eye, subj_right_eye, nose, subj_left_mouth, subj_right_mouth]
    # Template expects: [viewer_left, viewer_right, nose, viewer_left_mouth, viewer_right_mouth]
    # For frontal face: subj_left appears on viewer's right, so swap eyes and mouth corners
    landmarks = landmarks[[1, 0, 2, 4, 3], :]

    # Estimate similarity transform
    M = estimate_transform(landmarks)
    if M is None:
        return None

    # Apply affine warp
    # borderValue=0 fills out-of-bounds areas with black
    aligned = cv2.warpAffine(
        frame,
        M,
        (output_size, output_size),
        borderValue=0
    )

    return aligned


def validate_aligned_crop(aligned: np.ndarray, max_black_ratio: float = 0.3) -> bool:
    """Validate that an aligned crop has sufficient face content.

    Checks for excessive black pixels which indicate a failed or marginal
    alignment (face near frame edge, bad transform, etc.).

    Args:
        aligned: Aligned face crop from align_face()
        max_black_ratio: Maximum allowed ratio of black pixels (default 0.3)

    Returns:
        True if crop is valid, False if too much black content.
    """
    if aligned is None:
        return False

    black_pixels = np.sum(aligned == 0)
    total_pixels = aligned.size
    black_ratio = black_pixels / total_pixels

    return black_ratio <= max_black_ratio
