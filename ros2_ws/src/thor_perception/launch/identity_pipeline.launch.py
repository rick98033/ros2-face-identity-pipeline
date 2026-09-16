"""Curated launch surface for the face identity reference pipeline."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "camera_ip",
            default_value=os.environ.get("CAMERA_IP", "127.0.0.1"),
            description="RTSP camera host; must be set for a real deployment",
        ),
        DeclareLaunchArgument("rtsp_transport", default_value="udp"),
        DeclareLaunchArgument("rtsp_latency_ms", default_value="100"),
        DeclareLaunchArgument("camera_hfov_deg", default_value="65.0"),
        DeclareLaunchArgument("min_confidence", default_value="0.4"),
        DeclareLaunchArgument("face_min_confidence", default_value="0.6"),
        DeclareLaunchArgument("face_min_size_px", default_value="40"),
        DeclareLaunchArgument("face_id_threshold", default_value="0.5"),
        DeclareLaunchArgument("face_id_shadow_mode", default_value="false"),
        DeclareLaunchArgument("enable_face_id", default_value="true"),
        DeclareLaunchArgument("publish_metrics", default_value="true"),
        DeclareLaunchArgument("perception_min_fps", default_value="3.0"),
    ]

    nodes = [
        Node(
            package="thor_perception",
            executable="camera_ingest",
            name="camera_ingest_node",
            output="screen",
            parameters=[{
                "rtsp_uri": [
                    "rtsp://",
                    LaunchConfiguration("camera_ip"),
                    ":8554/camera",
                ],
                "rtsp_transport": LaunchConfiguration("rtsp_transport"),
                "rtsp_latency_ms": LaunchConfiguration("rtsp_latency_ms"),
                "camera_frame_id": "camera_head_optical_frame",
                "camera_hfov_deg": LaunchConfiguration("camera_hfov_deg"),
                "publish_timing": True,
                "publish_metrics": True,
                "min_fps_threshold": LaunchConfiguration("perception_min_fps"),
            }],
        ),
        Node(
            package="thor_perception",
            executable="person_detector",
            name="person_detector_node",
            output="screen",
            parameters=[{
                "engine_path": "/opt/models/perception/resnet34_peoplenet.onnx_b1_gpu0_fp16.engine",
                "onnx_path": "/opt/models/perception/resnet34_peoplenet.onnx",
                "min_confidence": LaunchConfiguration("min_confidence"),
                "nms_iou_threshold": 0.4,
                "publish_metrics": LaunchConfiguration("publish_metrics"),
            }],
        ),
        Node(
            package="thor_perception",
            executable="person_tracker",
            name="person_tracker_node",
            output="screen",
            parameters=[{
                "camera_hfov_deg": LaunchConfiguration("camera_hfov_deg"),
                "publish_metrics": LaunchConfiguration("publish_metrics"),
            }],
        ),
        Node(
            package="thor_perception",
            executable="perception_state_server",
            name="perception_state_server",
            output="screen",
        ),
        Node(
            package="thor_perception",
            executable="face_detection",
            name="face_detection_node",
            output="screen",
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                "engine_path": "/opt/models/face/yunet.engine",
                "onnx_path": "/opt/models/face/yunet.onnx",
                "input_size": 640,
                "min_confidence": LaunchConfiguration("face_min_confidence"),
                "min_face_size_px": LaunchConfiguration("face_min_size_px"),
                "publish_frames": LaunchConfiguration("enable_face_id"),
                "crop_buffer_enabled": True,
                "association.require_confirmed": True,
                "publish_metrics": LaunchConfiguration("publish_metrics"),
            }],
        ),
        Node(
            package="thor_perception",
            executable="face_state_server",
            name="face_state_server",
            output="screen",
        ),
        Node(
            package="thor_perception",
            executable="auraface_id",
            name="auraface_id_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_face_id")),
            parameters=[{
                "engine_path": "/opt/models/face_id/auraface.engine",
                "onnx_path": "/opt/models/face_id/auraface.onnx",
                "threshold": LaunchConfiguration("face_id_threshold"),
                "shadow_mode": LaunchConfiguration("face_id_shadow_mode"),
            }],
        ),
        Node(
            package="thor_perception",
            executable="authorization",
            name="authorization_node",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_face_id")),
            parameters=[{
                "acquisition_timeout_sec": 5.0,
                "acquisition_window_sec": 0.5,
                "acquisition_min_consistent": 2,
                "min_face_match_score": 0.5,
                "min_face_match_margin": 0.08,
                "require_quality_ok": True,
                "min_association_confidence": 0.3,
                "publish_rate_hz": 10.0,
            }],
        ),
    ]

    return LaunchDescription(arguments + nodes)
