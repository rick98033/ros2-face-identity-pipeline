#!/usr/bin/env python3
"""Camera Ingestion Node - Single RTSP decode for all downstream perception.

This node is the ONLY component that decodes the RTSP stream from X1.
All perception nodes (person_detector, face_detector, etc.) subscribe to
the ROS topics published here instead of opening their own RTSP connections.

Pipeline: RTSP → GStreamer/NVDEC → ROS 2 sensor_msgs/Image

Publishes:
  /sensors/camera/head/rgb/image (sensor_msgs/Image)
  /sensors/camera/head/rgb/camera_info (sensor_msgs/CameraInfo)
  /sensors/camera/head/rgb/timing (thor_msgs/CameraFrameTiming)

TF:
  base_link → head_link → camera_head_link → camera_head_optical_frame

Environment Variables:
  CAMERA_IP: RTSP camera host (reference default: 127.0.0.1)
  RTSP_TRANSPORT: udp or tcp (default: udp)
  RTSP_LATENCY_MS: GStreamer jitter buffer latency (default: 100)
"""

import json
import os
import sys
import time
import queue
import threading
import math
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from thor_msgs.msg import CameraFrameTiming, PerceptionMetrics
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# GStreamer imports
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib


# =============================================================================
# Configuration
# =============================================================================

# Reconnection backoff sequence (seconds)
RECONNECT_BACKOFF = [0.5, 1.0, 2.0, 4.0, 10.0]
MAX_RECONNECT_ATTEMPTS = 30
RECONNECT_TOTAL_TIMEOUT = 120.0

# Stall detection
DEFAULT_STALL_TIMEOUT_SEC = 5.0

# Metrics
DEFAULT_METRICS_INTERVAL_SEC = 5.0


# =============================================================================
# RTP Timestamp Unwrapper
# =============================================================================

class RTPUnwrapper:
    """Handle 32-bit RTP timestamp wrap (~13 hours at 90kHz).

    RTP timestamps are 32-bit counters at 90kHz clock rate.
    They wrap every 2^32 / 90000 ≈ 47721 seconds ≈ 13.25 hours.
    This class tracks wraps and provides a 64-bit unwrapped value.
    """

    def __init__(self):
        self.last_rtp: Optional[int] = None
        self.wrap_count: int = 0

    def unwrap(self, rtp_timestamp: int) -> int:
        """Unwrap 32-bit RTP timestamp to 64-bit monotonic value."""
        if self.last_rtp is not None:
            # Detect wrap: large backward jump indicates wrap
            # Use 0xC0000000 and 0x40000000 as thresholds (3/4 and 1/4 of range)
            if self.last_rtp > 0xC0000000 and rtp_timestamp < 0x40000000:
                self.wrap_count += 1
        self.last_rtp = rtp_timestamp
        return (self.wrap_count << 32) | rtp_timestamp

    def reset(self):
        """Reset unwrapper state (called on stream reconnect)."""
        # Note: We do NOT reset wrap_count to maintain monotonicity across reconnects
        self.last_rtp = None


# =============================================================================
# Frame Data Container
# =============================================================================

@dataclass
class FrameData:
    """Container for decoded frame and timing metadata."""
    bgr_data: np.ndarray
    width: int
    height: int

    # Timing (nanoseconds)
    appsink_receive_ns: int

    # RTP timestamps
    rtp_timestamp_raw: int
    rtp_timestamp_unwrapped: int

    # GStreamer PTS (nanoseconds, may be GST_CLOCK_TIME_NONE)
    gst_pts_ns: int

    # Sender frame sequence (from RTP, monotonic)
    frame_seq: int


# =============================================================================
# Camera Ingest Node
# =============================================================================

class CameraIngestNode(Node):
    """ROS 2 node that ingests RTSP video and publishes sensor_msgs/Image."""

    def __init__(self):
        super().__init__("camera_ingest_node")

        # Initialize GStreamer
        Gst.init(None)

        # Parameters
        default_camera_ip = os.environ.get("CAMERA_IP", "127.0.0.1")
        self.declare_parameter("rtsp_uri", f"rtsp://{default_camera_ip}:8554/camera")
        self.declare_parameter("rtsp_transport", os.environ.get("RTSP_TRANSPORT", "udp"))
        self.declare_parameter("rtsp_latency_ms", int(os.environ.get("RTSP_LATENCY_MS", "100")))
        self.declare_parameter("camera_frame_id", "camera_head_optical_frame")
        self.declare_parameter("camera_hfov_deg", 65.0)
        self.declare_parameter("camera_info_url", "")  # optional HTTP camera-info endpoint
        self.declare_parameter("stall_timeout_sec", DEFAULT_STALL_TIMEOUT_SEC)
        self.declare_parameter("metrics_interval_sec", DEFAULT_METRICS_INTERVAL_SEC)
        self.declare_parameter("publish_timing", True)
        self.declare_parameter("publish_metrics", True)
        self.declare_parameter("min_fps_threshold", float(os.environ.get("PERCEPTION_MIN_FPS", "5.0")))

        # Get parameters
        self.rtsp_uri = self.get_parameter("rtsp_uri").value
        self.rtsp_transport = self.get_parameter("rtsp_transport").value
        self.rtsp_latency_ms = self.get_parameter("rtsp_latency_ms").value
        self.camera_frame_id = self.get_parameter("camera_frame_id").value
        self.camera_hfov_deg = self.get_parameter("camera_hfov_deg").value
        camera_info_url = self.get_parameter("camera_info_url").value
        self.stall_timeout_sec = self.get_parameter("stall_timeout_sec").value

        # Fetch camera intrinsics from X1 if URL provided
        self._remote_camera_info: Optional[dict[str, Any]] = None
        if camera_info_url:
            self._remote_camera_info = self._fetch_camera_info(camera_info_url)
        elif default_camera_ip:
            # Try default URL based on camera IP
            default_url = f"http://{default_camera_ip}:8555/camera_info"
            self._remote_camera_info = self._fetch_camera_info(default_url)
        self.metrics_interval_sec = self.get_parameter("metrics_interval_sec").value
        self.publish_timing = self.get_parameter("publish_timing").value
        self.publish_metrics = self.get_parameter("publish_metrics").value
        self.min_fps_threshold = self.get_parameter("min_fps_threshold").value

        # QoS: best-effort, depth=1 for real-time (drop old, keep newest)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Timing topic uses reliable for diagnostics
        timing_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self.image_pub = self.create_publisher(
            Image, "/sensors/camera/head/rgb/image", sensor_qos)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "/sensors/camera/head/rgb/camera_info", sensor_qos)

        if self.publish_timing:
            self.timing_pub = self.create_publisher(
                CameraFrameTiming, "/sensors/camera/head/rgb/timing", timing_qos)
        else:
            self.timing_pub = None

        if self.publish_metrics:
            self.metrics_pub = self.create_publisher(
                PerceptionMetrics, "/perception/camera_ingest/metrics", 10)
            self.metrics_timer = self.create_timer(
                self.metrics_interval_sec, self._publish_metrics)
        else:
            self.metrics_pub = None

        # Static TF broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        # Frame queue: maxsize=1 for drop-old/keep-newest
        self.frame_queue: queue.Queue[FrameData] = queue.Queue(maxsize=1)

        # RTP unwrapper
        self.rtp_unwrapper = RTPUnwrapper()

        # State
        self.pipeline: Optional[Gst.Pipeline] = None
        self.glib_loop: Optional[GLib.MainLoop] = None
        self.glib_thread: Optional[threading.Thread] = None
        self.running = False

        # Timing state
        self.last_frame_time = 0.0  # monotonic
        self.last_rtp_arrival_ns = 0  # For network latency estimation
        self.last_rtp_timestamp = 0

        # Counters (monotonic, survive reconnects)
        self.thor_frame_seq = 0
        self.dropped_frames_total = 0
        self.reconnect_count_total = 0

        # RTSP instrumentation counters
        self.frames_decoded = 0          # Frames received from appsink
        self.frames_published = 0        # Frames published to ROS
        self.appsink_drops = 0           # Frames dropped by queue overflow

        # Metrics window tracking (for rate-based health checks)
        self._metrics_window_start = time.monotonic()
        self._metrics_window_drops = 0   # Drops in current metrics window
        self._metrics_window_decoded = 0 # Frames decoded in current window

        # Pipeline state
        self.pipeline_state = CameraFrameTiming.STATE_UNKNOWN

        # Frame dimensions (updated on first frame)
        self.frame_width = 640
        self.frame_height = 480

        # Latency tracking for metrics
        self.latencies_ms: list[float] = []
        self.frame_times: list[float] = []

        # Publish timer (runs at ~30Hz, publishes newest available frame)
        self.publish_timer = self.create_timer(1.0 / 30.0, self._publish_frame)

        self.get_logger().info(f"Camera ingest initialized: {self.rtsp_uri}")

    def _publish_static_tf(self):
        """Publish static transforms for camera frame hierarchy."""
        now = self.get_clock().now()
        transforms = []

        # base_link → head_link (stub as identity, will be dynamic when head motion available)
        t1 = TransformStamped()
        t1.header.stamp = now.to_msg()
        t1.header.frame_id = "base_link"
        t1.child_frame_id = "head_link"
        t1.transform.translation.x = 0.0
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.5  # Approximate head height
        t1.transform.rotation.w = 1.0
        transforms.append(t1)

        # head_link → camera_head_link (physical mount offset)
        t2 = TransformStamped()
        t2.header.stamp = now.to_msg()
        t2.header.frame_id = "head_link"
        t2.child_frame_id = "camera_head_link"
        t2.transform.translation.x = 0.05  # Camera forward offset
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.1  # Camera above head center
        t2.transform.rotation.w = 1.0
        transforms.append(t2)

        # camera_head_link → camera_head_optical_frame (REP-103 convention)
        # Optical frame: Z forward, X right, Y down
        # Standard camera frame: X forward, Y left, Z up
        # Rotation: -90° around Z, then -90° around X
        t3 = TransformStamped()
        t3.header.stamp = now.to_msg()
        t3.header.frame_id = "camera_head_link"
        t3.child_frame_id = "camera_head_optical_frame"
        t3.transform.translation.x = 0.0
        t3.transform.translation.y = 0.0
        t3.transform.translation.z = 0.0
        # Quaternion for optical frame convention
        # This rotates from camera body to optical frame
        t3.transform.rotation.x = -0.5
        t3.transform.rotation.y = 0.5
        t3.transform.rotation.z = -0.5
        t3.transform.rotation.w = 0.5
        transforms.append(t3)

        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info("Static TF published: base_link → head_link → camera_head_link → optical_frame")

    def _build_gst_pipeline(self) -> str:
        """Build GStreamer pipeline string for low-latency RTSP decode."""
        # Low-latency settings:
        # - buffer-mode=0: Disable deep jitter buffer
        # - drop-on-latency=true: Drop frames that arrive late
        # - appsink sync=false: No clock sync (immediate delivery)
        # - appsink drop=true max-buffers=1: Keep only newest frame
        pipeline = (
            f"rtspsrc location={self.rtsp_uri} "
            f"latency={self.rtsp_latency_ms} "
            f"protocols={self.rtsp_transport} "
            f"buffer-mode=0 "
            f"drop-on-latency=true "
            f"! rtph264depay "
            f"! h264parse "
            f'! capsfilter caps="video/x-h264,alignment=au,stream-format=byte-stream" '
            f"! nvv4l2decoder "
            f"! nvvidconv "
            f"! video/x-raw,format=BGRx "
            f"! videoconvert "
            f"! video/x-raw,format=BGR "
            f"! appsink name=sink emit-signals=true drop=true max-buffers=1 sync=false"
        )
        return pipeline

    def _on_new_sample(self, appsink: GstApp.AppSink) -> Gst.FlowReturn:
        """GStreamer callback when a new frame is available."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        receive_ns = time.time_ns()
        self.last_frame_time = time.monotonic()

        buffer = sample.get_buffer()
        caps = sample.get_caps()

        # Extract frame dimensions
        structure = caps.get_structure(0)
        width = structure.get_int("width")[1]
        height = structure.get_int("height")[1]

        # Update stored dimensions
        if width > 0 and height > 0:
            self.frame_width = width
            self.frame_height = height

        # Extract RTP timestamp from buffer metadata if available
        # Note: This requires custom metadata or we approximate from PTS
        rtp_timestamp_raw = 0
        gst_pts_ns = buffer.pts if buffer.pts != Gst.CLOCK_TIME_NONE else 0

        # Approximate RTP timestamp from PTS (90kHz clock)
        if gst_pts_ns > 0:
            rtp_timestamp_raw = int((gst_pts_ns / 1_000_000_000) * 90000) & 0xFFFFFFFF

        rtp_unwrapped = self.rtp_unwrapper.unwrap(rtp_timestamp_raw)

        # Increment thor-side frame sequence
        self.thor_frame_seq += 1
        self.frames_decoded += 1

        # Map buffer to numpy array
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            self.get_logger().warning("Failed to map buffer")
            return Gst.FlowReturn.OK

        try:
            # Create numpy array from buffer data
            frame_data = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()  # Copy to own the data

            frame = FrameData(
                bgr_data=frame_data,
                width=width,
                height=height,
                appsink_receive_ns=receive_ns,
                rtp_timestamp_raw=rtp_timestamp_raw,
                rtp_timestamp_unwrapped=rtp_unwrapped,
                gst_pts_ns=gst_pts_ns,
                frame_seq=self.thor_frame_seq,
            )

            # Put frame in queue (non-blocking, drop old if full)
            try:
                # First try to remove old frame
                try:
                    self.frame_queue.get_nowait()
                    self.dropped_frames_total += 1
                    self.appsink_drops += 1
                    self._metrics_window_drops += 1
                except queue.Empty:
                    pass
                # Then add new frame
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.dropped_frames_total += 1
                self.appsink_drops += 1
                self._metrics_window_drops += 1

            # Track frames in current metrics window
            self._metrics_window_decoded += 1

        finally:
            buffer.unmap(map_info)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message) -> bool:
        """Handle GStreamer bus messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.get_logger().error(f"GStreamer error: {err.message} ({debug})")
            self.pipeline_state = CameraFrameTiming.STATE_RECONNECTING
            # Trigger reconnect from GLib thread
            GLib.idle_add(self._trigger_reconnect)

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            self.get_logger().warning(f"GStreamer warning: {warn.message}")

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                if new == Gst.State.PLAYING:
                    self.pipeline_state = CameraFrameTiming.STATE_STREAMING
                    self.get_logger().info("Pipeline state: STREAMING")

        elif msg_type == Gst.MessageType.EOS:
            self.get_logger().warning("End of stream received")
            self.pipeline_state = CameraFrameTiming.STATE_RECONNECTING
            GLib.idle_add(self._trigger_reconnect)

        return True

    def _trigger_reconnect(self) -> bool:
        """Trigger pipeline reconnection (called from GLib thread)."""
        self.get_logger().info("Triggering pipeline reconnect...")
        self._stop_pipeline()

        # Reconnect with backoff
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            backoff_idx = min(attempt, len(RECONNECT_BACKOFF) - 1)
            delay = RECONNECT_BACKOFF[backoff_idx]

            self.get_logger().info(f"Reconnect attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS} in {delay:.1f}s")
            time.sleep(delay)

            if self._start_pipeline():
                self.reconnect_count_total += 1
                self.get_logger().info(f"Reconnected (total: {self.reconnect_count_total})")
                return False  # Don't call again

        self.get_logger().error(f"Failed to reconnect after {MAX_RECONNECT_ATTEMPTS} attempts")
        self.pipeline_state = CameraFrameTiming.STATE_UNKNOWN
        return False

    def _check_stall(self) -> bool:
        """Check for stream stall (called periodically from GLib thread)."""
        if self.pipeline_state != CameraFrameTiming.STATE_STREAMING:
            return True  # Keep timer running

        elapsed = time.monotonic() - self.last_frame_time
        if elapsed > self.stall_timeout_sec:
            self.get_logger().warning(f"Stream stall detected ({elapsed:.1f}s without frames)")
            self.pipeline_state = CameraFrameTiming.STATE_RECONNECTING
            GLib.idle_add(self._trigger_reconnect)

        return True  # Keep timer running

    def _start_pipeline(self) -> bool:
        """Start GStreamer pipeline."""
        try:
            pipeline_str = self._build_gst_pipeline()
            self.get_logger().info(f"Starting pipeline: {pipeline_str[:100]}...")

            self.pipeline = Gst.parse_launch(pipeline_str)

            # Get appsink and connect signal
            appsink = self.pipeline.get_by_name("sink")
            appsink.connect("new-sample", self._on_new_sample)

            # Setup bus for messages
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            # Start pipeline
            self.pipeline_state = CameraFrameTiming.STATE_CONNECTING
            ret = self.pipeline.set_state(Gst.State.PLAYING)

            if ret == Gst.StateChangeReturn.FAILURE:
                self.get_logger().error("Failed to start pipeline")
                return False

            self.last_frame_time = time.monotonic()
            return True

        except Exception as e:
            self.get_logger().error(f"Pipeline start error: {e}")
            return False

    # GStreamer set_state(NULL) can hang on hardware disconnect (CP-008)
    _GSTREAMER_STOP_TIMEOUT_SEC = 5.0

    def _stop_pipeline(self):
        """Stop GStreamer pipeline (budget: _GSTREAMER_STOP_TIMEOUT_SEC)."""
        if self.pipeline:
            # Force-kill pipeline if set_state(NULL) hangs (e.g., camera disconnect)
            pipeline_ref = self.pipeline
            timer = threading.Timer(
                self._GSTREAMER_STOP_TIMEOUT_SEC,
                lambda: pipeline_ref.send_event(
                    Gst.Event.new_eos()
                ) if pipeline_ref else None,
            )
            timer.start()
            try:
                self.pipeline.set_state(Gst.State.NULL)
            finally:
                timer.cancel()
                self.pipeline = None

    def _glib_thread_func(self):
        """GLib main loop thread."""
        self.glib_loop = GLib.MainLoop()

        # Add stall watchdog timer (1 Hz)
        GLib.timeout_add(1000, self._check_stall)

        self.glib_loop.run()

    def _publish_frame(self):
        """Publish the newest available frame (called by ROS timer)."""
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            return  # No frame available

        now = self.get_clock().now()
        publish_ns = time.time_ns()

        # Record timing for FPS calculation
        self.frame_times.append(time.monotonic())
        if len(self.frame_times) > 100:
            self.frame_times = self.frame_times[-100:]

        # Build Image message
        img_msg = Image()
        img_msg.header.stamp = now.to_msg()
        img_msg.header.frame_id = self.camera_frame_id
        img_msg.height = frame.height
        img_msg.width = frame.width
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = False
        img_msg.step = frame.width * 3
        img_msg.data = frame.bgr_data.tobytes()

        self.image_pub.publish(img_msg)
        self.frames_published += 1

        # Build CameraInfo message
        info_msg = self._build_camera_info(now, frame.width, frame.height)
        self.camera_info_pub.publish(info_msg)

        # Build timing message
        if self.timing_pub:
            timing_msg = self._build_timing_msg(now, frame, publish_ns)
            self.timing_pub.publish(timing_msg)

            # Record latency for metrics
            self.latencies_ms.append(timing_msg.total_latency_ms)
            if len(self.latencies_ms) > 100:
                self.latencies_ms = self.latencies_ms[-100:]

    def _fetch_camera_info(self, url: str) -> Optional[dict[str, Any]]:
        """Fetch camera intrinsics from X1 HTTP endpoint.

        Args:
            url: HTTP URL to the camera-info endpoint

        Returns:
            Dict with camera intrinsics, or None if fetch failed.
        """
        try:
            self.get_logger().info(f"Fetching camera info from: {url}")
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5.0) as response:
                data = json.loads(response.read().decode())

            # Validate required fields
            if "k" not in data or len(data["k"]) < 9:
                self.get_logger().warn(f"Invalid camera_info response: missing 'k' matrix")
                return None

            self.get_logger().info(
                f"Camera intrinsics loaded: fx={data['k'][0]:.1f} fy={data['k'][4]:.1f} "
                f"cx={data['k'][2]:.1f} cy={data['k'][5]:.1f} "
                f"hfov={data.get('horizontal_fov_deg', '?')}°"
            )
            return data

        except urllib.error.URLError as e:
            self.get_logger().warn(f"Failed to fetch camera_info from {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Invalid JSON from {url}: {e}")
            return None
        except Exception as e:
            self.get_logger().warn(f"Error fetching camera_info: {e}")
            return None

    def _build_camera_info(self, stamp: rclpy.time.Time, width: int, height: int) -> CameraInfo:
        """Build CameraInfo message with intrinsics from X1 or fallback to HFOV estimate."""
        msg = CameraInfo()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.camera_frame_id
        msg.height = height
        msg.width = width

        # Use remote camera_info if available
        if self._remote_camera_info is not None:
            remote = self._remote_camera_info
            # Use intrinsics from X1, but validate resolution matches
            remote_w = remote.get("width", width)
            remote_h = remote.get("height", height)

            # Scale intrinsics if resolution differs
            scale_x = width / remote_w if remote_w > 0 else 1.0
            scale_y = height / remote_h if remote_h > 0 else 1.0

            k = remote.get("k", [])
            if len(k) >= 9:
                fx = k[0] * scale_x
                fy = k[4] * scale_y
                cx = k[2] * scale_x
                cy = k[5] * scale_y
            else:
                # Fallback
                fx = fy = 554.0
                cx, cy = width / 2.0, height / 2.0

            msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

            # Rectification matrix
            r = remote.get("r", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            msg.r = list(r) if len(r) >= 9 else [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

            # Projection matrix
            msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

            # Distortion
            msg.distortion_model = remote.get("distortion_model", "plumb_bob")
            d = remote.get("d", [0.0, 0.0, 0.0, 0.0, 0.0])
            msg.d = list(d) if len(d) >= 5 else [0.0, 0.0, 0.0, 0.0, 0.0]

        else:
            # Fallback: compute focal length from HFOV parameter
            hfov_rad = self.camera_hfov_deg * math.pi / 180.0
            fx = (width / 2) / math.tan(hfov_rad / 2)
            fy = fx  # Assume square pixels
            cx = width / 2.0
            cy = height / 2.0

            msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            msg.distortion_model = "plumb_bob"
            msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        return msg

    def _build_timing_msg(self, stamp: rclpy.time.Time, frame: FrameData, publish_ns: int) -> CameraFrameTiming:
        """Build CameraFrameTiming message with latency breakdown."""
        msg = CameraFrameTiming()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.camera_frame_id

        # Frame identity
        msg.frame_seq = frame.frame_seq
        msg.thor_frame_seq = frame.frame_seq

        # RTP timestamps
        msg.rtp_timestamp = frame.rtp_timestamp_raw
        msg.rtp_timestamp_unwrapped = frame.rtp_timestamp_unwrapped
        msg.sender_pts_ns = frame.gst_pts_ns

        # Thor timestamps
        msg.appsink_receive_ns = frame.appsink_receive_ns
        msg.publish_ns = publish_ns

        # Latency breakdown
        queue_wait_ns = publish_ns - frame.appsink_receive_ns
        msg.queue_wait_ms = queue_wait_ns / 1_000_000.0

        # Decode latency (PTS to appsink receive)
        if frame.gst_pts_ns > 0:
            # This is approximate since PTS is sender-relative
            msg.decode_latency_ms = 0.0  # Can't measure without clock sync
        else:
            msg.decode_latency_ms = 0.0

        # Network latency estimate (simplified - would need RTP interarrival analysis)
        msg.network_latency_estimate_ms = 0.0

        # Total latency
        msg.total_latency_ms = msg.queue_wait_ms + msg.decode_latency_ms + msg.network_latency_estimate_ms

        # Pipeline state
        msg.pipeline_state = self.pipeline_state

        # Trust metadata
        msg.timestamp_source = "thor_receive"
        msg.intrinsics_source = "hfov_estimate"
        msg.intrinsics_warning = "SYNTHETIC_CALIBRATION"

        return msg

    def _calculate_fps(self) -> float:
        """Calculate current FPS from frame times."""
        if len(self.frame_times) < 2:
            return 0.0
        duration = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / duration if duration > 0 else 0.0

    def _publish_metrics(self):
        """Publish perception metrics."""
        if not self.metrics_pub:
            return

        msg = PerceptionMetrics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_ingest"

        msg.fps = self._calculate_fps()
        msg.latency_p95_ms = float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0
        msg.dropped_frames_total = self.dropped_frames_total
        msg.reconnect_count_total = self.reconnect_count_total
        msg.queue_depth = self.frame_queue.qsize()

        msg.model_name = "camera_ingest"
        msg.model_version = "1.0"
        msg.tracker_enabled = False

        # Calculate drop rate for current metrics window
        window_duration = time.monotonic() - self._metrics_window_start
        if window_duration > 0 and self._metrics_window_decoded > 0:
            drop_rate = self._metrics_window_drops / self._metrics_window_decoded
        else:
            drop_rate = 0.0

        # RTSP instrumentation (exposed via extra_metrics if available)
        self.get_logger().debug(
            f"RTSP_METRICS: decoded={self.frames_decoded} published={self.frames_published} "
            f"appsink_drops={self.appsink_drops} window_drop_rate={drop_rate:.2%}"
        )

        # Reset metrics window for next interval
        self._metrics_window_start = time.monotonic()
        self._metrics_window_drops = 0
        self._metrics_window_decoded = 0

        # Health status
        if self.pipeline_state == CameraFrameTiming.STATE_STREAMING:
            if msg.fps < self.min_fps_threshold:
                msg.health = PerceptionMetrics.DEGRADED
                msg.reason_code = "LOW_FPS"
            elif drop_rate > 0.2:  # >20% drop rate in current window
                msg.health = PerceptionMetrics.DEGRADED
                msg.reason_code = "HIGH_DROP_RATE"
            else:
                msg.health = PerceptionMetrics.OK
                msg.reason_code = ""
        elif self.pipeline_state == CameraFrameTiming.STATE_RECONNECTING:
            msg.health = PerceptionMetrics.DEGRADED
            msg.reason_code = "RTSP_RECONNECTING"
        else:
            msg.health = PerceptionMetrics.ERROR
            msg.reason_code = "RTSP_DISCONNECTED"

        self.metrics_pub.publish(msg)

    def get_rtsp_stats(self) -> dict:
        """Get RTSP instrumentation stats for diagnostics.

        Returns dict with:
        - frames_decoded: Total frames received from decoder
        - frames_published: Total frames published to ROS
        - appsink_drops: Frames dropped due to queue overflow
        - queue_size: Current frame queue size
        """
        return {
            "frames_decoded": self.frames_decoded,
            "frames_published": self.frames_published,
            "appsink_drops": self.appsink_drops,
            "queue_size": self.frame_queue.qsize(),
            "dropped_frames_total": self.dropped_frames_total,
            "reconnect_count_total": self.reconnect_count_total,
        }

    def start(self) -> bool:
        """Start the camera ingest pipeline."""
        self.get_logger().info("Starting camera ingest pipeline...")

        # Start GLib thread
        self.running = True
        self.glib_thread = threading.Thread(target=self._glib_thread_func, daemon=True)
        self.glib_thread.start()

        # Start GStreamer pipeline
        if not self._start_pipeline():
            self.get_logger().error("Failed to start GStreamer pipeline")
            return False

        self.get_logger().info("Camera ingest pipeline started")
        return True

    def stop(self):
        """Stop the camera ingest pipeline."""
        self.running = False

        # Stop pipeline
        self._stop_pipeline()

        # Stop GLib loop
        if self.glib_loop:
            self.glib_loop.quit()

        if self.glib_thread:
            self.glib_thread.join(timeout=2.0)

        self.get_logger().info("Camera ingest pipeline stopped")


def main(args=None):
    rclpy.init(args=args)
    node = CameraIngestNode()

    if not node.start():
        node.get_logger().error("Failed to start camera ingest pipeline")
        rclpy.shutdown()
        return 1

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
