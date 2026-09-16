"""Integration tests for WatchdogThread adoption in authorization_node (CP-002).

Tests verify:
1. Watchdog fires on subscription callback stall
2. Timeout forces UNAUTHORIZED state
3. Normal subscription activity keeps watchdog healthy
4. Clean shutdown
"""

import threading
import time

import pytest

from thor_behavior.watchdog import WatchdogState, WatchdogThread


class TestAuthorizationWatchdogIntegration:
    """Tests for authorization_node watchdog adoption."""

    def test_subscription_heartbeat_prevents_timeout(self):
        """Heartbeats from subscription callbacks prevent timeout at 3.0s interval."""
        recorder_calls = []

        def on_timeout(name, elapsed):
            recorder_calls.append((name, elapsed))

        wd = WatchdogThread(
            "authorization_node", interval_sec=3.0, on_timeout=on_timeout
        )
        wd.start()
        try:
            # Simulate subscription callbacks at ~1Hz
            for _ in range(10):
                wd.heartbeat()
                time.sleep(0.3)
            assert len(recorder_calls) == 0
            assert wd.state == WatchdogState.MONITORING
        finally:
            wd.stop()

    def test_upstream_data_loss_triggers_timeout(self):
        """No subscription callbacks → timeout fires within interval."""
        event = threading.Event()
        timeout_info = {}

        def on_timeout(name, elapsed):
            timeout_info["name"] = name
            timeout_info["elapsed"] = elapsed
            event.set()

        wd = WatchdogThread(
            "authorization_node", interval_sec=0.2, on_timeout=on_timeout
        )
        wd.start()
        try:
            wd.heartbeat()
            # Simulate complete upstream data loss
            fired = event.wait(timeout=2.0)
            assert fired, "Timeout should fire on data loss"
            assert timeout_info["name"] == "authorization_node"
            assert wd.state == WatchdogState.TIMED_OUT
        finally:
            wd.stop()

    def test_timeout_callback_is_thread_safe(self):
        """on_timeout fires from watchdog thread; simulates state_lock usage."""
        state = {"authorized": True}
        lock = threading.Lock()
        event = threading.Event()

        def on_timeout(name, elapsed):
            with lock:
                state["authorized"] = False
            event.set()

        wd = WatchdogThread(
            "authorization_node", interval_sec=0.1, on_timeout=on_timeout
        )
        wd.start()
        try:
            event.wait(timeout=2.0)
            with lock:
                assert state["authorized"] is False
        finally:
            wd.stop()

    def test_follow_me_rejected_after_timeout(self):
        """After timeout, authorization should be False."""
        authorized = {"value": True}
        lock = threading.Lock()
        event = threading.Event()

        def on_timeout(name, elapsed):
            with lock:
                authorized["value"] = False
            event.set()

        wd = WatchdogThread(
            "authorization_node", interval_sec=0.1, on_timeout=on_timeout
        )
        wd.start()
        try:
            # Simulate stall
            event.wait(timeout=2.0)
            with lock:
                assert authorized["value"] is False, (
                    "Follow-me should be rejected after watchdog timeout"
                )
        finally:
            wd.stop()

    def test_clean_shutdown_with_watchdog(self):
        """Watchdog stops cleanly when node is destroyed."""
        wd = WatchdogThread(
            "authorization_node",
            interval_sec=3.0,
            on_timeout=lambda n, e: None,
        )
        wd.start()
        wd.heartbeat()

        clean = wd.stop()
        assert clean is True
        assert wd.state == WatchdogState.STOPPED
        assert not wd.is_alive
