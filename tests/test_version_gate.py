# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Innate Inc

"""Tests for the client version gate: version comparison and throttling of
the spoken version warning so reconnect storms don't spam the robot's TTS."""

from src.auth import token_auth
from src.auth.token_auth import compare_versions, should_speak_version_warning


class TestCompareVersions:
    def test_old_version_rejected(self):
        is_valid, msg = compare_versions("0.2.4", min_version="0.3.0")
        assert not is_valid
        assert "0 point 2 point 4" in msg

    def test_min_version_accepted(self):
        is_valid, _ = compare_versions("0.3.0", min_version="0.3.0")
        assert is_valid

    def test_newer_version_accepted(self):
        is_valid, _ = compare_versions("0.6.0", min_version="0.3.0")
        assert is_valid

    def test_dev_version_always_allowed(self):
        is_valid, msg = compare_versions("0.2.0-dev", min_version="0.3.0")
        assert is_valid
        assert "dev version" in msg

    def test_invalid_version_rejected(self):
        is_valid, msg = compare_versions("sim-assets-258343f422b5")
        assert not is_valid
        assert "Invalid version format" in msg


class TestVersionWarningThrottle:
    def setup_method(self):
        token_auth._last_version_warning.clear()

    def test_first_warning_allowed(self):
        assert should_speak_version_warning("robot-a")

    def test_repeat_within_window_blocked(self):
        assert should_speak_version_warning("robot-a")
        assert not should_speak_version_warning("robot-a")
        assert not should_speak_version_warning("robot-a")

    def test_different_tokens_independent(self):
        assert should_speak_version_warning("robot-a")
        assert should_speak_version_warning("robot-b")

    def test_allowed_again_after_window(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(token_auth.time, "monotonic", lambda: clock["now"])

        assert should_speak_version_warning("robot-a")
        clock["now"] += token_auth.VERSION_WARNING_INTERVAL_S - 1
        assert not should_speak_version_warning("robot-a")
        clock["now"] += 2
        assert should_speak_version_warning("robot-a")

    def test_stale_entries_pruned(self, monkeypatch):
        clock = {"now": 1000.0}
        monkeypatch.setattr(token_auth.time, "monotonic", lambda: clock["now"])

        for i in range(1001):
            assert should_speak_version_warning(f"robot-{i}")
        clock["now"] += token_auth.VERSION_WARNING_INTERVAL_S + 1
        assert should_speak_version_warning("robot-new")
        # All stale entries were pruned once the dict exceeded the cap.
        assert len(token_auth._last_version_warning) == 1
