"""Marker read/write round trip, and the guarded launcher's absent/wrong-marker hard stop vs.
valid-marker exec path. Structural check that the launcher never uses shell=True/os.system."""

from __future__ import annotations

import json
import os
import time

import pytest


# --- marker.py --------------------------------------------------------------------------------

def test_write_then_check_round_trip(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader")
    assert result.ok, result.reason


def test_missing_marker_fails(marker_mod, fresh_profile_home):
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader")
    assert not result.ok


def test_wrong_profile_in_marker_fails(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="tradingresearch", plugin_id="trading-consent")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader")
    assert not result.ok


def test_wrong_plugin_id_fails(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="some-other-plugin")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader",
                                      expected_plugin_id="trading-consent")
    assert not result.ok


def test_malformed_json_fails(marker_mod, fresh_profile_home):
    path = marker_mod.marker_path(fresh_profile_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader")
    assert not result.ok


def test_stale_marker_fails(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["registered_at"] = time.time() - 999_999  # far in the past
    path.write_text(json.dumps(data), encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader", max_age_seconds=3600)
    assert not result.ok
    assert "stale" in result.reason


def test_future_registered_at_fails(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["registered_at"] = time.time() + 999_999
    path.write_text(json.dumps(data), encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader")
    assert not result.ok


def test_require_pid_alive_rejects_dead_pid(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["pid"] = 999_999_999  # astronomically unlikely to be a live pid
    path.write_text(json.dumps(data), encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader", require_pid_alive=True)
    assert not result.ok


def test_require_parent_pid_accepts_exact_match(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["pid"] = 4242
    path.write_text(json.dumps(data), encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, expected_profile="trader", require_parent_pid=4242)
    assert result.ok, result.reason


def test_require_parent_pid_rejects_a_live_but_different_pid(marker_mod, fresh_profile_home):
    """Item 2: freshness/liveness alone is not the gate. A marker pid that is a genuinely LIVE
    process (this very test process) but not the launcher's actual parent must still fail --
    proving "alive" and "is my parent" are checked as different things, not conflated."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["pid"] = os.getpid()  # alive (it's this test process) but not the "parent" we require
    path.write_text(json.dumps(data), encoding="utf-8")
    result = marker_mod.check_marker(
        fresh_profile_home, expected_profile="trader", require_parent_pid=os.getpid() + 1,
    )
    assert not result.ok
    assert "does not match" in result.reason


def test_require_parent_pid_rejects_stale_prior_process_marker(marker_mod, fresh_profile_home):
    """Simulates the exact scenario item 2 names: a marker recorded by a prior Hermes process
    that has since exited (or the plugin was disabled across a restart), while a NEW Hermes
    process -- with a different pid -- is now the one trying to launch the MCP server."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["pid"] = 111111  # the old, now-irrelevant Hermes process
    path.write_text(json.dumps(data), encoding="utf-8")
    new_hermes_pid = 222222  # the current launcher's real parent
    result = marker_mod.check_marker(
        fresh_profile_home, expected_profile="trader", require_parent_pid=new_hermes_pid,
    )
    assert not result.ok


def test_require_parent_pid_takes_precedence_over_require_pid_alive(marker_mod, fresh_profile_home):
    """If both are somehow passed, exact-parent-pid identity is the check that runs -- a merely
    alive-but-wrong pid must not be waved through by the weaker liveness flag."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    data = json.loads(path.read_text())
    data["pid"] = os.getpid()  # alive
    path.write_text(json.dumps(data), encoding="utf-8")
    result = marker_mod.check_marker(
        fresh_profile_home, expected_profile="trader",
        require_pid_alive=True, require_parent_pid=os.getpid() + 1,
    )
    assert not result.ok


def test_marker_file_permissions_are_restrictive(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    path = marker_mod.marker_path(fresh_profile_home)
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600


# --- guarded_mcp_launcher.py --------------------------------------------------------------------

def test_launcher_hard_stops_without_env_var(launcher_mod, monkeypatch):
    monkeypatch.delenv(launcher_mod.ENV_PROFILE, raising=False)
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main(["node", "server.js"])
    assert exc.value.code != 0


def test_launcher_hard_stops_when_marker_absent(launcher_mod, monkeypatch, fresh_profile_home):
    monkeypatch.setenv(launcher_mod.ENV_PROFILE, "trader")
    monkeypatch.setenv(launcher_mod.ENV_PROFILE_HOME, str(fresh_profile_home))
    execs = []
    monkeypatch.setattr(os, "execvp", lambda *a: execs.append(a))
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main(["node", "server.js"])
    assert exc.value.code != 0
    assert execs == []  # the real command must never be exec'd


def _write_marker_via_launcher(fresh_profile_home, *, profile, pid):
    """marker.py is imported fresh inside guarded_mcp_launcher.py's own module namespace (see
    conftest) -- write through THAT copy so the launcher's check_marker sees it, with an explicit
    pid (write_marker's own os.getpid() would record the TEST process's pid, not whatever value
    a given test needs to exercise against os.getppid())."""
    import json
    import sys
    marker_via_launcher = sys.modules["marker"]
    path = marker_via_launcher.write_marker(fresh_profile_home, profile=profile, plugin_id="trading-consent")
    data = json.loads(path.read_text())
    data["pid"] = pid
    path.write_text(json.dumps(data), encoding="utf-8")


def test_launcher_hard_stops_when_marker_profile_mismatched(
    launcher_mod, monkeypatch, fresh_profile_home,
):
    _write_marker_via_launcher(fresh_profile_home, profile="tradingresearch", pid=os.getppid())

    monkeypatch.setenv(launcher_mod.ENV_PROFILE, "trader")
    monkeypatch.setenv(launcher_mod.ENV_PROFILE_HOME, str(fresh_profile_home))
    execs = []
    monkeypatch.setattr(os, "execvp", lambda *a: execs.append(a))
    with pytest.raises(SystemExit):
        launcher_mod.main(["node", "server.js"])
    assert execs == []


def test_launcher_hard_stops_on_stale_prior_process_pid(launcher_mod, monkeypatch, fresh_profile_home):
    """Item 2: correct profile, correct plugin, fresh marker -- but recorded by a pid that is
    NOT this launcher invocation's actual parent (a prior Hermes process, or one that ran
    without the plugin re-registering across a restart). Must still hard-stop."""
    _write_marker_via_launcher(fresh_profile_home, profile="trader", pid=os.getppid() + 1)

    monkeypatch.setenv(launcher_mod.ENV_PROFILE, "trader")
    monkeypatch.setenv(launcher_mod.ENV_PROFILE_HOME, str(fresh_profile_home))
    execs = []
    monkeypatch.setattr(os, "execvp", lambda *a: execs.append(a))
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main(["node", "server.js"])
    assert exc.value.code != 0
    assert execs == []


def test_launcher_execs_real_command_when_marker_valid(launcher_mod, monkeypatch, fresh_profile_home):
    # The launcher checks require_parent_pid=os.getppid() -- so the marker must record exactly
    # THIS test process's actual parent pid to count as valid.
    _write_marker_via_launcher(fresh_profile_home, profile="trader", pid=os.getppid())

    monkeypatch.setenv(launcher_mod.ENV_PROFILE, "trader")
    monkeypatch.setenv(launcher_mod.ENV_PROFILE_HOME, str(fresh_profile_home))
    execs = []
    monkeypatch.setattr(os, "execvp", lambda resolved, args: execs.append((resolved, args)))
    monkeypatch.setattr(launcher_mod, "_which", lambda cmd: f"/usr/bin/{cmd}")

    # main() only calls sys.exit() via _fail(); a successful exec path calls os.execvp
    # (mocked here to a no-op recorder) and falls off the end of the function. In real
    # execution a successful execvp() never returns at all (it replaces the process image) --
    # the mock is what makes this path observable in a test.
    launcher_mod.main(["node", "server.js", "--flag"])

    assert len(execs) == 1
    resolved, args = execs[0]
    assert resolved == "/usr/bin/node"
    assert args == ["node", "server.js", "--flag"]


def test_launcher_requires_at_least_one_argument(launcher_mod, monkeypatch, fresh_profile_home):
    monkeypatch.setenv(launcher_mod.ENV_PROFILE, "trader")
    monkeypatch.setenv(launcher_mod.ENV_PROFILE_HOME, str(fresh_profile_home))
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main([])
    assert exc.value.code != 0


def test_launcher_source_never_uses_shell_true_or_os_system(plugin_dir):
    source = (plugin_dir / "guarded_mcp_launcher.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "os.system(" not in source
    assert "os.popen(" not in source
