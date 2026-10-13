"""Marker read/write round trip, and the guarded launcher's absent/wrong-marker hard stop vs.
valid-marker exec path. Structural check that the launcher never uses shell=True/os.system."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest


# --- marker.py --------------------------------------------------------------------------------

def _rewrite(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_write_then_check_round_trip(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert result.ok, result.reason


def test_marker_file_is_named_for_the_writing_pid(marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    assert path.name == f"registered.{os.getpid()}.json"
    assert json.loads(path.read_text())["pid"] == os.getpid()


def test_missing_marker_fails(marker_mod, fresh_profile_home):
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert not result.ok


def test_wrong_profile_in_marker_fails(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="tradingresearch", plugin_id="trading-consent")
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert not result.ok


def test_wrong_plugin_id_fails(marker_mod, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="some-other-plugin")
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader",
                                      expected_plugin_id="trading-consent")
    assert not result.ok


def test_malformed_json_fails(marker_mod, fresh_profile_home):
    path = marker_mod.marker_path(fresh_profile_home, os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert not result.ok


def test_stale_marker_fails(marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    _rewrite(path, registered_at=time.time() - 999_999)  # far in the past
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader",
                                      max_age_seconds=3600)
    assert not result.ok
    assert "stale" in result.reason


def test_future_registered_at_fails(marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    _rewrite(path, registered_at=time.time() + 999_999)
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert not result.ok


def test_check_reads_only_the_requested_pids_marker(marker_mod, fresh_profile_home):
    """A valid, fresh marker from a DIFFERENT live process (this test process) never satisfies a
    check for another parent pid -- that pid's own file is the only one consulted."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid() + 1, expected_profile="trader")
    assert not result.ok
    assert f"registered.{os.getpid() + 1}.json" in result.reason


def test_prior_process_marker_does_not_satisfy_a_new_process(marker_mod, fresh_profile_home):
    """A marker left by a prior Hermes process (exited, or the plugin was disabled across a
    restart) never opens the gate for the NEW Hermes process now launching the MCP server."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=111111)
    assert marker_mod.check_marker(fresh_profile_home, parent_pid=111111, expected_profile="trader").ok
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=222222, expected_profile="trader")
    assert not result.ok


def test_pid_inside_file_must_match_filename_pid(marker_mod, fresh_profile_home):
    """A marker copied/renamed onto another pid's filename does not pass: the recorded pid is
    checked too, not just the filename."""
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=111111)
    path.rename(marker_mod.marker_path(fresh_profile_home, 222222))
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=222222, expected_profile="trader")
    assert not result.ok
    assert "111111" in result.reason


@pytest.mark.parametrize("bad_pid", [0, -1, None, "123", True, 1.5])
def test_invalid_parent_pid_fails_closed(marker_mod, fresh_profile_home, bad_pid):
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=bad_pid, expected_profile="trader")
    assert not result.ok


@pytest.mark.parametrize("bad_pid", [0, -1, "123", True])
def test_write_marker_rejects_invalid_pid(marker_mod, fresh_profile_home, bad_pid):
    with pytest.raises(ValueError):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=bad_pid)


def test_legacy_shared_marker_is_ignored(marker_mod, fresh_profile_home):
    """The pre-fix single registered.json -- even one recording exactly the right pid, profile,
    and a fresh timestamp -- is never read."""
    legacy = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR / "registered.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "plugin_id": "trading-consent", "profile": "trader",
        "registered_at": time.time(), "pid": os.getpid(),
    }), encoding="utf-8")
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert not result.ok


def test_one_process_writing_cannot_invalidate_another(marker_mod, fresh_profile_home):
    """The live defect: the gateway registers, then a one-shot CLI process loads the plugin and
    registers too. The gateway's marker must be byte-for-byte untouched and still valid."""
    gateway = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=4242)
    before = gateway.read_bytes()
    for cli_pid in (56147, 56148, 56149):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=cli_pid)
    assert gateway.read_bytes() == before
    assert marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader").ok
    for cli_pid in (56147, 56148, 56149):
        assert marker_mod.check_marker(fresh_profile_home, parent_pid=cli_pid, expected_profile="trader").ok


def test_rewrite_by_same_pid_only_refreshes_its_own_file(marker_mod, fresh_profile_home):
    other = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=4242)
    other_before = other.read_bytes()
    first = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    second = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    assert first == second
    assert other.read_bytes() == other_before


def test_write_leaves_no_temp_files_and_never_deletes_other_markers(marker_mod, fresh_profile_home):
    """Stale markers from exited processes are inert and deliberately not garbage-collected
    (deleting by "pid is dead" races with PID reuse) -- a write must never remove them."""
    stale = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=999_999_999)
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    names = sorted(p.name for p in stale.parent.iterdir())
    assert names == sorted([stale.name, f"registered.{os.getpid()}.json"])


def test_marker_file_permissions_are_restrictive(marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


_WRITER_SCRIPT = """
import sys, time
sys.path.insert(0, sys.argv[1])
import marker
go = float(sys.argv[3])
while time.time() < go:
    pass
for _ in range(25):
    marker.write_marker(sys.argv[2], profile="trader", plugin_id="trading-consent")
print(marker.marker_path(sys.argv[2], __import__("os").getpid()).name)
"""


def test_concurrent_process_registrations_coexist(plugin_dir, marker_mod, fresh_profile_home):
    """Real OS processes, released at the same instant, each re-registering repeatedly: every
    process's own marker survives intact and validates for that pid, and no temp file leaks."""
    go = time.time() + 1.0
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER_SCRIPT, str(plugin_dir), str(fresh_profile_home), str(go)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(6)
    ]
    results = []
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        results.append((proc.pid, out.strip()))

    marker_dir = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR
    assert sorted(p.name for p in marker_dir.iterdir()) == sorted(f"registered.{pid}.json" for pid, _ in results)
    for pid, name in results:
        assert name == f"registered.{pid}.json"
        check = marker_mod.check_marker(fresh_profile_home, parent_pid=pid, expected_profile="trader",
                                        expected_plugin_id="trading-consent")
        assert check.ok, check.reason


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
    pid (write_marker's default os.getpid() would record the TEST process's pid, not whatever
    value a given test needs to exercise against os.getppid())."""
    marker_via_launcher = sys.modules["marker"]
    return marker_via_launcher.write_marker(
        fresh_profile_home, profile=profile, plugin_id="trading-consent", pid=pid,
    )


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
    """Correct profile, correct plugin, fresh marker -- but written by a pid that is NOT this
    launcher invocation's actual parent (a prior Hermes process, a concurrent CLI command, or one
    that ran without the plugin re-registering across a restart). Must still hard-stop."""
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
    # The launcher reads only registered.<os.getppid()>.json -- so the marker must be written for
    # exactly THIS test process's actual parent pid to count as valid.
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


# --- end-to-end: real launcher processes, real parent pids ---------------------------------------

def _run_launcher(plugin_dir, profile_home, *, profile="trader"):
    """Run guarded_mcp_launcher.py as a real child of THIS test process, so its os.getppid() is
    genuinely os.getpid() here. The 'real command' is a harmless python one-liner."""
    env = dict(os.environ)
    env["TRADING_CONSENT_PROFILE"] = profile
    env["TRADING_CONSENT_PROFILE_HOME"] = str(profile_home)
    return subprocess.run(
        [sys.executable, str(plugin_dir / "guarded_mcp_launcher.py"),
         sys.executable, "-c", "print('REAL-MCP-STARTED')"],
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_real_launcher_selects_only_its_parents_marker(plugin_dir, marker_mod, fresh_profile_home):
    """Gateway (this process) and several other registered processes all hold valid markers: the
    launcher starts the real command, because its own parent's marker is valid."""
    for other_pid in (56147, 56148):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=other_pid)
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    for other_pid in (56149,):  # a registration AFTER ours must not invalidate ours either
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=other_pid)

    proc = _run_launcher(plugin_dir, fresh_profile_home)
    assert proc.returncode == 0, proc.stderr
    assert "REAL-MCP-STARTED" in proc.stdout


def test_real_launcher_blocks_when_only_other_processes_are_registered(plugin_dir, marker_mod, fresh_profile_home):
    """Valid, fresh markers for other pids (including a live one) never stand in for the parent."""
    for other_pid in (56147, os.getppid()):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=other_pid)

    proc = _run_launcher(plugin_dir, fresh_profile_home)
    assert proc.returncode != 0
    assert "REAL-MCP-STARTED" not in proc.stdout
    assert f"registered.{os.getpid()}.json" in proc.stderr


def test_real_launcher_blocks_on_stale_parent_marker(plugin_dir, marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    data = json.loads(path.read_text())
    data["registered_at"] = time.time() - 999_999
    path.write_text(json.dumps(data), encoding="utf-8")

    proc = _run_launcher(plugin_dir, fresh_profile_home)
    assert proc.returncode != 0
    assert "REAL-MCP-STARTED" not in proc.stdout
    assert "stale" in proc.stderr


def test_real_launcher_ignores_legacy_shared_marker(plugin_dir, marker_mod, fresh_profile_home):
    legacy = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR / "registered.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "plugin_id": "trading-consent", "profile": "trader",
        "registered_at": time.time(), "pid": os.getpid(),
    }), encoding="utf-8")

    proc = _run_launcher(plugin_dir, fresh_profile_home)
    assert proc.returncode != 0
    assert "REAL-MCP-STARTED" not in proc.stdout
