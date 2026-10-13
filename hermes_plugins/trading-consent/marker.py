"""Profile-bound plugin-registration marker: written by __init__.py:register() when Hermes loads
this plugin for a profile, checked by guarded_mcp_launcher.py before it will exec the real MCP
server command.

Why a file at all: the guarded launcher runs as a completely separate OS process (whatever
`mcp_servers.<name>.command` names), spawned by Hermes fresh each time it needs that MCP server.
It has no access to the parent Hermes process's in-memory plugin registry (PluginContext /
ctx.has_plugin()) and Hermes's own subprocess-env filtering (tools/mcp_tool_config.py:
_build_safe_env) does not pass HERMES_HOME through by default -- so the launcher cannot assume it
knows its own profile home from its environment alone. A small, profile-scoped, freshness-checked
marker file is the practical signal available across that process boundary.

This is a tamper-resistance *speed bump*, not a cryptographic guarantee: anyone with filesystem
write access to the profile directory could fabricate a marker. Its purpose is to fail hard,
loudly, and by default when the plugin is simply not registered (disabled, removed, never
installed, or a config mistake) -- the exact scenario that would otherwise leave the gate silently
absent with no error at all.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MARKER_RELATIVE_DIR = ".trading_consent"
MARKER_FILENAME_TEMPLATE = "registered.{pid}.json"
DEFAULT_MAX_MARKER_AGE_SECONDS = 24 * 60 * 60  # generous: one Hermes session per day still passes


def _valid_pid(pid: object) -> bool:
    return isinstance(pid, int) and not isinstance(pid, bool) and pid > 0


def marker_path(profile_home: Path, pid: int) -> Path:
    """Marker for the Hermes process *pid*. One file per process: every Hermes process that loads
    this plugin (the long-lived gateway, a one-shot `hermes ... plugins list`, a status command)
    writes only its own file, so no process can overwrite or invalidate another's marker. A
    single shared file was clobbered by exactly that -- a read-only CLI command replaced the
    gateway's marker, so the gateway's next MCP (re)launch hard-stopped.
    """
    if not _valid_pid(pid):
        raise ValueError(f"marker pid must be a positive int, got {pid!r}")
    return Path(profile_home) / MARKER_RELATIVE_DIR / MARKER_FILENAME_TEMPLATE.format(pid=pid)


def write_marker(profile_home: Path, *, profile: str, plugin_id: str, pid: Optional[int] = None) -> Path:
    """Write/refresh the marker for this process (*pid* defaults to ``os.getpid()``). Called from
    register(ctx) on every plugin load, so the marker's age tracks "how long ago did this Hermes
    process load the plugin" -- not "ever, once."

    Never deletes or touches any other process's marker. Stale markers left by exited processes
    are inert -- the launcher only ever reads its own parent's file -- and are deliberately not
    garbage-collected: deciding "that pid is dead, delete its file" races with PID reuse (a new
    Hermes process can be assigned that pid and write the same filename between the liveness
    check and the unlink), which would delete a valid marker and recreate the outage this
    per-process layout exists to prevent.
    """
    pid = os.getpid() if pid is None else pid
    path = marker_path(profile_home, pid)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    payload = {
        "plugin_id": plugin_id,
        "profile": profile,
        "registered_at": time.time(),
        "pid": pid,
    }
    # Unique temp file per write (mkstemp creates it 0600) -- a fixed temp name would itself be a
    # shared file that concurrent writers could clobber before the rename.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
        os.replace(tmp_name, path)  # atomic on POSIX -- no reader ever sees a partial write
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


@dataclass(frozen=True)
class MarkerCheck:
    ok: bool
    reason: str


def check_marker(
    profile_home: Path,
    *,
    parent_pid: int,
    expected_profile: str,
    expected_plugin_id: Optional[str] = None,
    max_age_seconds: int = DEFAULT_MAX_MARKER_AGE_SECONDS,
) -> MarkerCheck:
    """Validate the marker written by Hermes process *parent_pid* for *expected_profile*. Every
    failure mode returns ``ok=False`` with a specific reason -- there is no path that returns
    ok=True on missing/malformed/stale/mismatched input.

    The guarded launcher passes its own ``os.getppid()`` -- the Hermes process that just spawned
    it -- and only that process's file (``registered.<parent_pid>.json``) is ever read. Markers
    from any other process, live or exited, are never consulted, and the legacy shared
    ``registered.json`` is ignored. The recorded ``pid`` inside the file must also equal
    *parent_pid*, so a marker copied or renamed onto another pid's filename does not pass.
    Freshness alone is never sufficient: a recent marker from some other Hermes process (e.g. a
    prior run, or one where the plugin was disabled across a restart) must not open the gate.
    """
    if not _valid_pid(parent_pid):
        return MarkerCheck(False, f"invalid parent pid {parent_pid!r}")
    path = marker_path(profile_home, parent_pid)
    if not path.is_file():
        return MarkerCheck(False, (
            f"no plugin-registration marker for parent process {parent_pid} at {path}"
        ))

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return MarkerCheck(False, f"marker at {path} is unreadable/invalid JSON: {exc}")

    if not isinstance(data, dict):
        return MarkerCheck(False, f"marker at {path} is not a JSON object")

    marker_pid = data.get("pid")
    if not _valid_pid(marker_pid) or marker_pid != parent_pid:
        return MarkerCheck(False, (
            f"marker at {path} records pid {marker_pid!r}, not the launcher's parent process "
            f"{parent_pid!r}"
        ))

    marker_profile = data.get("profile")
    if marker_profile != expected_profile:
        return MarkerCheck(False, (
            f"marker profile {marker_profile!r} does not match expected profile "
            f"{expected_profile!r}"
        ))

    if expected_plugin_id is not None and data.get("plugin_id") != expected_plugin_id:
        return MarkerCheck(False, (
            f"marker plugin_id {data.get('plugin_id')!r} does not match expected "
            f"{expected_plugin_id!r}"
        ))

    registered_at = data.get("registered_at")
    if not isinstance(registered_at, (int, float)):
        return MarkerCheck(False, "marker is missing a numeric registered_at timestamp")
    age = time.time() - registered_at
    if age < 0:
        return MarkerCheck(False, "marker registered_at is in the future -- clock skew or tampering")
    if age > max_age_seconds:
        return MarkerCheck(False, f"marker is stale ({age:.0f}s old, max {max_age_seconds}s)")

    return MarkerCheck(True, "marker valid")
