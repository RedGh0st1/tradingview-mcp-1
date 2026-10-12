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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MARKER_RELATIVE_DIR = ".trading_consent"
MARKER_FILENAME = "registered.json"
DEFAULT_MAX_MARKER_AGE_SECONDS = 24 * 60 * 60  # generous: one Hermes session per day still passes


def marker_path(profile_home: Path) -> Path:
    return Path(profile_home) / MARKER_RELATIVE_DIR / MARKER_FILENAME


def write_marker(profile_home: Path, *, profile: str, plugin_id: str) -> Path:
    """Write/refresh the marker for *profile*. Called from register(ctx) on every plugin load
    (i.e., every Hermes startup for that profile), so the marker's age tracks "how long ago did
    a Hermes process last start with this plugin enabled for this profile" -- not "ever, once."
    """
    directory = Path(profile_home) / MARKER_RELATIVE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    path = directory / MARKER_FILENAME
    payload = {
        "plugin_id": plugin_id,
        "profile": profile,
        "registered_at": time.time(),
        "pid": os.getpid(),
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    tmp_path.replace(path)  # atomic on POSIX -- no reader ever sees a partial write
    return path


def _pid_alive(pid: object) -> bool:
    """Best-effort POSIX liveness check. A malformed/missing pid or an unsupported platform is
    treated as "cannot confirm alive" -- callers decide whether that alone should block; it never
    silently upgrades to "confirmed alive."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by someone else -- still evidence a process is running.
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class MarkerCheck:
    ok: bool
    reason: str


def check_marker(
    profile_home: Path,
    *,
    expected_profile: str,
    expected_plugin_id: Optional[str] = None,
    max_age_seconds: int = DEFAULT_MAX_MARKER_AGE_SECONDS,
    require_pid_alive: bool = False,
    require_parent_pid: Optional[int] = None,
) -> MarkerCheck:
    """Validate the marker for *expected_profile*. Every failure mode returns ``ok=False`` with a
    specific reason -- there is no path that returns ok=True on missing/malformed/stale/
    mismatched input.

    ``require_parent_pid`` is the hard startup gate the guarded launcher actually uses: it must
    equal the marker's recorded ``pid`` EXACTLY. Freshness (``max_age_seconds``) alone is not
    sufficient -- a marker can be recent yet belong to a Hermes process that has since exited and
    been replaced by a new one that never re-registered this plugin (disabled mid-restart, a
    config mistake, etc.); a "just alive" check on the recorded pid does not catch that either,
    since PID reuse or an unrelated process happening to hold that pid would pass. Only "this
    marker's pid is literally MY parent process" (the launcher's own ``os.getppid()``, i.e. the
    Hermes process that just spawned it) proves the marker came from the process that is trying
    to launch this MCP server right now, not a stale prior one. ``require_pid_alive`` is kept as
    a weaker, optional check (liveness without identity) for callers that want it explicitly;
    the launcher does not use it.
    """
    path = marker_path(profile_home)
    if not path.is_file():
        return MarkerCheck(False, f"no plugin-registration marker at {path}")

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        return MarkerCheck(False, f"marker at {path} is unreadable/invalid JSON: {exc}")

    if not isinstance(data, dict):
        return MarkerCheck(False, f"marker at {path} is not a JSON object")

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

    marker_pid = data.get("pid")
    if require_parent_pid is not None:
        if not isinstance(marker_pid, int) or marker_pid != require_parent_pid:
            return MarkerCheck(False, (
                f"marker pid {marker_pid!r} does not match the launcher's parent process "
                f"{require_parent_pid!r} -- stale marker from a prior/disabled-plugin Hermes "
                "process, not the one launching this server now"
            ))
    elif require_pid_alive and not _pid_alive(marker_pid):
        return MarkerCheck(False, f"marker pid {marker_pid!r} is not a live process")

    return MarkerCheck(True, "marker valid")
