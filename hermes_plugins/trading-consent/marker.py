"""Profile-bound plugin-registration marker: written by __init__.py:register() when Hermes loads
this plugin for a profile, checked by guarded_mcp_launcher.py before it will exec the real MCP
server command.

Why a file at all: the guarded launcher runs as a completely separate OS process (whatever
`mcp_servers.<name>.command` names), spawned by Hermes fresh each time it needs that MCP server.
It has no access to the parent Hermes process's in-memory plugin registry (PluginContext /
ctx.has_plugin()) and Hermes's own subprocess-env filtering (tools/mcp_tool_config.py:
_build_safe_env) does not pass HERMES_HOME through by default -- so the launcher cannot assume it
knows its own profile home from its environment alone. A small, profile-scoped marker file bound
to one exact Hermes process (pid plus process start identity) is the practical signal available
across that process boundary.

This is a tamper-resistance *speed bump*, not a cryptographic guarantee: anyone with filesystem
write access to the profile directory could fabricate a marker. Its purpose is to fail hard,
loudly, and by default when the plugin is simply not registered (disabled, removed, never
installed, or a config mistake) -- the exact scenario that would otherwise leave the gate silently
absent with no error at all.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MARKER_RELATIVE_DIR = ".trading_consent"
MARKER_FILENAME_TEMPLATE = "registered.{pid}.json"

# A process-instance identity is "<platform>:<start-of-process value>" and is bound to one exact
# process: the same pid reused later by a different process has a different start value. It
# replaces wall-clock marker age as the authorization condition -- a long-running gateway's marker
# stays valid for exactly as long as that gateway process lives, and no longer.
#   linux:<boot_id>:<starttime clock ticks since boot>   (from /proc; boot_id rules out reboots)
#   darwin:<process start, UTC epoch seconds>            (from /bin/ps -o lstart=)
_IDENTITY_RE = re.compile(
    r"^(?:linux:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:[0-9]{1,20}"
    r"|darwin:[0-9]{1,20})$"
)
_PS_PATH = "/bin/ps"
_PS_TIMEOUT_SECONDS = 5
_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1)}


class ProcessIdentityError(Exception):
    """The start identity of a process could not be determined. Callers fail closed."""


def _valid_pid(pid: object) -> bool:
    return isinstance(pid, int) and not isinstance(pid, bool) and pid > 0


def is_valid_identity(identity: object) -> bool:
    return isinstance(identity, str) and _IDENTITY_RE.fullmatch(identity) is not None


def _parse_proc_stat_starttime(stat_text: str) -> int:
    """Field 22 (starttime) of /proc/<pid>/stat. The comm field (2) is parenthesised and may
    itself contain spaces or ')', so fields are counted from the LAST ')'."""
    close = stat_text.rfind(")")
    if close < 0:
        raise ProcessIdentityError("malformed /proc stat: no comm terminator")
    fields = stat_text[close + 1:].split()
    # fields[0] is field 3 (state), so field 22 is fields[19].
    if len(fields) < 20 or not fields[19].isdigit():
        raise ProcessIdentityError("malformed /proc stat: no numeric starttime field")
    return int(fields[19])


def _linux_identity(pid: int, proc_root: Path = Path("/proc")) -> str:
    try:
        boot_id = (proc_root / "sys" / "kernel" / "random" / "boot_id").read_text(encoding="ascii").strip()
        stat_text = (proc_root / str(pid) / "stat").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ProcessIdentityError(f"cannot read /proc identity for pid {pid}: {exc}") from exc
    return f"linux:{boot_id}:{_parse_proc_stat_starttime(stat_text)}"


def _parse_ps_lstart(text: str) -> int:
    """`ps -o lstart=` under LC_ALL=C TZ=UTC, e.g. ``Mon Oct 12 20:02:17 2026`` -> UTC epoch
    seconds. Parsed by hand (not time.strptime) so the host process's locale cannot change it."""
    parts = text.split()
    if len(parts) != 5 or parts[1] not in _MONTHS:
        raise ProcessIdentityError(f"unrecognised ps lstart output {text.strip()!r}")
    _weekday, month, day, clock, year = parts
    hms = clock.split(":")
    if not (day.isdigit() and year.isdigit() and len(hms) == 3 and all(x.isdigit() for x in hms)):
        raise ProcessIdentityError(f"unrecognised ps lstart output {text.strip()!r}")
    import calendar
    y, mo, d, hh, mm, ss = int(year), _MONTHS[month], int(day), int(hms[0]), int(hms[1]), int(hms[2])
    # timegm silently normalises out-of-range fields (Oct 32 -> Nov 1), so range-check first.
    if not (1970 <= y <= 9999 and 1 <= d <= calendar.monthrange(y, mo)[1]
            and hh < 24 and mm < 60 and ss < 60):
        raise ProcessIdentityError(f"unrecognised ps lstart output {text.strip()!r}")
    return calendar.timegm((y, mo, d, hh, mm, ss, 0, 0, 0))


def _darwin_identity(pid: int) -> str:
    # Absolute path, explicit argv, no shell, and a pinned environment: lstart is printed in the
    # caller's local time zone, and Hermes filters the environment it gives MCP subprocesses, so
    # writer and launcher must not depend on whatever TZ/locale each happened to inherit.
    try:
        proc = subprocess.run(
            [_PS_PATH, "-o", "lstart=", "-p", str(pid)],
            capture_output=True, text=True, timeout=_PS_TIMEOUT_SECONDS,
            env={"LC_ALL": "C", "TZ": "UTC", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProcessIdentityError(f"cannot run {_PS_PATH} for pid {pid}: {exc}") from exc
    if proc.returncode != 0 or not proc.stdout.strip():
        raise ProcessIdentityError(f"{_PS_PATH} reports no process {pid} (exit {proc.returncode})")
    return f"darwin:{_parse_ps_lstart(proc.stdout)}"


def process_start_identity(pid: int) -> str:
    """Identity of the process currently holding *pid*. Raises ProcessIdentityError when it
    cannot be determined -- including on unsupported platforms (Windows and others), where the
    guarded launcher therefore always fails closed."""
    if not _valid_pid(pid):
        raise ProcessIdentityError(f"invalid pid {pid!r}")
    if sys.platform.startswith("linux"):
        identity = _linux_identity(pid)
    elif sys.platform == "darwin":
        identity = _darwin_identity(pid)
    else:
        raise ProcessIdentityError(f"process start identity is not supported on {sys.platform!r}")
    if not is_valid_identity(identity):
        raise ProcessIdentityError(f"malformed process identity {identity!r} for pid {pid}")
    return identity


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


def write_marker(
    profile_home: Path, *, profile: str, plugin_id: str, pid: Optional[int] = None,
    process_identity: Optional[str] = None,
) -> Path:
    """Write/refresh the marker for this process (*pid* defaults to ``os.getpid()``), recording
    its process-instance identity (looked up when *process_identity* is not given). Called from
    register(ctx) on every plugin load. Raises ProcessIdentityError -- writing no marker at all --
    when the identity cannot be determined, so the launcher then fails closed.

    Never deletes or touches any other process's marker. Stale markers left by exited processes
    are inert -- their identity no longer matches any live process -- and are deliberately not
    garbage-collected: deciding "that pid is dead, delete its file" races with PID reuse (a new
    Hermes process can be assigned that pid and write the same filename between the liveness
    check and the unlink), which would delete a valid marker.
    """
    pid = os.getpid() if pid is None else pid
    path = marker_path(profile_home, pid)
    identity = process_start_identity(pid) if process_identity is None else process_identity
    if not is_valid_identity(identity):
        raise ProcessIdentityError(f"refusing to write malformed process identity {identity!r}")
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    payload = {
        "plugin_id": plugin_id,
        "profile": profile,
        "pid": pid,
        "process_identity": identity,
        "registered_at": time.time(),  # informational only -- never an authorization condition
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
) -> MarkerCheck:
    """Validate the marker written by Hermes process *parent_pid* for *expected_profile*. Every
    failure mode returns ``ok=False`` with a specific reason -- there is no path that returns
    ok=True on missing/malformed/mismatched input or when identity cannot be determined.

    The guarded launcher passes its own ``os.getppid()`` -- the Hermes process that just spawned
    it -- and only that process's file (``registered.<parent_pid>.json``) is ever read; the legacy
    shared ``registered.json`` is ignored. The recorded ``pid`` must equal *parent_pid*, and the
    recorded ``process_identity`` must equal the identity of the process holding *parent_pid*
    right now. That last check is what makes a marker valid for exactly the lifetime of the
    process that wrote it: a marker from an exited process whose pid the OS has since reused for a
    different process (e.g. one where the plugin is disabled) does not match. There is no
    wall-clock age limit -- a gateway that has run for weeks is still the same process.
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

    marker_identity = data.get("process_identity")
    if not is_valid_identity(marker_identity):
        return MarkerCheck(False, f"marker at {path} has a missing/malformed process_identity")

    try:
        current_identity = process_start_identity(parent_pid)
    except ProcessIdentityError as exc:
        return MarkerCheck(False, f"cannot determine identity of parent process {parent_pid}: {exc}")

    if marker_identity != current_identity:
        return MarkerCheck(False, (
            f"marker process_identity {marker_identity!r} does not match parent process "
            f"{parent_pid}'s current identity {current_identity!r} -- the marker was written by "
            "an earlier process that held the same pid"
        ))

    return MarkerCheck(True, "marker valid")
