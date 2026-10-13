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

import ctypes
import json
import os
import re
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
#   linux:<boot_id>:<starttime clock ticks since boot>    (from /proc; boot_id rules out reboots)
#   darwin:<start tv_sec>.<start tv_usec, 6 digits>       (kernel, via libproc proc_pidinfo)
_IDENTITY_RE = re.compile(
    r"^(?:linux:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:[0-9]{1,20}"
    r"|darwin:[0-9]{1,20}\.[0-9]{6})$"
)

# macOS <sys/proc_info.h>: proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &struct proc_bsdinfo, size).
_LIBPROC_PATH = "/usr/lib/libproc.dylib"  # resolved from the dyld shared cache, not a file on disk
_PROC_PIDTBSDINFO = 3
_PROC_BSDINFO_SIZE = 136  # sizeof(struct proc_bsdinfo), checked against the SDK header


class _ProcBsdInfo(ctypes.Structure):
    """``struct proc_bsdinfo`` field-for-field (MAXCOMLEN = 16). Offsets verified against a C
    program compiled with the macOS SDK: pbi_pid @12, pbi_start_tvsec @120, pbi_start_tvusec @128."""
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32), ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32), ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32), ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32), ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64),
    ]


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


def _load_proc_pidinfo():
    try:
        fn = ctypes.CDLL(_LIBPROC_PATH, use_errno=True).proc_pidinfo
    except (OSError, AttributeError) as exc:
        raise ProcessIdentityError(f"libproc proc_pidinfo unavailable: {exc}") from exc
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    return fn


def _darwin_identity(pid: int, *, proc_pidinfo=None) -> str:
    """Kernel process start time with microsecond precision, from proc_pidinfo(PROC_PIDTBSDINFO).
    There is deliberately no fallback (e.g. second-precision `ps`): anything short of a complete,
    self-consistent kernel answer raises, and callers fail closed."""
    if ctypes.sizeof(_ProcBsdInfo) != _PROC_BSDINFO_SIZE:
        raise ProcessIdentityError(f"proc_bsdinfo layout mismatch ({ctypes.sizeof(_ProcBsdInfo)} bytes)")
    fn = proc_pidinfo if proc_pidinfo is not None else _load_proc_pidinfo()
    info = _ProcBsdInfo()
    ctypes.set_errno(0)
    try:
        written = fn(pid, _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    except (ctypes.ArgumentError, OSError) as exc:
        raise ProcessIdentityError(f"proc_pidinfo failed for pid {pid}: {exc}") from exc
    if written != ctypes.sizeof(info):
        # 0 with errno ESRCH (no such process) / EPERM (not ours), or a short write: the buffer is
        # not trustworthy -- on failure the kernel leaves it unfilled.
        raise ProcessIdentityError(
            f"proc_pidinfo(PROC_PIDTBSDINFO) for pid {pid} returned {written!r} "
            f"(errno {ctypes.get_errno()}), expected {ctypes.sizeof(info)}")
    if info.pbi_pid != pid:
        raise ProcessIdentityError(f"proc_pidinfo answered for pid {info.pbi_pid}, not {pid}")
    if info.pbi_start_tvsec <= 0 or info.pbi_start_tvusec >= 1_000_000:
        raise ProcessIdentityError(
            f"implausible start time {info.pbi_start_tvsec}.{info.pbi_start_tvusec} for pid {pid}")
    return f"darwin:{info.pbi_start_tvsec}.{info.pbi_start_tvusec:06d}"


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
