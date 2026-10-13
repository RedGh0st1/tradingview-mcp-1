"""Per-process registration markers bound to process start identity, and the guarded launcher's
hard stop vs. exec path -- in-process (mocked exec) and as a real child process of this test
process (so its os.getppid() is genuinely os.getpid() here). Structural check that neither the
launcher nor the marker module ever uses a shell."""

from __future__ import annotations

import calendar
import json
import os
import subprocess
import sys
import time

import pytest

SUPPORTED_PLATFORM = sys.platform == "darwin" or sys.platform.startswith("linux")
needs_identity = pytest.mark.skipif(
    not SUPPORTED_PLATFORM, reason="process start identity is unsupported (fails closed) here",
)

ID_A = "darwin:1700000001.000000"
ID_B = "darwin:1700000002.000000"


def _fake_identities(monkeypatch, module, mapping):
    """Replace *module*'s live identity lookup with a fixed pid -> identity table; unknown pids
    raise, exactly like a pid with no running process."""
    def lookup(pid):
        if pid not in mapping:
            raise module.ProcessIdentityError(f"no process {pid}")
        return mapping[pid]
    monkeypatch.setattr(module, "process_start_identity", lookup)


def _rewrite(path, **changes):
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def _same_second_other_process(identity):
    """Same platform/format as *identity* and, on macOS, the SAME start second with a different
    microsecond -- what a different process that received the same pid within that second would
    report. (Linux starttime is a single tick counter, so it is simply bumped.)"""
    head, _, tail = identity.rpartition(":")
    if head == "darwin":
        sec, usec = tail.split(".")
        return f"darwin:{sec}.{(int(usec) + 1) % 1_000_000:06d}"
    return f"{head}:{int(tail) + 1}"


# --- identity: format, parsing, platform lookup --------------------------------------------------

@pytest.mark.parametrize("identity", [
    "darwin:1791849777.000000", "darwin:1791850258.373621",
    "linux:0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0:123456",
])
def test_valid_identity_formats(marker_mod, identity):
    assert marker_mod.is_valid_identity(identity)


@pytest.mark.parametrize("identity", [
    None, "", 1791849777, "darwin:", "darwin:abc", "darwin:-1", "darwin:1.000000 ", " darwin:1.000000",
    "win32:123", "linux:not-a-uuid:1", "linux:0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0:",
    "darwin:1.000000\n", "darwin:1:2", ["darwin:1.000000"],
    "darwin:1791849777",  # the earlier second-precision ps format is no longer accepted
    "darwin:1791849777.37362", "darwin:1791849777.3736210", "darwin:1791849777.",
])
def test_malformed_identities_are_rejected(marker_mod, identity):
    assert not marker_mod.is_valid_identity(identity)


def test_proc_bsdinfo_struct_matches_the_sdk_layout(marker_mod):
    """Offsets from a C program compiled against the macOS SDK's <sys/proc_info.h>."""
    info = marker_mod._ProcBsdInfo
    assert marker_mod.ctypes.sizeof(info) == 136 == marker_mod._PROC_BSDINFO_SIZE
    assert info.pbi_pid.offset == 12
    assert info.pbi_start_tvsec.offset == 120
    assert info.pbi_start_tvusec.offset == 128
    assert marker_mod._PROC_PIDTBSDINFO == 3


def _fake_proc_pidinfo(marker_mod, *, ret=None, pid=None, tvsec=1700000000, tvusec=123, errno=0, raises=None):
    """Stand-in for libproc's proc_pidinfo: fills the caller's struct like the kernel would."""
    calls = []

    def fn(req_pid, flavor, arg, ref, size):
        calls.append((req_pid, flavor, arg, size))
        if raises is not None:
            raise raises
        info = ref._obj
        info.pbi_pid = req_pid if pid is None else pid
        info.pbi_start_tvsec = tvsec
        info.pbi_start_tvusec = tvusec
        marker_mod.ctypes.set_errno(errno)
        return size if ret is None else ret
    fn.calls = calls
    return fn


def test_darwin_identity_from_proc_pidinfo(marker_mod):
    fn = _fake_proc_pidinfo(marker_mod, tvsec=1791850258, tvusec=373621)
    assert marker_mod._darwin_identity(4242, proc_pidinfo=fn) == "darwin:1791850258.373621"
    assert fn.calls == [(4242, 3, 0, 136)]


def test_darwin_identity_zero_pads_microseconds(marker_mod):
    fn = _fake_proc_pidinfo(marker_mod, tvsec=1791850258, tvusec=7)
    assert marker_mod._darwin_identity(4242, proc_pidinfo=fn) == "darwin:1791850258.000007"


@pytest.mark.parametrize("kwargs", [
    {"ret": 0, "errno": 3},          # ESRCH: no such process
    {"ret": 0, "errno": 1},          # EPERM: another user's process
    {"ret": -1, "errno": 22},
    {"ret": 100},                    # short write: struct not fully filled
    {"ret": 200},                    # over-long answer: not the struct we asked for
    {"pid": 4343},                   # kernel answered for a different pid
    {"tvsec": 0},                    # no start time
    {"tvusec": 1_000_000},           # not a valid microsecond value
    {"tvusec": 2 ** 63},
    {"raises": OSError("dlsym")},
    {"raises": "argument_error"},
])
def test_darwin_identity_rejects_failed_or_inconsistent_answers(marker_mod, kwargs):
    if kwargs.get("raises") == "argument_error":
        kwargs["raises"] = marker_mod.ctypes.ArgumentError("bad argument")
    fn = _fake_proc_pidinfo(marker_mod, **kwargs)
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod._darwin_identity(4242, proc_pidinfo=fn)


@pytest.mark.parametrize("error", [OSError("image not found"), AttributeError("proc_pidinfo")])
def test_libproc_unavailable_fails_closed(marker_mod, monkeypatch, error):
    def broken_cdll(*args, **kwargs):
        raise error
    monkeypatch.setattr(marker_mod.ctypes, "CDLL", broken_cdll)
    with pytest.raises(marker_mod.ProcessIdentityError, match="libproc"):
        marker_mod._darwin_identity(os.getpid())


def test_struct_layout_mismatch_fails_closed(marker_mod, monkeypatch):
    monkeypatch.setattr(marker_mod, "_PROC_BSDINFO_SIZE", 140)
    with pytest.raises(marker_mod.ProcessIdentityError, match="layout"):
        marker_mod._darwin_identity(4242, proc_pidinfo=_fake_proc_pidinfo(marker_mod))


def test_no_second_precision_ps_fallback(marker_mod, plugin_dir, monkeypatch):
    """When the kernel identity is unavailable on macOS, nothing weaker is substituted."""
    source = (plugin_dir / "marker.py").read_text(encoding="utf-8")
    assert "/bin/ps" not in source
    assert "import subprocess" not in source
    monkeypatch.setattr(marker_mod.sys, "platform", "darwin")
    monkeypatch.setattr(marker_mod.ctypes, "CDLL", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod.process_start_identity(os.getpid())


def test_same_second_pid_reuse_is_rejected(marker_mod, monkeypatch, fresh_profile_home):
    """Pi R-001: the earlier holder of pid 4242 and the later one started within the same second.
    Whole-second identity would have matched; microsecond kernel identity must not."""
    earlier, later = "darwin:1791850258.000001", "darwin:1791850258.999999"
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=4242, process_identity=earlier)
    _fake_identities(monkeypatch, marker_mod, {4242: later})
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader")
    assert not result.ok
    assert "does not match" in result.reason


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS kernel identity")
def test_real_same_second_processes_have_distinct_identities(marker_mod):
    """Real processes started within the same wall-clock second get different identities."""
    for _ in range(50):
        procs = [subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"],
                                  stdin=subprocess.PIPE) for _ in range(2)]
        try:
            ids = [marker_mod.process_start_identity(p.pid) for p in procs]
        finally:
            for p in procs:
                p.stdin.close()
                p.wait(timeout=60)
        seconds = {i.split(":")[1].split(".")[0] for i in ids}
        if len(seconds) == 1:
            assert ids[0] != ids[1]
            return
    pytest.fail("could not start two processes within the same second in 50 attempts")


def _fake_proc(tmp_path, *, pid, stat, boot_id="0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"):
    (tmp_path / "sys" / "kernel" / "random").mkdir(parents=True)
    (tmp_path / "sys" / "kernel" / "random" / "boot_id").write_text(boot_id + "\n")
    (tmp_path / str(pid)).mkdir()
    (tmp_path / str(pid) / "stat").write_text(stat)
    return tmp_path


def test_linux_identity_from_proc(marker_mod, tmp_path):
    # comm containing spaces and ')' must not shift the field count
    fields_after_comm = ["S"] + [str(i) for i in range(4, 22)] + ["987654321"] + ["0"] * 30
    stat = "4242 (evil ) name) " + " ".join(fields_after_comm)
    root = _fake_proc(tmp_path, pid=4242, stat=stat)
    assert marker_mod._linux_identity(4242, proc_root=root) == (
        "linux:0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0:987654321")


@pytest.mark.parametrize("stat", ["4242 no-parens S 1 2", "4242 (x) S 1 2 3", "4242 (x) " + "S " + "y " * 25])
def test_linux_identity_malformed_stat_raises(marker_mod, tmp_path, stat):
    root = _fake_proc(tmp_path, pid=4242, stat=stat)
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod._linux_identity(4242, proc_root=root)


def test_linux_identity_missing_process_raises(marker_mod, tmp_path):
    root = _fake_proc(tmp_path, pid=4242, stat="4242 (x) S")
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod._linux_identity(9999, proc_root=root)


@pytest.mark.parametrize("platform", ["win32", "cygwin", "freebsd14", "emscripten"])
def test_unsupported_platform_fails_closed(marker_mod, monkeypatch, platform):
    monkeypatch.setattr(marker_mod.sys, "platform", platform)
    with pytest.raises(marker_mod.ProcessIdentityError, match="not supported"):
        marker_mod.process_start_identity(os.getpid())


@pytest.mark.parametrize("bad_pid", [0, -1, None, "123", True, 1.5])
def test_identity_of_invalid_pid_raises(marker_mod, bad_pid):
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod.process_start_identity(bad_pid)


@needs_identity
def test_real_identity_is_stable_and_well_formed(marker_mod):
    first = marker_mod.process_start_identity(os.getpid())
    assert marker_mod.is_valid_identity(first)
    assert marker_mod.process_start_identity(os.getpid()) == first


@needs_identity
def test_real_identity_of_exited_process_is_unavailable(marker_mod):
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod.process_start_identity(child.pid)


# --- marker write/check --------------------------------------------------------------------------

@needs_identity
def test_write_then_check_round_trip(marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    data = json.loads(path.read_text())
    assert path.name == f"registered.{os.getpid()}.json"
    assert data["pid"] == os.getpid()
    assert data["process_identity"] == marker_mod.process_start_identity(os.getpid())
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert result.ok, result.reason


def test_long_running_process_marker_is_accepted(marker_mod, monkeypatch, fresh_profile_home):
    """No wall-clock expiry: a gateway that registered 30 days ago is still the same process."""
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=4242, process_identity=ID_A)
    _rewrite(path, registered_at=time.time() - 30 * 24 * 3600)
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader")
    assert result.ok, result.reason


@pytest.mark.parametrize("registered_at", [time.time() + 999_999, None, "x"])
def test_registered_at_is_not_an_authorization_condition(marker_mod, monkeypatch, fresh_profile_home, registered_at):
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=4242, process_identity=ID_A)
    _rewrite(path, registered_at=registered_at)
    assert marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader").ok


def test_pid_reuse_identity_mismatch_is_rejected(marker_mod, monkeypatch, fresh_profile_home):
    """A marker from an exited process whose pid the OS has since handed to a different process."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=4242, process_identity=ID_A)
    _fake_identities(monkeypatch, marker_mod, {4242: ID_B})  # same pid, a newer process
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader")
    assert not result.ok
    assert "does not match" in result.reason


def test_identity_lookup_failure_at_check_fails_closed(marker_mod, monkeypatch, fresh_profile_home):
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=4242, process_identity=ID_A)
    _fake_identities(monkeypatch, marker_mod, {})
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader")
    assert not result.ok
    assert "cannot determine identity" in result.reason


def test_identity_lookup_failure_at_write_writes_nothing(marker_mod, monkeypatch, fresh_profile_home):
    _fake_identities(monkeypatch, marker_mod, {})
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=4242)
    marker_dir = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR
    assert not marker_dir.exists() or list(marker_dir.iterdir()) == []


def test_write_refuses_malformed_identity(marker_mod, fresh_profile_home):
    with pytest.raises(marker_mod.ProcessIdentityError):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                pid=4242, process_identity="darwin:not-a-number")


@pytest.mark.parametrize("identity", [None, "", 1700000001, "darwin:abc", "win32:1", ["darwin:1"]])
def test_malformed_identity_in_marker_is_rejected(marker_mod, monkeypatch, fresh_profile_home, identity):
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=4242, process_identity=ID_A)
    _rewrite(path, process_identity=identity)
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader")
    assert not result.ok
    assert "process_identity" in result.reason


def test_missing_identity_key_is_rejected(marker_mod, monkeypatch, fresh_profile_home):
    """Markers written by the previous plugin version (no process_identity) do not pass."""
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=4242, process_identity=ID_A)
    data = json.loads(path.read_text())
    del data["process_identity"]
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader").ok


def test_missing_marker_fails(marker_mod, fresh_profile_home):
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader")
    assert not result.ok


def test_wrong_profile_in_marker_fails(marker_mod, monkeypatch, fresh_profile_home):
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    marker_mod.write_marker(fresh_profile_home, profile="tradingresearch", plugin_id="trading-consent",
                            pid=4242, process_identity=ID_A)
    assert not marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader").ok


def test_wrong_plugin_id_fails(marker_mod, monkeypatch, fresh_profile_home):
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="some-other-plugin",
                            pid=4242, process_identity=ID_A)
    assert not marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader",
                                       expected_plugin_id="trading-consent").ok


def test_malformed_json_fails(marker_mod, fresh_profile_home):
    path = marker_mod.marker_path(fresh_profile_home, os.getpid())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert not marker_mod.check_marker(fresh_profile_home, parent_pid=os.getpid(), expected_profile="trader").ok


def test_check_reads_only_the_requested_pids_marker(marker_mod, monkeypatch, fresh_profile_home):
    """A valid marker from a DIFFERENT live process never satisfies a check for another pid."""
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A, 4343: ID_B})
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=4242, process_identity=ID_A)
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=4343, expected_profile="trader")
    assert not result.ok
    assert "registered.4343.json" in result.reason


def test_pid_inside_file_must_match_filename_pid(marker_mod, monkeypatch, fresh_profile_home):
    _fake_identities(monkeypatch, marker_mod, {111111: ID_A, 222222: ID_A})
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=111111, process_identity=ID_A)
    path.rename(marker_mod.marker_path(fresh_profile_home, 222222))
    result = marker_mod.check_marker(fresh_profile_home, parent_pid=222222, expected_profile="trader")
    assert not result.ok
    assert "111111" in result.reason


@pytest.mark.parametrize("bad_pid", [0, -1, None, "123", True, 1.5])
def test_invalid_parent_pid_fails_closed(marker_mod, fresh_profile_home, bad_pid):
    assert not marker_mod.check_marker(fresh_profile_home, parent_pid=bad_pid, expected_profile="trader").ok


@pytest.mark.parametrize("bad_pid", [0, -1, "123", True])
def test_write_marker_rejects_invalid_pid(marker_mod, fresh_profile_home, bad_pid):
    with pytest.raises(ValueError):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                pid=bad_pid, process_identity=ID_A)


def test_legacy_shared_marker_is_ignored(marker_mod, monkeypatch, fresh_profile_home):
    """The pre-fix single registered.json -- even with the right pid, profile and identity -- is
    never read."""
    _fake_identities(monkeypatch, marker_mod, {4242: ID_A})
    legacy = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR / "registered.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "plugin_id": "trading-consent", "profile": "trader", "registered_at": time.time(),
        "pid": 4242, "process_identity": ID_A,
    }), encoding="utf-8")
    assert not marker_mod.check_marker(fresh_profile_home, parent_pid=4242, expected_profile="trader").ok


def test_one_process_writing_cannot_invalidate_another(marker_mod, monkeypatch, fresh_profile_home):
    """The live defect: the gateway registers, then one-shot CLI processes load the plugin and
    register too. The gateway's marker must be byte-for-byte untouched and still valid."""
    ids = {4242: ID_A, 56147: "darwin:1700000147.000000", 56148: "darwin:1700000148.000000"}
    _fake_identities(monkeypatch, marker_mod, ids)
    gateway = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                      pid=4242, process_identity=ID_A)
    before = gateway.read_bytes()
    for cli_pid in (56147, 56148):
        marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                pid=cli_pid, process_identity=ids[cli_pid])
    assert gateway.read_bytes() == before
    for pid in ids:
        assert marker_mod.check_marker(fresh_profile_home, parent_pid=pid, expected_profile="trader").ok


def test_write_leaves_no_temp_files_and_never_deletes_other_markers(marker_mod, fresh_profile_home):
    """Stale markers from exited processes are inert and deliberately not garbage-collected
    (deleting by "pid is dead" races with PID reuse) -- a write must never remove them."""
    stale = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                    pid=999_999_999, process_identity=ID_A)
    mine = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=4242, process_identity=ID_B)
    assert sorted(p.name for p in stale.parent.iterdir()) == sorted([stale.name, mine.name])


def test_marker_file_permissions_are_restrictive(marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                                   pid=4242, process_identity=ID_A)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_marker_source_never_uses_a_shell(plugin_dir):
    source = (plugin_dir / "marker.py").read_text(encoding="utf-8")
    assert "shell=True" not in source
    assert "os.system(" not in source
    assert "os.popen(" not in source


_WRITER_SCRIPT = """
import os, sys, time
sys.path.insert(0, sys.argv[1])
import marker
go = float(sys.argv[3])
while time.time() < go:
    pass
for _ in range(10):
    marker.write_marker(sys.argv[2], profile="trader", plugin_id="trading-consent")
print(marker.marker_path(sys.argv[2], os.getpid()).name, flush=True)
sys.stdin.read()  # stay alive until the test has checked this process's marker
"""


@needs_identity
def test_concurrent_live_process_registrations_coexist(plugin_dir, marker_mod, fresh_profile_home):
    """Real OS processes, released at the same instant, each re-registering repeatedly: while they
    are alive every process's own marker validates against its real identity, no temp file leaks,
    and once they exit their markers are inert (identity unavailable), not deleted."""
    go = time.time() + 1.0
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER_SCRIPT, str(plugin_dir), str(fresh_profile_home), str(go)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(6)
    ]
    try:
        for proc in procs:
            assert proc.stdout.readline().strip() == f"registered.{proc.pid}.json"
        marker_dir = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR
        assert sorted(p.name for p in marker_dir.iterdir()) == sorted(f"registered.{p.pid}.json" for p in procs)
        for proc in procs:
            check = marker_mod.check_marker(fresh_profile_home, parent_pid=proc.pid, expected_profile="trader",
                                            expected_plugin_id="trading-consent")
            assert check.ok, check.reason
    finally:
        for proc in procs:
            proc.stdin.close()
        for proc in procs:
            proc.wait(timeout=60)
    for proc in procs:
        assert proc.returncode == 0, proc.stderr.read()
        assert not marker_mod.check_marker(fresh_profile_home, parent_pid=proc.pid, expected_profile="trader").ok
    assert len(list(marker_dir.iterdir())) == len(procs)


# --- guarded_mcp_launcher.py (in-process, exec mocked) ---------------------------------------------

def _launcher_marker_module():
    """marker.py is imported fresh inside guarded_mcp_launcher.py's own module namespace (see
    conftest) -- write/patch through THAT copy so the launcher's check_marker sees it."""
    return sys.modules["marker"]


def _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home, *, profile="trader"):
    monkeypatch.setenv(launcher_mod.ENV_PROFILE, profile)
    monkeypatch.setenv(launcher_mod.ENV_PROFILE_HOME, str(fresh_profile_home))
    execs = []
    monkeypatch.setattr(os, "execvp", lambda resolved, args: execs.append((resolved, args)))
    monkeypatch.setattr(launcher_mod, "_which", lambda cmd: f"/usr/bin/{cmd}")
    return execs


def test_launcher_hard_stops_without_env_var(launcher_mod, monkeypatch):
    monkeypatch.delenv(launcher_mod.ENV_PROFILE, raising=False)
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main(["node", "server.js"])
    assert exc.value.code != 0


def test_launcher_hard_stops_when_marker_absent(launcher_mod, monkeypatch, fresh_profile_home):
    execs = _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home)
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main(["node", "server.js"])
    assert exc.value.code != 0
    assert execs == []  # the real command must never be exec'd


def test_launcher_hard_stops_when_marker_profile_mismatched(launcher_mod, monkeypatch, fresh_profile_home):
    marker = _launcher_marker_module()
    _fake_identities(monkeypatch, marker, {os.getppid(): ID_A})
    marker.write_marker(fresh_profile_home, profile="tradingresearch", plugin_id="trading-consent",
                        pid=os.getppid(), process_identity=ID_A)
    execs = _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home)
    with pytest.raises(SystemExit):
        launcher_mod.main(["node", "server.js"])
    assert execs == []


def test_launcher_hard_stops_when_parent_identity_lookup_fails(launcher_mod, monkeypatch, fresh_profile_home):
    marker = _launcher_marker_module()
    marker.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                        pid=os.getppid(), process_identity=ID_A)
    _fake_identities(monkeypatch, marker, {})
    execs = _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home)
    with pytest.raises(SystemExit) as exc:
        launcher_mod.main(["node", "server.js"])
    assert exc.value.code != 0
    assert execs == []


def test_launcher_hard_stops_on_unsupported_platform(launcher_mod, monkeypatch, fresh_profile_home):
    marker = _launcher_marker_module()
    marker.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                        pid=os.getppid(), process_identity=ID_A)
    monkeypatch.setattr(marker.sys, "platform", "win32")
    execs = _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home)
    with pytest.raises(SystemExit):
        launcher_mod.main(["node", "server.js"])
    assert execs == []


def test_launcher_hard_stops_on_reused_parent_pid(launcher_mod, monkeypatch, fresh_profile_home):
    marker = _launcher_marker_module()
    marker.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                        pid=os.getppid(), process_identity=ID_A)
    _fake_identities(monkeypatch, marker, {os.getppid(): ID_B})
    execs = _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home)
    with pytest.raises(SystemExit):
        launcher_mod.main(["node", "server.js"])
    assert execs == []


def test_launcher_execs_real_command_when_marker_valid(launcher_mod, monkeypatch, fresh_profile_home):
    marker = _launcher_marker_module()
    _fake_identities(monkeypatch, marker, {os.getppid(): ID_A})
    marker.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                        pid=os.getppid(), process_identity=ID_A)
    execs = _prepare_launcher(launcher_mod, monkeypatch, fresh_profile_home)

    # main() only calls sys.exit() via _fail(); with os.execvp mocked to a recorder, a successful
    # exec path falls off the end. In real execution a successful execvp() never returns.
    launcher_mod.main(["node", "server.js", "--flag"])

    assert execs == [("/usr/bin/node", ["node", "server.js", "--flag"])]


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


# --- end-to-end: the real launcher as a real child process, real parent identity -------------------

def _run_launcher(plugin_dir, profile_home, *, profile="trader"):
    """Run guarded_mcp_launcher.py as a real child of THIS test process, so its os.getppid() is
    genuinely os.getpid() here and identity lookups are real. The 'real command' is a harmless
    python one-liner."""
    env = dict(os.environ)
    env["TRADING_CONSENT_PROFILE"] = profile
    env["TRADING_CONSENT_PROFILE_HOME"] = str(profile_home)
    return subprocess.run(
        [sys.executable, str(plugin_dir / "guarded_mcp_launcher.py"),
         sys.executable, "-c", "print('REAL-MCP-STARTED')"],
        env=env, capture_output=True, text=True, timeout=60,
    )


def _assert_blocked(proc, *needles):
    assert proc.returncode != 0
    assert "REAL-MCP-STARTED" not in proc.stdout
    assert "HARD STOP" in proc.stderr
    for needle in needles:
        assert needle in proc.stderr


@needs_identity
def test_real_launcher_selects_only_its_parents_marker(plugin_dir, marker_mod, fresh_profile_home):
    """This process (the 'gateway') and other registered processes all hold markers; one is written
    after ours. The launcher starts the real command because its own parent's marker is valid."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=56147, process_identity=ID_A)
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=56148, process_identity=ID_B)
    proc = _run_launcher(plugin_dir, fresh_profile_home)
    assert proc.returncode == 0, proc.stderr
    assert "REAL-MCP-STARTED" in proc.stdout


@needs_identity
def test_real_launcher_accepts_a_long_running_parents_marker(plugin_dir, marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    _rewrite(path, registered_at=time.time() - 30 * 24 * 3600)
    proc = _run_launcher(plugin_dir, fresh_profile_home)
    assert proc.returncode == 0, proc.stderr
    assert "REAL-MCP-STARTED" in proc.stdout


@needs_identity
def test_real_launcher_blocks_when_only_other_processes_are_registered(plugin_dir, marker_mod, fresh_profile_home):
    """Valid markers for other pids -- including a genuinely live process -- never stand in."""
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent", pid=os.getppid())
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=56147, process_identity=ID_A)
    _assert_blocked(_run_launcher(plugin_dir, fresh_profile_home), f"registered.{os.getpid()}.json")


@needs_identity
def test_real_launcher_blocks_marker_inherited_from_earlier_pid_holder(plugin_dir, marker_mod, fresh_profile_home):
    """PID reuse, reproduced with real processes: a real (now exited) process registered; its
    marker is then placed under OUR pid -- as if the OS had handed its pid to us. Its genuine
    identity does not match ours, so the launcher hard-stops."""
    writer = subprocess.run(
        [sys.executable, "-c",
         "import os, sys; sys.path.insert(0, sys.argv[1]); import marker; "
         "print(marker.process_start_identity(os.getpid()))", str(plugin_dir)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    earlier_identity = writer.stdout.strip()
    assert marker_mod.is_valid_identity(earlier_identity)
    assert earlier_identity != marker_mod.process_start_identity(os.getpid())
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=os.getpid(), process_identity=earlier_identity)
    _assert_blocked(_run_launcher(plugin_dir, fresh_profile_home), "does not match")


@needs_identity
def test_real_launcher_blocks_on_malformed_identity(plugin_dir, marker_mod, fresh_profile_home):
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    _rewrite(path, process_identity="darwin:; touch /tmp/pwned")
    _assert_blocked(_run_launcher(plugin_dir, fresh_profile_home), "process_identity")


@needs_identity
def test_real_launcher_ignores_legacy_shared_marker(plugin_dir, marker_mod, fresh_profile_home):
    legacy = fresh_profile_home / marker_mod.MARKER_RELATIVE_DIR / "registered.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "plugin_id": "trading-consent", "profile": "trader", "registered_at": time.time(),
        "pid": os.getpid(), "process_identity": marker_mod.process_start_identity(os.getpid()),
    }), encoding="utf-8")
    _assert_blocked(_run_launcher(plugin_dir, fresh_profile_home))


@needs_identity
def test_real_launcher_blocks_same_second_earlier_pid_holder(plugin_dir, marker_mod, fresh_profile_home):
    """Pi R-001 end to end: the marker under OUR pid carries an identity from the same start second
    as us but a different sub-second value (macOS: microsecond; Linux: tick) -- i.e. a different
    process that held this pid within the same second. The real launcher must hard-stop."""
    ours = marker_mod.process_start_identity(os.getpid())
    marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent",
                            pid=os.getpid(), process_identity=_same_second_other_process(ours))
    _assert_blocked(_run_launcher(plugin_dir, fresh_profile_home), "does not match")


@needs_identity
def test_real_launcher_blocks_legacy_second_precision_identity(plugin_dir, marker_mod, fresh_profile_home):
    """A marker in the earlier whole-second format, even for the right second, is malformed now."""
    ours = marker_mod.process_start_identity(os.getpid())
    if not ours.startswith("darwin:"):
        pytest.skip("macOS identity format")
    path = marker_mod.write_marker(fresh_profile_home, profile="trader", plugin_id="trading-consent")
    _rewrite(path, process_identity=ours.split(".")[0])
    _assert_blocked(_run_launcher(plugin_dir, fresh_profile_home), "process_identity")
