"""pre_approval_request: fires a best-effort macOS notification, can never approve/veto, and a
notification failure must never propagate or affect anything else."""

from __future__ import annotations

import shutil
import subprocess

import pytest


def test_observer_returns_none_on_our_pattern_key(monkeypatch, trading_consent_pkg):
    calls = []
    monkeypatch.setattr(trading_consent_pkg, "_send_macos_notification",
                         lambda **kw: calls.append(kw))
    result = trading_consent_pkg._on_pre_approval_request(
        command="<yahoo_price>", description="[trader/market-data] yahoo_price — reads data",
        pattern_key="plugin_rule:trader:market-data:2026-06-15",
    )
    assert result is None  # observer hook -- cannot approve/veto
    assert len(calls) == 1


def test_ignores_prompts_that_are_not_ours(monkeypatch, trading_consent_pkg):
    """Must not spam a notification for every unrelated dangerous-command approval in the
    session -- only our own plugin_rule:<profile>:<category>:<date> escalations."""
    calls = []
    monkeypatch.setattr(trading_consent_pkg, "_send_macos_notification",
                         lambda **kw: calls.append(kw))
    for pattern_key in (
        "rm_rf_root", None, "plugin_rule:some_other_plugin:not-a-real-category:2026-06-15",
        "plugin_rule:trader:market-data",  # malformed -- missing date segment
    ):
        result = trading_consent_pkg._on_pre_approval_request(
            command="x", description="y", pattern_key=pattern_key,
        )
        assert result is None
    assert calls == []


def test_notification_subprocess_failure_does_not_raise(monkeypatch, trading_consent_pkg):
    def boom(*args, **kwargs):
        raise OSError("osascript not found")
    monkeypatch.setattr(subprocess, "run", boom)
    # Must not raise -- a failed/missing osascript must never affect the approval outcome.
    result = trading_consent_pkg._on_pre_approval_request(
        command="x", description="y", pattern_key="plugin_rule:trader:alerts:2026-06-15",
    )
    assert result is None


def test_notification_timeout_does_not_raise(monkeypatch, trading_consent_pkg):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=3)
    monkeypatch.setattr(subprocess, "run", boom)
    trading_consent_pkg._send_macos_notification(title="t", message="m")  # must not raise


def test_notification_uses_a_short_explicit_timeout(monkeypatch, trading_consent_pkg):
    """pre_approval_request is NOT wrapped by Hermes's bounded hook-callback timeout (it's an
    observer hook run synchronously) -- our own subprocess call must carry its own short bound
    so it can never stall the approval prompt it's announcing."""
    captured = {}

    def fake_run(args, timeout=None, **kwargs):
        captured["timeout"] = timeout
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    trading_consent_pkg._send_macos_notification(title="t", message="m")
    assert captured["timeout"] is not None and captured["timeout"] <= 5


class TestAppleScriptEscaping:
    """Item 7: Python's repr()/!r produces Python single-quoted syntax, which is not a valid
    AppleScript string literal at all -- osascript would fail to compile it, not merely
    mis-display it. These check the actual generated argv, and (where osacompile is available)
    real AppleScript syntax validity, without ever executing `display notification`."""

    def test_literal_uses_double_quotes_not_python_repr(self, trading_consent_pkg):
        out = trading_consent_pkg._applescript_string_literal("hello")
        assert out == '"hello"'
        assert not out.startswith("'")

    def test_literal_escapes_double_quotes_and_backslashes(self, trading_consent_pkg):
        out = trading_consent_pkg._applescript_string_literal('Say "hi" and a backslash \\ here')
        assert out == '"Say \\"hi\\" and a backslash \\\\ here"'

    def test_literal_flattens_newlines(self, trading_consent_pkg):
        out = trading_consent_pkg._applescript_string_literal("line one\nline two\r\nline three")
        assert "\n" not in out and "\r" not in out
        assert out == '"line one line two line three"'

    def test_generated_argv_uses_double_quoted_applescript_literals(self, monkeypatch, trading_consent_pkg):
        """Inspect the exact argv passed to subprocess.run -- must contain properly double-quoted
        AppleScript literals, never a bare Python repr with single quotes."""
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        trading_consent_pkg._send_macos_notification(
            title="Hermes trading approval needed",
            message='approve replay_trade for trader — "quoted" reason',
        )
        args = captured["args"]
        assert args[0] == "osascript"
        assert args[1] == "-e"
        script = args[2]
        assert script.startswith("display notification ")
        assert '\\"quoted\\"' in script  # escaped, not a raw unescaped quote breaking the literal
        message_part = script.split("with title")[0]
        assert message_part.count('"') >= 2  # double-quoted AppleScript literal, not Python repr
        assert not message_part.strip().endswith("'")  # no stray Python-repr single-quote artifact

    @pytest.mark.skipif(shutil.which("osacompile") is None, reason="osacompile not available on this host")
    @pytest.mark.parametrize("message", [
        "plain message",
        'has "double quotes" inside',
        "has a backslash \\ and \\\\ two",
        "has\nan embedded\nnewline",
        "mix: \"quote\", \\backslash\\, and\nnewline",
    ])
    def test_generated_script_is_valid_applescript_syntax(self, tmp_path, trading_consent_pkg, message):
        """Real syntax-compile check via osacompile -- compiles the script to a throwaway .scpt
        file WITHOUT ever running it (no notification is displayed by this test)."""
        script = (
            f"display notification {trading_consent_pkg._applescript_string_literal(message)} "
            f"with title {trading_consent_pkg._applescript_string_literal('Hermes trading approval needed')}"
        )
        out_path = tmp_path / "check.scpt"
        result = subprocess.run(
            ["osacompile", "-e", script, "-o", str(out_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f"generated AppleScript failed to compile: {result.stderr}\nscript was: {script!r}"
        )

    @pytest.mark.skipif(shutil.which("osacompile") is None, reason="osacompile not available on this host")
    def test_old_python_repr_style_script_fails_to_compile(self, tmp_path):
        """Regression anchor: proves the bug this fix addresses was real, not hypothetical --
        the previous `{message!r}` approach does not even parse as AppleScript."""
        message = 'Say "hi" and a backslash \\ here'
        title = "Hermes trading approval needed"
        broken_script = f"display notification {message!r} with title {title!r}"
        out_path = tmp_path / "broken.scpt"
        result = subprocess.run(
            ["osacompile", "-e", broken_script, "-o", str(out_path)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0


def test_is_our_pattern_key(trading_consent_pkg):
    fn = trading_consent_pkg._is_our_pattern_key
    assert fn("plugin_rule:trader:market-data:2026-06-15") is True
    assert fn("plugin_rule:tradingresearch:pine-write-build-save:2026-06-15") is True
    assert fn(None) is False
    assert fn("") is False
    assert fn("plugin_rule:trader:not-a-real-category:2026-06-15") is False
    assert fn("dangerous_command_pattern") is False
