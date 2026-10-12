"""trading-consent -- daily human-approval gate for the trader/tradingresearch Hermes profiles.

Registers two hooks (see plugin.yaml):
  * pre_tool_call        -- returns Python None for anything that isn't an MCP-namespaced
                             trading tool call (built-ins like clarify/vision -- no opinion);
                             for an MCP-namespaced call, requires the server to be the ONE
                             reviewed server for the active profile (see categories.py:
                             PROFILE_SERVERS), classifies the bare tool name into one of seven
                             categories, and escalates to Hermes's existing human-approval gate
                             via tools.approval.request_tool_approval with a date-scoped
                             rule_key so an "always" choice expires at local midnight. See
                             categories.py / rule_key.py for the classification and key logic
                             this module wires together; both are dependency-free so they can be
                             unit tested without importing Hermes at all.
  * pre_approval_request -- observer-only (cannot approve/veto): fires a best-effort macOS
                             notification alongside Hermes's own CLI/gateway approval prompt.

Fail-open note (see architecture review): Hermes's plugin dispatcher (hermes_cli/plugins.py,
hermes_cli/plugins_dispatch.py) fails CLOSED on a pre_tool_call timeout or on any error from the
approval gate itself, but fails OPEN if this callback raises before returning any directive at
all -- an uncaught exception is caught by the dispatcher, logged, and simply excluded from the
hook results, which is indistinguishable from "no plugin has an opinion" and lets the tool call
through. _on_pre_tool_call's top-level try/except exists specifically to make that impossible:
every code path below returns an explicit block/approve directive (or, deliberately, Python
None for a non-trading built-in) -- nothing here lets an exception escape.

Registration order matters (see register() below): both hooks are registered BEFORE the
profile-bound marker is written, so a marker can never exist for a partially-failed registration
-- if either register_hook call raises, the marker line is never reached, and
guarded_mcp_launcher.py correctly hard-stops the MCP server on the next launch because no valid
marker exists.
"""

from __future__ import annotations

import datetime as _dt
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from . import categories as _categories
from . import marker as _marker
from . import rule_key as _rule_key

logger = logging.getLogger("hermes_plugin.trading_consent")

PLUGIN_ID = "trading-consent"

_EFFECT_DESCRIPTIONS: dict[str, str] = {
    _categories.MARKET_DATA: "reads market/quote/chart data",
    _categories.PINE_READ_OPEN: "reads or opens Pine script content",
    _categories.PINE_WRITE_BUILD_SAVE: "writes, compiles, or saves a Pine script",
    _categories.BACKTESTING: "runs or reads a backtest / strategy-tester result",
    _categories.CHART_CHANGE: "changes chart, indicator, watchlist, layout, or drawing state",
    _categories.ALERTS: "creates or lists a price alert",
    _categories.REPLAY_PAPER: "runs a paper/replay-mode trading action (not live)",
}

_NOTIFICATION_TIMEOUT_SECONDS = 3


def _current_profile_name() -> Optional[str]:
    """Best-effort profile name for the running Hermes process, derived from its resolved home
    directory (``.../profiles/<name>``). Returns None (never raises) when it cannot be
    determined -- callers must treat that as "cannot mark/approve", not "assume default"."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()).resolve().name
    except Exception:
        logger.warning("trading-consent: could not determine current profile name", exc_info=True)
        return None


def _write_registration_marker() -> None:
    """Best-effort marker refresh at plugin-load time. Only ever called AFTER both hooks have
    registered successfully (see register() below) -- never on a partial/failed registration.
    Never raises itself: a marker-write failure must not prevent the plugin from otherwise having
    registered its (fail-closed) hooks -- it only means the guarded launcher's own separate check
    may then correctly hard-stop, which is the safe outcome.
    """
    try:
        from hermes_constants import get_hermes_home
        profile_home = Path(get_hermes_home()).resolve()
        profile = profile_home.name
        _marker.write_marker(profile_home, profile=profile, plugin_id=PLUGIN_ID)
    except Exception:
        logger.warning("trading-consent: failed to write registration marker", exc_info=True)


def _is_our_pattern_key(pattern_key: Optional[str]) -> bool:
    """True when *pattern_key* is one of THIS plugin's own escalations (``plugin_rule:<profile>:
    <category>:<date>``), so pre_approval_request only fires a trading-consent notification for
    trading-consent prompts -- not every unrelated dangerous-command approval in the session."""
    if not isinstance(pattern_key, str) or not pattern_key.startswith("plugin_rule:"):
        return False
    rest = pattern_key[len("plugin_rule:"):]
    parts = rest.split(":")
    return len(parts) == 3 and parts[1] in _categories.ALL_CATEGORIES


def _applescript_string_literal(value: Any) -> str:
    """A valid AppleScript double-quoted string literal for *value*.

    Python's repr()/!r produces Python single-quoted syntax, which AppleScript's parser does not
    understand as a string literal at all -- osascript would fail (or, worse, parse the leftover
    text as something unintended) rather than display the notification. AppleScript string
    literals are double-quoted with backslash-escaped backslashes and double quotes; they also
    cannot contain a raw newline, so any line breaks are flattened to spaces first.
    """
    text = str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return f'"{text}"'


def _send_macos_notification(*, title: str, message: str) -> None:
    """Best-effort only. pre_approval_request is NOT wrapped by Hermes's bounded hook-callback
    timeout (it's an observer hook, run synchronously to completion), so this call carries its
    OWN short timeout to guarantee it can never stall the human approval prompt it's announcing.
    Any failure (osascript missing, not on macOS, timeout, a malformed script) is swallowed here
    -- notification failure must never affect the approval outcome, only whether a banner
    appeared.
    """
    script = (
        f"display notification {_applescript_string_literal(message)} "
        f"with title {_applescript_string_literal(title)}"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            timeout=_NOTIFICATION_TIMEOUT_SECONDS,
            check=False,
            capture_output=True,
        )
    except Exception:
        logger.debug("trading-consent: macOS notification failed (non-fatal)", exc_info=True)


def _on_pre_tool_call(tool_name: str, args: dict, **_kwargs: Any) -> Optional[dict]:
    """The one hook that can gate a tool call. See module docstring for the fail-open rationale
    behind this function's own try/except -- every return statement produces an explicit
    Hermes-shaped directive, or Python None for a deliberate non-trading built-in; nothing here
    is allowed to propagate an exception upward.
    """
    try:
        # Not MCP-namespaced at all (clarify, vision, and every other built-in toolset tool) --
        # this plugin has no opinion. This check must run BEFORE resolving the active profile,
        # so a profile-lookup failure never turns an unrelated built-in into a false block.
        if _categories.parse_mcp_tool_name(tool_name) is None:
            return None

        profile = _current_profile_name()
        if not profile:
            # An MCP-namespaced call reached us, but we cannot determine which profile's single
            # reviewed server to check it against -- block rather than approve or ignore.
            return {
                "action": "block",
                "message": _rule_key.block_message(
                    tool_name=str(tool_name),
                    reason="could not determine the active Hermes profile -- blocking by design",
                ),
            }

        classification = _categories.classify(tool_name, args, profile=profile)
        bare_name = classification.bare_name or _categories.normalize_tool_name(tool_name)

        if classification.is_blocked:
            return {
                "action": "block",
                "message": _rule_key.block_message(
                    tool_name=bare_name, reason=classification.block_reason or "blocked"
                ),
            }

        category = classification.category
        assert category is not None  # not is_blocked and not ignore, so classify() guarantees this

        now = _dt.datetime.now()
        message = _rule_key.redact_message(
            profile=profile, category=category, tool_name=bare_name,
            effect=_EFFECT_DESCRIPTIONS.get(category, "requires human approval"), now=now,
        )
        key = _rule_key.build_rule_key(profile, category, now=now)
        return _rule_key.ApprovalRequest(action="approve", message=message, rule_key=key).as_directive()
    except Exception as exc:  # noqa: BLE001 -- intentional: see module docstring
        logger.warning("trading-consent: pre_tool_call classification error for %r: %s", tool_name, exc, exc_info=True)
        return {
            "action": "block",
            "message": (
                f"BLOCKED by trading-consent: internal classification error for "
                f"{tool_name!r} -- blocking by design rather than allowing an unclassified "
                "trading tool call through."
            ),
        }


def _on_pre_approval_request(
    command: Optional[str] = None, description: Optional[str] = None,
    pattern_key: Optional[str] = None, session_key: Optional[str] = None,
    surface: Optional[str] = None, **_kwargs: Any,
) -> None:
    """Observer only -- cannot approve or veto (Hermes discards this hook's return value and
    never lets it raise; see tools/approval_context.py:_fire_approval_hook). Its only job is the
    best-effort macOS notification."""
    if not _is_our_pattern_key(pattern_key):
        return None
    try:
        title = "Hermes trading approval needed"
        body = description or command or "A trading action requires your approval."
        _send_macos_notification(title=title, message=body)
    except Exception:
        logger.debug("trading-consent: pre_approval_request notification error (non-fatal)", exc_info=True)
    return None


def register(ctx) -> None:
    """Plugin entry point (Hermes's own convention: def register(ctx) -> None, ctx.register_hook).

    Hooks are registered FIRST; the profile-bound marker is written LAST, and only if both
    register_hook calls succeeded. If either raises, this function propagates the exception and
    the marker is never written -- guarded_mcp_launcher.py then correctly hard-stops on the next
    MCP launch, because no valid marker exists for a plugin that only partially registered.
    """
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("pre_approval_request", _on_pre_approval_request)
    _write_registration_marker()
