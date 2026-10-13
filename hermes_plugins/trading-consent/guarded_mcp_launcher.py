#!/usr/bin/env python3
"""Guarded MCP launcher: hard-stops an MCP server's startup unless a profile-bound
trading-consent plugin-registration marker is present, fresh, AND written by the exact Hermes
process that is spawning this launcher right now: only ``registered.<os.getppid()>.json`` is
read, never any other process's marker -- see marker.py:check_marker.

Usage (this is what ``hermes mcp add``'s --command/--args should point at, not the underlying
node/docker command directly):

    guarded_mcp_launcher.py <real-command> [real-arg ...]

Example mcp_servers entries (illustrative, not written to any live config by this plugin):

    trader / tradingview-desktop:
        command: python3
        args: [guarded_mcp_launcher.py, node, /path/to/tradingview-mcp/src/server.js]
        env: {TRADING_CONSENT_PROFILE: trader}

    tradingresearch / tradingview-market-data:
        command: python3
        args: [guarded_mcp_launcher.py, docker, run, --rm, -i, --read-only, ..., tradingview-mcp-hermes:<tag>]
        env: {TRADING_CONSENT_PROFILE: tradingresearch}

Why this exists: without it, disabling/removing the trading-consent plugin from a profile's
`plugins.enabled` (accidentally or otherwise) leaves the underlying mcp_servers command entry
completely functional -- every trading tool becomes reachable with zero consent gate and no
error of any kind, because "no pre_tool_call hook registered" and "hook registered but never
prompts" are indistinguishable to Hermes's tool dispatcher. This launcher makes "the consent-gate
plugin is not currently active for this profile" a hard, loud MCP-startup failure instead.

No shell involved anywhere: the real command is exec'd via `os.execvp` with an explicit argv list,
never through a shell, so no argument is ever subject to shell metacharacter interpretation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marker import check_marker  # noqa: E402  (sys.path must be set first)

DEFAULT_PLUGIN_ID = "trading-consent"

# Environment variables this launcher reads. All non-secret identifiers -- a profile name and an
# optional filesystem path -- set via the MCP server's own `env:` config, never a credential.
ENV_PROFILE = "TRADING_CONSENT_PROFILE"
ENV_PROFILE_HOME = "TRADING_CONSENT_PROFILE_HOME"  # optional override for non-default HERMES_HOME
ENV_PLUGIN_ID = "TRADING_CONSENT_PLUGIN_ID"  # optional override, defaults to DEFAULT_PLUGIN_ID


def _default_profile_home(profile: str) -> Path:
    """``~/.hermes/profiles/<profile>`` -- the pattern confirmed for every named (non-default)
    Hermes profile on this host (hermes --profile <name> config path). Only used when
    TRADING_CONSENT_PROFILE_HOME is not set; an installation with a non-default HERMES_HOME must
    set that override explicitly rather than rely on this guess.
    """
    return Path.home() / ".hermes" / "profiles" / profile


def _resolve_profile_home(profile: str) -> Path:
    override = os.environ.get(ENV_PROFILE_HOME, "").strip()
    if override:
        return Path(override).expanduser()
    return _default_profile_home(profile)


def _fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"guarded_mcp_launcher: HARD STOP: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def main(argv: list[str]) -> "NoReturn":  # type: ignore[name-defined]
    """*argv* is this launcher's own arguments with the script name already stripped, i.e.
    ``[real_command, real_arg1, real_arg2, ...]`` (call as ``main(sys.argv[1:])``)."""
    if len(argv) < 1:
        _fail("no real command given -- usage: guarded_mcp_launcher.py <real-command> [args...]")

    profile = os.environ.get(ENV_PROFILE, "").strip()
    if not profile:
        _fail(f"{ENV_PROFILE} is not set -- cannot determine which profile's marker to check")

    plugin_id = os.environ.get(ENV_PLUGIN_ID, "").strip() or DEFAULT_PLUGIN_ID
    profile_home = _resolve_profile_home(profile)

    # The hard gate is the parent's own marker, not mere freshness/liveness: Hermes spawns this
    # launcher as a direct child, so os.getppid() here IS the pid of the Hermes process trying to
    # launch the MCP server right now, and only that process's per-pid marker file is read. A
    # marker written by any other process -- a concurrent CLI command, a prior Hermes run that
    # has since exited, one where the plugin was disabled across a restart -- is never consulted.
    result = check_marker(
        profile_home, parent_pid=os.getppid(), expected_profile=profile,
        expected_plugin_id=plugin_id,
    )
    if not result.ok:
        _fail(
            f"trading-consent plugin registration marker check failed for profile {profile!r} "
            f"at {profile_home}: {result.reason}. The underlying MCP server was NOT started. "
            "Ensure the trading-consent plugin is installed and enabled for this profile and "
            "that the Hermes process launching this server loaded it (which writes that "
            f"process's own marker), then retry. ({ENV_PROFILE_HOME} can override the profile-home guess if this "
            "installation uses a non-default HERMES_HOME.)"
        )

    real_command = argv[0]
    resolved = real_command if os.sep in real_command else _which(real_command)
    if resolved is None:
        _fail(f"real command {real_command!r} not found on PATH")

    # execvp replaces this process image in place -- no shell, no extra process, stdio (the MCP
    # stdio transport) is inherited unchanged. argv (not argv[1:]) is passed as the child's own
    # argv, per execvp's convention that args[0] is the program name the child sees.
    os.execvp(resolved, argv)  # noqa: S606 -- intentional, no shell involved


def _which(command: str) -> str | None:
    import shutil
    return shutil.which(command)


if __name__ == "__main__":
    main(sys.argv[1:])
