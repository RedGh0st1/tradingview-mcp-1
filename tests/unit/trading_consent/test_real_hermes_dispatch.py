"""Real Hermes loader/dispatcher tests.

These import the ACTUAL installed hermes-agent (~/.hermes/hermes-agent) and register our real
plugin.register(ctx) against a freshly constructed, isolated PluginManager -- never the global
singleton returned by hermes_cli.plugins.get_plugin_manager(), and never any real profile's
config -- so this exercises Hermes's genuine pre_tool_call dispatch mechanics (per-callback
exception isolation, multi-callback result aggregation) without touching this machine's live
Hermes install, its plugins, or any profile.

Skipped entirely (not failed) when hermes-agent isn't importable, e.g. a CI machine or a
checkout of this repo without a local Hermes install -- this plugin's correctness does not
depend on this test running everywhere, only on it running where it CAN and catching a real
contract mismatch when it does.
"""

from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import pytest

HERMES_AGENT_DIR = Path.home() / ".hermes" / "hermes-agent"


def _try_import_hermes_plugins():
    """Import the real hermes_cli.plugins module, or return the ImportError that stopped it.

    hermes-agent is a separate, much larger application with its own dependency set (PyYAML and
    others) that this repository's own venv has no reason to install. Skipping on ANY import
    failure -- not just a missing directory -- keeps this test honest about "practical": it runs
    wherever a full local Hermes install's own dependencies are available, and skips (never
    errors) everywhere else.
    """
    if not (HERMES_AGENT_DIR / "hermes_cli" / "plugins.py").is_file():
        return None, f"no local Hermes agent install found at {HERMES_AGENT_DIR}"
    if str(HERMES_AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT_DIR))
    try:
        import hermes_cli.plugins as plugins_mod
    except Exception as exc:  # ModuleNotFoundError for a missing hermes-agent dependency, etc.
        return None, f"hermes_cli.plugins not importable in this environment: {exc}"
    return plugins_mod, None


_PLUGINS_MODULE, _SKIP_REASON = _try_import_hermes_plugins()

pytestmark = pytest.mark.skipif(_PLUGINS_MODULE is None, reason=_SKIP_REASON or "unavailable")


@pytest.fixture()
def hermes_plugins_module():
    return _PLUGINS_MODULE


@pytest.fixture()
def isolated_manager(hermes_plugins_module, tmp_path, monkeypatch):
    """A brand-new PluginManager scoped to a random test key -- never the process-wide default
    singleton, so this can never discover or interact with any real installed plugin or profile.

    Constructing a PluginManager(scope_key=...) turns out to have a real filesystem side effect:
    it bootstraps a profile-shaped directory skeleton (SOUL.md, cron/, hooks/, sessions/, ...)
    relative to the current working directory (observed directly -- a first run of this suite
    left pytest-trading-consent-<uuid>/ directories in the repo root, cleaned up before this fix
    landed). chdir into a disposable tmp_path first so that side effect can never touch this
    repository -- or, if the resolution logic ever changes, anything outside a throwaway dir.
    """
    monkeypatch.chdir(tmp_path)
    return hermes_plugins_module.PluginManager(scope_key=f"pytest-trading-consent-{uuid.uuid4()}")


@pytest.fixture()
def plugin_context(hermes_plugins_module, isolated_manager):
    # manifest_key() (hermes_cli/plugins_manifest.py) reads manifest.key before falling back to
    # .name -- a bare SimpleNamespace without `key` raises AttributeError there, not a falsy
    # lookup, so it must be present (as None) even though we want name-based keying.
    manifest = types.SimpleNamespace(name="trading-consent", key=None)
    return hermes_plugins_module.PluginContext(manifest, isolated_manager)


@pytest.fixture()
def pinned_trader_profile(monkeypatch, trading_consent_pkg):
    """Without this, _current_profile_name() would resolve via the REAL hermes_constants to
    whatever profile this actual machine's default HERMES_HOME is -- not a profile this plugin
    governs, and not something a test should depend on. Pin it explicitly."""
    monkeypatch.setattr(trading_consent_pkg, "_current_profile_name", lambda: "trader")


@pytest.fixture()
def isolated_marker_home(tmp_path):
    """A throwaway profile-home for marker writes in these tests -- never ~/.hermes."""
    home = tmp_path / "fake-profile-home"
    home.mkdir()
    return home


@pytest.fixture()
def pinned_marker_home(monkeypatch, isolated_marker_home):
    """_write_registration_marker() does `from hermes_constants import get_hermes_home` inside
    the function body, re-resolving the module attribute on every call -- so patching
    hermes_constants.get_hermes_home directly is picked up without needing to patch our own
    module. Ensures no test in this file can ever write a marker under the real ~/.hermes.
    """
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: isolated_marker_home)
    return isolated_marker_home


def test_isolated_manager_starts_with_no_hooks(isolated_manager):
    """Guards the isolation assumption itself: if this ever fails, the manager is not as fresh
    as assumed and the rest of this file's isolation claims would be false."""
    assert isolated_manager._hooks == {}


def test_our_register_wires_into_the_real_plugin_context(
    trading_consent_pkg, plugin_context, isolated_manager, pinned_marker_home,
):
    trading_consent_pkg.register(plugin_context)
    assert "pre_tool_call" in isolated_manager._hooks
    assert "pre_approval_request" in isolated_manager._hooks
    assert trading_consent_pkg._on_pre_tool_call in isolated_manager._hooks["pre_tool_call"]


def test_real_invoke_hook_returns_our_directive(
    trading_consent_pkg, plugin_context, isolated_manager, pinned_trader_profile, pinned_marker_home,
):
    trading_consent_pkg.register(plugin_context)
    results = isolated_manager.invoke_hook(
        "pre_tool_call", tool_name="mcp__tradingview_desktop__replay_trade", args={},
    )
    assert len(results) == 1
    assert results[0]["action"] == "approve"
    assert results[0]["rule_key"]


def test_real_invoke_hook_returns_none_for_a_builtin(
    trading_consent_pkg, plugin_context, isolated_manager, pinned_trader_profile, pinned_marker_home,
):
    """Item 3, exercised against Hermes's real invoke_hook: a non-MCP built-in (clarify) must
    contribute nothing at all -- invoke_hook only appends non-None results."""
    trading_consent_pkg.register(plugin_context)
    results = isolated_manager.invoke_hook("pre_tool_call", tool_name="clarify", args={})
    assert results == []


def test_real_invoke_hook_blocks_hard_excluded_tool(
    trading_consent_pkg, plugin_context, isolated_manager, pinned_trader_profile, pinned_marker_home,
):
    trading_consent_pkg.register(plugin_context)
    results = isolated_manager.invoke_hook(
        "pre_tool_call", tool_name="mcp__tradingview_desktop__ui_evaluate", args={},
    )
    assert len(results) == 1
    assert results[0]["action"] == "block"
    assert results[0]["message"]


def test_real_invoke_hook_blocks_wrong_server(
    trading_consent_pkg, plugin_context, isolated_manager, pinned_trader_profile, pinned_marker_home,
):
    """Item 4, against the real dispatcher: trader only accepts tradingview_desktop."""
    trading_consent_pkg.register(plugin_context)
    results = isolated_manager.invoke_hook(
        "pre_tool_call", tool_name="mcp__tradingview_market_data__yahoo_price", args={},
    )
    assert len(results) == 1
    assert results[0]["action"] == "block"


def test_real_dispatcher_isolates_a_crashing_sibling_callback(
    hermes_plugins_module, trading_consent_pkg, plugin_context, isolated_manager,
    pinned_trader_profile, pinned_marker_home,
):
    """Confirms, against Hermes's REAL invoke_hook (not our reading of its source), that one
    callback raising does not prevent another registered callback's valid directive from being
    returned -- and that the raising callback itself contributes nothing (no synthesized block),
    which is precisely the fail-open gap our own callback's top-level catch-all exists to avoid
    for ITS OWN code path (see __init__.py). This test documents Hermes's real behavior; it does
    not by itself prove our callback is safe -- test_hook_callback.py's TestCatchAllOnInternalError
    proves that separately.
    """
    def crashing_sibling(tool_name, args, **kwargs):
        raise RuntimeError("a different, broken plugin")

    isolated_manager.register_hook("pre_tool_call", crashing_sibling) if hasattr(
        isolated_manager, "register_hook"
    ) else plugin_context._track_callback(
        "hook", "pre_tool_call", crashing_sibling, isolated_manager._hooks,
        getattr(hermes_plugins_module, "VALID_HOOKS", {"pre_tool_call"}),
    )
    trading_consent_pkg.register(plugin_context)

    results = isolated_manager.invoke_hook(
        "pre_tool_call", tool_name="mcp__tradingview_desktop__replay_trade", args={},
    )
    # Our callback's real directive is present; the crashing sibling contributed nothing (no
    # exception propagated out of invoke_hook, no synthesized entry for it either).
    actions = [r["action"] for r in results]
    assert "approve" in actions
    assert len(results) == 1  # only our callback's result -- the crash produced no entry at all


def test_real_get_pre_tool_call_directive_details_resolves_approve(
    hermes_plugins_module, trading_consent_pkg, plugin_context, monkeypatch,
    pinned_trader_profile, pinned_marker_home,
):
    """Exercises hermes_cli.plugins._get_pre_tool_call_directive_details -- the actual function
    model_tools.py calls on every tool dispatch -- against our registered hook, redirected (via
    monkeypatch, auto-reverted) from the global singleton to our isolated manager for the
    duration of this test only.
    """
    isolated_manager = plugin_context._manager
    trading_consent_pkg.register(plugin_context)
    monkeypatch.setattr(hermes_plugins_module, "get_plugin_manager", lambda: isolated_manager)

    # invoke_lifecycle_hook resolves to hermes_cli.plugins.invoke_hook, which (per source) calls
    # get_plugin_manager().invoke_hook(...) -- patched above to our isolated instance.
    details = hermes_plugins_module._get_pre_tool_call_directive_details(
        "mcp__tradingview_desktop__replay_trade", {},
    )
    assert details.action == "approve"
    assert details.rule_key is not None


def test_real_get_pre_tool_call_directive_details_resolves_block(
    hermes_plugins_module, trading_consent_pkg, plugin_context, monkeypatch,
    pinned_trader_profile, pinned_marker_home,
):
    isolated_manager = plugin_context._manager
    trading_consent_pkg.register(plugin_context)
    monkeypatch.setattr(hermes_plugins_module, "get_plugin_manager", lambda: isolated_manager)

    details = hermes_plugins_module._get_pre_tool_call_directive_details(
        "mcp__tradingview_desktop__tv_update", {},
    )
    assert details.action == "block"
    assert details.message


def test_real_get_pre_tool_call_directive_details_none_for_builtin(
    hermes_plugins_module, trading_consent_pkg, plugin_context, monkeypatch,
    pinned_trader_profile, pinned_marker_home,
):
    isolated_manager = plugin_context._manager
    trading_consent_pkg.register(plugin_context)
    monkeypatch.setattr(hermes_plugins_module, "get_plugin_manager", lambda: isolated_manager)

    details = hermes_plugins_module._get_pre_tool_call_directive_details("clarify", {})
    assert details.action is None


class TestRegistrationOrderingAgainstRealPluginContext:
    """Item 1: hooks must register successfully BEFORE the marker is written -- a marker must
    never exist for a partial registration. Exercised against the REAL PluginContext.register_hook
    (hermes_cli/plugins.py), not a stand-in, so a future signature/behavior change there is
    caught here too.
    """

    def test_successful_registration_writes_the_marker(
        self, trading_consent_pkg, plugin_context, pinned_marker_home,
    ):
        trading_consent_pkg.register(plugin_context)
        path = trading_consent_pkg._marker.marker_path(pinned_marker_home)
        assert path.is_file()

    def test_second_hook_registration_failing_leaves_no_marker(
        self, hermes_plugins_module, trading_consent_pkg, plugin_context, monkeypatch, pinned_marker_home,
    ):
        """Simulates the real PluginContext.register_hook itself failing on the SECOND call
        (pre_approval_request) -- proves the marker line is genuinely unreachable when
        registration only partially succeeds, using Hermes's actual register_hook method.
        """
        real_register_hook = hermes_plugins_module.PluginContext.register_hook
        call_count = {"n": 0}

        def flaky_register_hook(self, hook_name, callback):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise RuntimeError("simulated Hermes-side registration failure on the 2nd hook")
            return real_register_hook(self, hook_name, callback)

        monkeypatch.setattr(hermes_plugins_module.PluginContext, "register_hook", flaky_register_hook)

        with pytest.raises(RuntimeError, match="simulated"):
            trading_consent_pkg.register(plugin_context)

        path = trading_consent_pkg._marker.marker_path(pinned_marker_home)
        assert not path.is_file(), "marker must not exist when registration only partially succeeded"

    def test_first_hook_registration_failing_leaves_no_marker(
        self, hermes_plugins_module, trading_consent_pkg, plugin_context, monkeypatch, pinned_marker_home,
    ):
        def always_fails(self, hook_name, callback):
            raise RuntimeError("simulated Hermes-side registration failure on the 1st hook")

        monkeypatch.setattr(hermes_plugins_module.PluginContext, "register_hook", always_fails)

        with pytest.raises(RuntimeError, match="simulated"):
            trading_consent_pkg.register(plugin_context)

        path = trading_consent_pkg._marker.marker_path(pinned_marker_home)
        assert not path.is_file()
