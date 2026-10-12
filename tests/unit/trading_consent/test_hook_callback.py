"""_on_pre_tool_call: None for non-trading built-ins, directive shape for approve/block, and the
top-level catch-all that closes Hermes's one real fail-open gap (an uncaught exception here is
otherwise indistinguishable from "no plugin has an opinion" -- see __init__.py's module
docstring)."""

from __future__ import annotations

import pytest


def trader_tool(tool: str) -> str:
    return f"mcp__tradingview_desktop__{tool}"


def research_tool(tool: str) -> str:
    return f"mcp__tradingview_market_data__{tool}"


@pytest.fixture(autouse=True)
def _patch_profile(monkeypatch, trading_consent_pkg):
    """Most tests want a deterministic profile name rather than depending on whatever
    hermes_constants resolves to in this test process (which may not even be importable)."""
    monkeypatch.setattr(trading_consent_pkg, "_current_profile_name", lambda: "trader")


class TestNonTradingBuiltins:
    """Item 3: pre_tool_call must return None -- not approve, not block -- for a built-in tool
    that isn't MCP-namespaced at all."""

    @pytest.mark.parametrize("builtin", ["clarify", "vision", "web", "memory", "todo", "delegation"])
    def test_returns_none_for_builtins(self, trading_consent_pkg, builtin):
        assert trading_consent_pkg._on_pre_tool_call(builtin, {}) is None

    def test_returns_none_even_when_profile_lookup_would_fail(self, monkeypatch, trading_consent_pkg):
        """A built-in must be ignored regardless of whether the active profile can even be
        determined -- profile resolution must not run (or matter) before the ignore check."""
        def boom():
            raise RuntimeError("profile lookup should never be reached for a built-in")
        monkeypatch.setattr(trading_consent_pkg, "_current_profile_name", boom)
        assert trading_consent_pkg._on_pre_tool_call("clarify", {}) is None


def test_approve_directive_for_in_scope_tool(trading_consent_pkg):
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("replay_trade"), {})
    assert result["action"] == "approve"
    assert result["rule_key"].startswith("trader:replay-paper:")
    assert "message" in result and result["message"]


def test_block_directive_for_hard_excluded_tool(trading_consent_pkg):
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("ui_evaluate"), {"code": "whatever"})
    assert result["action"] == "block"
    assert result["message"]  # non-empty -- an empty message fails open in real Hermes
    assert "rule_key" not in result


def test_block_directive_for_unknown_tool_on_expected_server(trading_consent_pkg):
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("definitely_not_reviewed"), {})
    assert result["action"] == "block"
    assert result["message"]


def test_block_directive_for_wrong_server(trading_consent_pkg):
    """trader's profile only accepts tradingview_desktop -- a call namespaced for the other
    profile's server must block, not be silently ignored."""
    result = trading_consent_pkg._on_pre_tool_call(research_tool("yahoo_price"), {})
    assert result["action"] == "block"
    assert "server" in result["message"].lower()


def test_batch_run_malformed_action_blocks(trading_consent_pkg):
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("batch_run"), {"action": "place_order"})
    assert result["action"] == "block"


def test_batch_run_valid_action_approves_as_chart_change(trading_consent_pkg):
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("batch_run"), {"action": "get_ohlcv"})
    assert result["action"] == "approve"
    assert ":chart-change:" in result["rule_key"]


def test_tv_launch_approves_as_chart_change(trading_consent_pkg):
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("tv_launch"), {})
    assert result["action"] == "approve"
    assert ":chart-change:" in result["rule_key"]


def test_watchlist_add_bulk_blocks_as_unreviewed(trading_consent_pkg):
    """Phantom pre-authorization regression guard (item 5)."""
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("watchlist_add_bulk"), {})
    assert result["action"] == "block"


def test_approval_message_states_local_date_and_always_scope(trading_consent_pkg):
    """Item 8: the message must visibly say the date, and that Always means this
    category/profile for that date -- not forever."""
    import datetime as dt
    result = trading_consent_pkg._on_pre_tool_call(trader_tool("replay_trade"), {})
    today = dt.date.today().isoformat()
    assert today in result["message"]
    assert "Always" in result["message"]
    assert "trader" in result["message"] and "replay-paper" in result["message"]


def test_missing_args_does_not_raise(trading_consent_pkg):
    """args can legitimately be an empty dict or None depending on caller; must never crash.
    research_tool() is used here even though the autouse fixture pins profile to "trader" --
    that's deliberate: it must still resolve to a clean block (wrong server), not a crash.
    """
    result = trading_consent_pkg._on_pre_tool_call(research_tool("yahoo_price"), None)
    assert result["action"] in ("approve", "block")


class TestCatchAllOnInternalError:
    """These simulate a bug in OUR OWN classification code (not in Hermes) -- exactly the
    scenario that fails OPEN if it escapes this callback uncaught. Every one must still return
    an explicit, non-empty block directive."""

    def test_classify_raising_still_blocks(self, monkeypatch, trading_consent_pkg):
        def boom(tool_name, args, **kwargs):
            raise RuntimeError("simulated bug in categories.classify")
        monkeypatch.setattr(trading_consent_pkg._categories, "classify", boom)
        result = trading_consent_pkg._on_pre_tool_call(trader_tool("yahoo_price"), {})
        assert result["action"] == "block"
        assert result["message"]

    def test_rule_key_construction_raising_still_blocks(self, monkeypatch, trading_consent_pkg):
        def boom(profile, category, **kwargs):
            raise RuntimeError("simulated bug in rule_key.build_rule_key")
        monkeypatch.setattr(trading_consent_pkg._rule_key, "build_rule_key", boom)
        result = trading_consent_pkg._on_pre_tool_call(trader_tool("yahoo_price"), {})
        assert result["action"] == "block"
        assert result["message"]

    def test_profile_lookup_raising_still_blocks_for_an_mcp_tool(self, monkeypatch, trading_consent_pkg):
        def boom():
            raise RuntimeError("simulated bug resolving the current profile")
        monkeypatch.setattr(trading_consent_pkg, "_current_profile_name", boom)
        result = trading_consent_pkg._on_pre_tool_call(trader_tool("yahoo_price"), {})
        assert result["action"] == "block"
        assert result["message"]

    def test_unresolvable_profile_blocks_rather_than_approving_with_a_degraded_key(
        self, monkeypatch, trading_consent_pkg,
    ):
        monkeypatch.setattr(trading_consent_pkg, "_current_profile_name", lambda: None)
        result = trading_consent_pkg._on_pre_tool_call(trader_tool("yahoo_price"), {})
        assert result["action"] == "block"

    def test_completely_malformed_tool_name_input_blocks_not_raises(self, trading_consent_pkg):
        # A non-string tool_name isn't a valid mcp__ form either -- parse_mcp_tool_name handles
        # it by returning None, so this actually returns None (ignored), not a block. What must
        # never happen is an exception; assert that instead of a specific outcome shape.
        result = trading_consent_pkg._on_pre_tool_call(None, {})  # type: ignore[arg-type]
        assert result is None or result.get("action") == "block"
