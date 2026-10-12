"""Tool -> category classification: mapping completeness, hard excludes, unknown/prefix
variants, profile/server binding, and batch_run action validation."""

from __future__ import annotations

import pytest

TRADER = "trader"
RESEARCH = "tradingresearch"


def mcp(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


def trader_tool(tool: str) -> str:
    return mcp("tradingview_desktop", tool)


def research_tool(tool: str) -> str:
    return mcp("tradingview_market_data", tool)


# Exact tool names read directly from the two reviewed MCP servers' own source during the
# architecture review (not the abstract category names) -- this is the ground truth the
# completeness test below is checked against. watchlist_add_bulk, watchlist_remove, and
# indicator_search are deliberately excluded: they exist in the underlying server's full tool
# surface but are absent from trader's live, reviewed 45-tool allowlist.
TRADINGRESEARCH_TOOLS = {
    "yahoo_price", "market_snapshot", "coin_analysis", "financial_news",
    "backtest_strategy", "compare_strategies",
}

TRADER_ALLOWLISTED_TOOLS = {
    "tv_health_check", "tv_discover", "tv_ui_state", "tv_launch",
    "chart_get_state", "chart_set_symbol", "chart_set_timeframe", "chart_set_type",
    "chart_manage_indicator", "chart_get_visible_range", "chart_set_visible_range",
    "chart_scroll_to_date", "symbol_info", "symbol_search",
    "data_get_ohlcv", "data_get_indicator", "data_get_strategy_results", "data_get_trades",
    "data_get_equity", "quote_get", "depth_get", "data_get_study_values",
    "capture_screenshot", "draw_shape", "draw_list", "draw_get_properties",
    "alert_create", "alert_list", "batch_run",
    "replay_start", "replay_step", "replay_autoplay", "replay_stop", "replay_trade", "replay_status",
    "indicator_set_inputs", "indicator_toggle_visibility",
    "watchlist_get", "watchlist_add",
    "layout_list", "layout_switch",
    "pane_list", "pane_set_layout", "pane_focus", "pane_set_symbol",
}

# Pine tools required by this task ("Include exact Pine tools for authoring/compile/save/
# analyze/check") -- not yet in trader's live allowlist, added here as the tools this plugin
# must classify once the allowlist is expanded to include them.
PINE_TOOLS = {
    "pine_get_source", "pine_get_errors", "pine_get_console", "pine_list_scripts",
    "pine_analyze", "pine_check",
    "pine_set_source", "pine_compile", "pine_save", "pine_smart_compile", "pine_new", "pine_open",
}

PHANTOM_TOOLS_NEVER_CLASSIFIED = {"watchlist_add_bulk", "watchlist_remove", "indicator_search"}

HARD_EXCLUDED_EVEN_IF_PRESENT = {
    "ui_click", "ui_mouse_click", "ui_keyboard", "ui_type_text", "ui_hover", "ui_scroll",
    "ui_find_element", "ui_evaluate", "ui_open_panel",
    "tv_update", "draw_clear", "draw_remove_one", "alert_delete",
}


# --- Non-MCP built-ins: must be ignored, never blocked -----------------------------------------

@pytest.mark.parametrize("builtin", ["clarify", "vision", "web", "memory", "todo"])
def test_non_mcp_builtins_are_ignored(categories_mod, builtin):
    result = categories_mod.classify(builtin, {}, profile=TRADER)
    assert result.ignore is True
    assert result.category is None
    assert result.block_reason is None


def test_parse_mcp_tool_name_returns_none_for_bare_names(categories_mod):
    assert categories_mod.parse_mcp_tool_name("clarify") is None
    assert categories_mod.parse_mcp_tool_name("vision") is None
    assert categories_mod.parse_mcp_tool_name("") is None
    assert categories_mod.parse_mcp_tool_name(None) is None


def test_parse_mcp_tool_name_parses_real_wire_form(categories_mod):
    assert categories_mod.parse_mcp_tool_name("mcp__tradingview_desktop__replay_trade") == (
        "tradingview_desktop", "replay_trade",
    )


# --- Mapping completeness ------------------------------------------------------------------

def test_every_reviewed_research_tool_classifies_without_error(categories_mod):
    for tool in sorted(TRADINGRESEARCH_TOOLS):
        result = categories_mod.classify(research_tool(tool), profile=RESEARCH)
        assert not result.is_blocked, f"{tool} unexpectedly blocked: {result.block_reason}"
        assert not result.ignore
        assert result.category in categories_mod.ALL_CATEGORIES


def test_every_reviewed_trader_tool_classifies_without_error(categories_mod):
    reviewed = (TRADER_ALLOWLISTED_TOOLS | PINE_TOOLS) - {"batch_run"}
    for tool in sorted(reviewed):
        result = categories_mod.classify(trader_tool(tool), profile=TRADER)
        assert not result.is_blocked, f"{tool} unexpectedly blocked: {result.block_reason}"
        assert not result.ignore
        assert result.category in categories_mod.ALL_CATEGORIES


def test_tv_launch_is_chart_change(categories_mod):
    """tv_launch starts/attaches TradingView Desktop -- a real state change, not a passive read."""
    result = categories_mod.classify(trader_tool("tv_launch"), profile=TRADER)
    assert not result.is_blocked
    assert result.category == categories_mod.CHART_CHANGE


@pytest.mark.parametrize("phantom", sorted(PHANTOM_TOOLS_NEVER_CLASSIFIED))
def test_phantom_tools_absent_from_live_allowlist_block_as_unknown(categories_mod, phantom):
    """watchlist_add_bulk/watchlist_remove/indicator_search exist in the server's full tool
    surface but were never reviewed/allowlisted for trader -- must not be pre-authorized."""
    result = categories_mod.classify(trader_tool(phantom), profile=TRADER)
    assert result.is_blocked
    assert phantom not in categories_mod.reviewed_tool_names()


# --- Hard exclusions ------------------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(HARD_EXCLUDED_EVEN_IF_PRESENT))
def test_hard_excluded_tools_always_block(categories_mod, tool):
    result = categories_mod.classify(trader_tool(tool), profile=TRADER)
    assert result.is_blocked
    assert result.category is None
    assert tool in result.block_reason or "hard-excluded" in result.block_reason


@pytest.mark.parametrize("variant", [
    "ui_clickfoo", "ui_click_v2", "ui_", "ui_totally_new_primitive_added_later",
])
def test_ui_prefix_variants_block_even_when_not_explicitly_listed(categories_mod, variant):
    result = categories_mod.classify(trader_tool(variant), profile=TRADER)
    assert result.is_blocked
    assert "ui_" in result.block_reason or "UI-automation" in result.block_reason


@pytest.mark.parametrize("variant", [
    "place_order", "broker_execute", "order_submit", "live_broker_buy",
])
def test_order_broker_substring_variants_block(categories_mod, variant):
    result = categories_mod.classify(trader_tool(variant), profile=TRADER)
    assert result.is_blocked


def test_replay_trade_is_not_caught_by_the_order_broker_substring_rule(categories_mod):
    result = categories_mod.classify(trader_tool("replay_trade"), profile=TRADER)
    assert not result.is_blocked
    assert result.category == categories_mod.REPLAY_PAPER


# --- Unknown tools under the correct server still block --------------------------------------

@pytest.mark.parametrize("unknown", [
    "some_new_tool_nobody_reviewed", "get_account_balance", "withdraw_funds",
])
def test_unknown_tools_under_expected_server_block(categories_mod, unknown):
    result = categories_mod.classify(trader_tool(unknown), profile=TRADER)
    assert result.is_blocked
    assert not result.ignore


# --- Profile/server binding (item 4) ----------------------------------------------------------

def test_trader_accepts_only_tradingview_desktop_server(categories_mod):
    result = categories_mod.classify("mcp__tradingview_desktop__replay_trade", profile=TRADER)
    assert not result.is_blocked


def test_tradingresearch_accepts_only_tradingview_market_data_server(categories_mod):
    result = categories_mod.classify("mcp__tradingview_market_data__yahoo_price", profile=RESEARCH)
    assert not result.is_blocked


@pytest.mark.parametrize("wrong_server", [
    "tradingview_market_data",  # the OTHER profile's server
    "gbrain",
    "omniroute",
    "tradingview",  # unnamespaced/legacy Desktop CDP-bridge key, not the sanitized wire form
])
def test_trader_blocks_calls_to_any_other_server(categories_mod, wrong_server):
    result = categories_mod.classify(f"mcp__{wrong_server}__replay_trade", profile=TRADER)
    assert result.is_blocked
    assert not result.ignore


@pytest.mark.parametrize("wrong_server", ["tradingview_desktop", "gbrain", "tradingview-market-data"])
def test_tradingresearch_blocks_calls_to_any_other_server(categories_mod, wrong_server):
    result = categories_mod.classify(f"mcp__{wrong_server}__yahoo_price", profile=RESEARCH)
    assert result.is_blocked


@pytest.mark.parametrize("prefix_variant", [
    "tradingview_desktopx", "tradingview_desktop2", "xtradingview_desktop",
])
def test_server_prefix_variants_of_the_expected_server_still_block(categories_mod, prefix_variant):
    """A server name that merely starts with (or contains) the expected server string must not
    pass -- only an EXACT match is accepted."""
    result = categories_mod.classify(f"mcp__{prefix_variant}__replay_trade", profile=TRADER)
    assert result.is_blocked


def test_unknown_profile_blocks(categories_mod):
    result = categories_mod.classify(trader_tool("replay_trade"), profile="some-other-profile")
    assert result.is_blocked
    assert not result.ignore


def test_wrong_server_blocks_before_hard_exclusion_or_category_mapping_would_even_apply(categories_mod):
    """A hard-excluded-sounding OR perfectly legitimate tool name on the wrong server still
    blocks on the server mismatch -- server binding is checked first."""
    result = categories_mod.classify("mcp__tradingview_market_data__ui_evaluate", profile=TRADER)
    assert result.is_blocked
    assert "server" in result.block_reason.lower()


class TestBatchRun:
    @pytest.mark.parametrize("action", ["screenshot", "get_ohlcv", "get_strategy_results"])
    def test_valid_actions_classify_as_chart_change(self, categories_mod, action):
        result = categories_mod.classify(trader_tool("batch_run"), {"action": action}, profile=TRADER)
        assert not result.is_blocked
        assert result.category == categories_mod.CHART_CHANGE

    @pytest.mark.parametrize("args", [
        {}, {"action": None}, {"action": ""}, {"action": "place_order"},
        {"action": 123}, {"action": "SCREENSHOT"},  # case-sensitive, matches the real if/else chain
    ])
    def test_malformed_or_unknown_action_blocks(self, categories_mod, args):
        result = categories_mod.classify(trader_tool("batch_run"), args, profile=TRADER)
        assert result.is_blocked

    def test_batch_run_never_classifies_as_market_data_or_backtesting(self, categories_mod):
        for action in categories_mod.batch_run_allowed_actions():
            result = categories_mod.classify(trader_tool("batch_run"), {"action": action}, profile=TRADER)
            assert result.category not in (categories_mod.MARKET_DATA, categories_mod.BACKTESTING)

    def test_batch_run_on_wrong_server_blocks_before_action_validation(self, categories_mod):
        result = categories_mod.classify("mcp__gbrain__batch_run", {"action": "screenshot"}, profile=TRADER)
        assert result.is_blocked
        assert "server" in result.block_reason.lower()
