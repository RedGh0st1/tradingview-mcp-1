"""Tool -> consent-category classification for the trading-consent plugin.

Pure, side-effect-free logic so it can be exhaustively unit tested without a live Hermes
process. classify() returns exactly one of three states for a tool call: IGNORE (not an
MCP-namespaced trading tool at all -- a built-in like clarify/vision this plugin has no opinion
on), APPROVE-worthy (a recognized category, route to the human-approval gate), or BLOCK (an
explicit reason, never reaches the approval gate). There is no fourth "no opinion" state for an
MCP-namespaced call reaching the reviewed server -- that state is exactly Hermes's fail-open gap
(see plugin.yaml / __init__.py docstrings): a directive that isn't returned can't be blocked.

Reviewed scope: the two profiles this plugin governs are `tradingresearch` (Docker-backed
Python market-data server, MCP key `tradingview-market-data`) and `trader` (Node.js
TradingView-Desktop CDP bridge, MCP key `tradingview-desktop`). Each profile is bound to exactly
one MCP server -- see PROFILE_SERVERS. A tool call namespaced for any other server (wrong,
unknown, or a prefix-variant of the expected one) blocks before bare-name category mapping ever
runs; it is never silently ignored, because "ignore" is reserved for calls that aren't
MCP-namespaced trading tools at all.

Tool names below were read directly out of each server's own source during architecture review,
not guessed. Only tools actually present in trader's live, reviewed 45-tool allowlist (plus the
Pine tools this plugin's own requirements explicitly add) are classified -- a tool that exists in
the underlying MCP server's full ~78-tool surface but was never allowlisted for trader is
deliberately NOT pre-authorized here; adding it to the live allowlist without also reviewing and
adding it here means it blocks as unknown until that review happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

# --- Categories -----------------------------------------------------------------------------

MARKET_DATA = "market-data"
PINE_READ_OPEN = "pine-read-open"
PINE_WRITE_BUILD_SAVE = "pine-write-build-save"
BACKTESTING = "backtesting"
CHART_CHANGE = "chart-change"
ALERTS = "alerts"
REPLAY_PAPER = "replay-paper"

ALL_CATEGORIES = frozenset({
    MARKET_DATA, PINE_READ_OPEN, PINE_WRITE_BUILD_SAVE, BACKTESTING,
    CHART_CHANGE, ALERTS, REPLAY_PAPER,
})

# --- Profile -> single reviewed MCP server (sanitized wire form) -----------------------------
# Sanitization matches hermes-agent's own mcp_prefixed_tool_name(): every char outside
# [A-Za-z0-9_] (hyphens included) becomes "_", so "tradingview-desktop" and
# "tradingview-market-data" arrive here as shown. Each profile accepts calls to ITS server only
# -- a call namespaced for any other server (including the other profile's) blocks.

PROFILE_SERVERS: dict[str, str] = {
    "trader": "tradingview_desktop",
    "tradingresearch": "tradingview_market_data",
}

# --- Reviewed tool -> category map -----------------------------------------------------------
# One tool name maps to exactly one category. Deliberately a flat dict (not per-server) because
# both servers' bare tool names are already disjoint in practice; a future collision is caught by
# the completeness test asserting every entry in a reviewed profile's tools.include list resolves
# to exactly one category (see tests/unit/trading_consent/test_categories.py).

_TOOL_CATEGORY: dict[str, str] = {
    # tradingresearch (tradingview-market-data)
    "yahoo_price": MARKET_DATA,
    "market_snapshot": MARKET_DATA,
    "coin_analysis": MARKET_DATA,
    "financial_news": MARKET_DATA,
    "backtest_strategy": BACKTESTING,
    "compare_strategies": BACKTESTING,

    # trader (tradingview-desktop) -- market-data (read-only, no chart/editor mutation)
    "tv_health_check": MARKET_DATA,
    "tv_discover": MARKET_DATA,
    "tv_ui_state": MARKET_DATA,
    "symbol_info": MARKET_DATA,
    "symbol_search": MARKET_DATA,
    "quote_get": MARKET_DATA,
    "depth_get": MARKET_DATA,
    "data_get_ohlcv": MARKET_DATA,
    "data_get_indicator": MARKET_DATA,
    "data_get_study_values": MARKET_DATA,
    "capture_screenshot": MARKET_DATA,
    "chart_get_state": MARKET_DATA,
    "chart_get_visible_range": MARKET_DATA,
    "watchlist_get": MARKET_DATA,
    "layout_list": MARKET_DATA,
    "pane_list": MARKET_DATA,
    "draw_list": MARKET_DATA,
    "draw_get_properties": MARKET_DATA,
    # tv_launch is state-changing (starts/attaches a real GUI application and a live automation
    # session) -- classified chart-change, not market-data.
    "tv_launch": CHART_CHANGE,

    # NOTE: watchlist_add_bulk, watchlist_remove, and indicator_search exist in the underlying
    # server's full tool surface but are absent from trader's live, reviewed 45-tool allowlist.
    # Deliberately NOT classified here -- see module docstring. If they are ever added to the
    # live allowlist, add them here as part of that review, not before.

    # trader -- Pine read/open/analyze/check (no persisted mutation; pine_analyze and pine_check
    # do not even require a live chart connection)
    "pine_get_source": PINE_READ_OPEN,
    "pine_get_errors": PINE_READ_OPEN,
    "pine_get_console": PINE_READ_OPEN,
    "pine_list_scripts": PINE_READ_OPEN,
    "pine_analyze": PINE_READ_OPEN,
    "pine_check": PINE_READ_OPEN,

    # trader -- Pine write/build/save (authoring, compiling, persisting)
    "pine_set_source": PINE_WRITE_BUILD_SAVE,
    "pine_compile": PINE_WRITE_BUILD_SAVE,
    "pine_save": PINE_WRITE_BUILD_SAVE,
    "pine_smart_compile": PINE_WRITE_BUILD_SAVE,
    "pine_new": PINE_WRITE_BUILD_SAVE,
    "pine_open": PINE_WRITE_BUILD_SAVE,

    # trader -- backtesting (Strategy Tester reads)
    "data_get_strategy_results": BACKTESTING,
    "data_get_trades": BACKTESTING,
    "data_get_equity": BACKTESTING,

    # trader -- chart/indicator/watchlist changes (state mutation)
    "chart_set_symbol": CHART_CHANGE,
    "chart_set_timeframe": CHART_CHANGE,
    "chart_set_type": CHART_CHANGE,
    "chart_manage_indicator": CHART_CHANGE,
    "chart_set_visible_range": CHART_CHANGE,
    "chart_scroll_to_date": CHART_CHANGE,
    "indicator_set_inputs": CHART_CHANGE,
    "indicator_toggle_visibility": CHART_CHANGE,
    "watchlist_add": CHART_CHANGE,
    "layout_switch": CHART_CHANGE,
    "pane_set_layout": CHART_CHANGE,
    "pane_focus": CHART_CHANGE,
    "pane_set_symbol": CHART_CHANGE,
    "draw_shape": CHART_CHANGE,
    # batch_run is intentionally NOT listed here -- every one of its branches calls
    # setSymbol()/setResolution() regardless of the requested action, so it is classified by
    # classify_batch_run() below rather than a static entry, and always lands in CHART_CHANGE
    # once its action argument is validated.

    # trader -- alerts
    "alert_create": ALERTS,
    "alert_list": ALERTS,

    # trader -- replay / paper trading (explicitly NOT live order execution)
    "replay_start": REPLAY_PAPER,
    "replay_step": REPLAY_PAPER,
    "replay_autoplay": REPLAY_PAPER,
    "replay_stop": REPLAY_PAPER,
    "replay_trade": REPLAY_PAPER,
    "replay_status": REPLAY_PAPER,
}

# --- Hard exclusions --------------------------------------------------------------------------
# Never approved, regardless of category, regardless of what a profile's mcp_servers.tools.include
# allowlist says. This is defense in depth behind the MCP-level allowlist reviewed for trader/
# tradingresearch, not a replacement for it -- an allowlist mistake that re-adds one of these
# still gets caught here.

_HARD_EXCLUDED_EXACT: frozenset[str] = frozenset({
    "tv_update",       # self-updates the MCP via git fast-forward + npm ci -- supply-chain action
    "draw_clear",      # destructive: removes all drawings
    "draw_remove_one", # destructive: removes a specific drawing
    "alert_delete",    # destructive: deletes alert(s)
})

# Any tool literally starting with "ui_" is a raw UI-automation primitive (click/type/keyboard/
# scroll/hover/find/evaluate/open_panel). These can reconstruct live order placement by driving
# the UI even though no dedicated "place order" tool exists -- see architecture review. Prefix
# match (not an exact-name list) so a future ui_* addition is caught automatically.
_HARD_EXCLUDED_PREFIXES: tuple[str, ...] = ("ui_",)

# Defense-in-depth substring block for live brokerage/order execution. No tool with either
# substring exists in the reviewed servers today; this exists so that if one is ever added to an
# MCP server this plugin governs, it is blocked by default rather than silently falling through
# to "unknown tool -> block anyway" (which also blocks it, but this makes the *reason* explicit
# in the block message rather than a generic "unrecognized tool"). Deliberately narrow tokens:
# "trade" alone would incorrectly catch the legitimate `replay_trade` tool.
_HARD_EXCLUDED_SUBSTRINGS: tuple[str, ...] = ("order", "broker")

# batch_run's real, validated action values (read from core/batch.js's own if/else-if chain --
# not the tool schema's free-text description, which is documentation, not a schema constraint).
_BATCH_RUN_ALLOWED_ACTIONS: frozenset[str] = frozenset({"screenshot", "get_ohlcv", "get_strategy_results"})


@dataclass(frozen=True)
class Classification:
    """Result of classifying one tool call.

    Exactly one of three states:
      * ``ignore=True``            -- not an MCP-namespaced trading tool call at all (a built-in
                                       like clarify/vision); the pre_tool_call callback must
                                       return Python ``None`` for this, not a directive dict.
      * ``category`` set           -- route to the human-approval gate under this category.
      * ``block_reason`` set       -- block outright, never reaches the approval gate at all
                                       (wrong/unknown server, hard exclusion, unknown tool, or
                                       malformed arguments).

    ``server``/``bare_name`` are populated whenever the input parsed as an MCP-namespaced call
    (regardless of outcome) so callers can build messages without re-parsing.
    """
    category: Optional[str] = None
    block_reason: Optional[str] = None
    ignore: bool = False
    server: Optional[str] = None
    bare_name: Optional[str] = None

    @property
    def is_blocked(self) -> bool:
        return self.block_reason is not None


def parse_mcp_tool_name(raw_tool_name: Any) -> Optional[tuple[str, str]]:
    """Parse the REAL pre_tool_call wire form ``mcp__<server>__<tool>`` into (server, tool).

    Confirmed against hermes-agent source (tools/mcp_tool_schema.py:mcp_prefixed_tool_name --
    "the convention shared by Claude Code, Codex and OpenCode"; agent/agent_runtime_helpers.py's
    own comment that hooks see this unwrapped internal name). Server and tool components are
    each sanitized to ``[A-Za-z0-9_]`` first (hyphens become underscores) before being joined
    with the double-underscore separator, so ``tradingview-desktop`` arrives as
    ``tradingview_desktop`` -- still unambiguous from the surrounding "__" boundary since neither
    reviewed server nor tool name contains a literal double underscore of its own.

    Returns None for anything that doesn't parse as this exact form -- including a bare/built-in
    tool name (clarify, vision, ...), which is not MCP-namespaced at all and is never routed
    through this parser's callers as anything but "ignore."
    """
    if not isinstance(raw_tool_name, str):
        return None
    name = raw_tool_name.strip()
    if not name.startswith("mcp__"):
        return None
    parts = name.split("__")
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], "__".join(parts[2:])


def normalize_tool_name(raw_tool_name: str) -> str:
    """Best-effort bare tool name for display/legacy callers. Prefers the real ``mcp__`` wire
    form; tolerates ``server:tool`` (Hermes CLI's own `hermes tools enable/disable` argument
    syntax, not the pre_tool_call wire form) as a fallback; returns the input unchanged if
    neither pattern matches. classify() itself does NOT use this for server binding -- it uses
    parse_mcp_tool_name() directly so a non-mcp__ name is correctly treated as "ignore," not
    "bare name with no server to check."
    """
    parsed = parse_mcp_tool_name(raw_tool_name)
    if parsed is not None:
        return parsed[1]
    name = raw_tool_name.strip() if isinstance(raw_tool_name, str) else str(raw_tool_name)
    if ":" in name:
        _, _, rest = name.partition(":")
        if rest:
            return rest
    return name


def _hard_exclusion_reason(bare_name: str) -> Optional[str]:
    lowered = bare_name.lower()
    if bare_name in _HARD_EXCLUDED_EXACT:
        return f"'{bare_name}' is a hard-excluded tool (destructive/self-update action)."
    if any(lowered.startswith(prefix) for prefix in _HARD_EXCLUDED_PREFIXES):
        return f"'{bare_name}' is a generic UI-automation primitive (ui_* prefix) -- hard-excluded."
    for token in _HARD_EXCLUDED_SUBSTRINGS:
        if token in lowered:
            return f"'{bare_name}' matches the hard-excluded live brokerage/order token '{token}'."
    return None


def classify_batch_run(args: Optional[Mapping[str, Any]]) -> Classification:
    """Validate batch_run's ``action`` argument (server/hard-exclusion checks already passed).

    Every branch of the real implementation (core/batch.js) calls setSymbol()/setResolution()
    before doing anything else, regardless of which action was requested, so a valid batch_run
    call always lands in CHART_CHANGE -- never market-data or backtesting, which would let it
    slip past a lighter-touch approval. A missing, non-string, or unrecognized action is treated
    as malformed and blocked before it ever reaches the approval gate (fail closed on malformed
    input, not "approve and let the server error out").
    """
    action = (args or {}).get("action") if isinstance(args, Mapping) else None
    if not isinstance(action, str) or action not in _BATCH_RUN_ALLOWED_ACTIONS:
        return Classification(block_reason=(
            f"batch_run action {action!r} is missing or not one of "
            f"{sorted(_BATCH_RUN_ALLOWED_ACTIONS)} -- blocked as malformed/unknown."
        ), bare_name="batch_run")
    return Classification(category=CHART_CHANGE, bare_name="batch_run")


def classify(raw_tool_name: Any, args: Optional[Mapping[str, Any]] = None, *, profile: str) -> Classification:
    """Classify one tool call by its raw tool_name, arguments, and the ACTIVE profile.

    Order, deliberately fail-closed at every step:
      1. Not MCP-namespaced (parse_mcp_tool_name returns None) -> IGNORE. This is the only path
         that produces "no opinion" -- for built-ins like clarify/vision, never for a trading
         tool this plugin's reviewed servers expose.
      2. profile isn't one this plugin governs -> BLOCK.
      3. Parsed server != this profile's single reviewed server (wrong server, unknown server,
         or a prefix-variant of the expected one) -> BLOCK, before any bare-name mapping.
      4. Hard exclusions (exact name / ui_* prefix / order|broker substring) -> BLOCK.
      5. batch_run gets its own action-validating classifier.
      6. Anything not in the reviewed category map for THIS server -> BLOCK (unknown).
    """
    parsed = parse_mcp_tool_name(raw_tool_name)
    if parsed is None:
        return Classification(ignore=True)

    server, bare_name = parsed

    expected_server = PROFILE_SERVERS.get(profile)
    if expected_server is None:
        return Classification(
            block_reason=f"'{profile}' is not a profile this plugin governs -- blocking by design.",
            server=server, bare_name=bare_name,
        )
    if server != expected_server:
        return Classification(
            block_reason=(
                f"MCP server '{server}' is not the reviewed server for profile '{profile}' "
                f"(expected '{expected_server}') -- wrong/unknown server blocks before category "
                "mapping."
            ),
            server=server, bare_name=bare_name,
        )

    hard_reason = _hard_exclusion_reason(bare_name)
    if hard_reason is not None:
        return Classification(block_reason=hard_reason, server=server, bare_name=bare_name)

    if bare_name == "batch_run":
        result = classify_batch_run(args)
        return Classification(
            category=result.category, block_reason=result.block_reason,
            server=server, bare_name=bare_name,
        )

    category = _TOOL_CATEGORY.get(bare_name)
    if category is None:
        return Classification(
            block_reason=(
                f"'{bare_name}' is not a reviewed trading-consent tool for server '{server}' -- "
                "unknown tools block before approval by design."
            ),
            server=server, bare_name=bare_name,
        )
    return Classification(category=category, server=server, bare_name=bare_name)


def reviewed_tool_names() -> frozenset[str]:
    """All bare tool names this module has an explicit category for (excludes batch_run, which
    is validated dynamically, and hard-excluded names, which never reach a category)."""
    return frozenset(_TOOL_CATEGORY)


def hard_excluded_exact_names() -> frozenset[str]:
    return _HARD_EXCLUDED_EXACT


def hard_excluded_prefixes() -> tuple[str, ...]:
    return _HARD_EXCLUDED_PREFIXES


def batch_run_allowed_actions() -> frozenset[str]:
    return _BATCH_RUN_ALLOWED_ACTIONS
