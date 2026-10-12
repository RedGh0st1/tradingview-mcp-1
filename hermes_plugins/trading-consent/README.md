# trading-consent

Daily human-approval gate for the `trader` and `tradingresearch` Hermes profiles. See
`../../HERMES_INTEGRATION.md` for the full integration story; this file covers just this plugin.

## What it does

- `pre_tool_call` classifies every call to a reviewed TradingView MCP tool into one of:
  `market-data`, `pine-read-open`, `pine-write-build-save`, `backtesting`, `chart-change`,
  `alerts`, `replay-paper` (see `categories.py` for the exact tool-name map).
- The first call in a category on a given **local calendar day** escalates to Hermes's existing
  human-approval gate (`tools.approval.request_tool_approval`) with a `rule_key` of
  `<profile>:<category>:<local-date>`. Choosing **always** persists that exact key — it stops
  applying the moment the date changes, i.e. it expires at local midnight, not permanently.
  **once** persists nothing (re-prompts next call); **session** lasts for the running Hermes
  process regardless of date; **deny**/**timeout**/no interactive human all BLOCK.
- `pre_approval_request` (observer only, cannot approve) fires a best-effort macOS notification
  alongside Hermes's own CLI/gateway prompt.
- Generic `ui_*` tools, `tv_update`, `draw_clear`, `draw_remove_one`, `alert_delete`, anything
  matching `order`/`broker`, and any tool this plugin doesn't recognize are **hard-blocked**
  before the approval gate is ever reached — never merely "not yet approved."
- `batch_run`'s `action` argument is validated against the real implementation's only three
  branches (`screenshot`, `get_ohlcv`, `get_strategy_results`); anything else is blocked as
  malformed. A valid call is classified `chart-change` (every branch mutates chart state).

## Fail-open limitation this plugin closes itself

Hermes's plugin dispatcher fails closed on a `pre_tool_call` **timeout** and on any failure
**after** a directive is returned (gate error, deny, timeout, no human present) — but fails
**open** if this callback raises an exception before returning any directive at all (the
exception is logged and the result is simply dropped, indistinguishable from "no plugin has an
opinion"). `_on_pre_tool_call` therefore wraps its entire body in a catch-all that returns an
explicit `{"action": "block", ...}` on any internal error — see `__init__.py`'s docstring.

## `guarded_mcp_launcher.py`

A second, independent gate: disabling/removing this plugin from a profile's `plugins.enabled`
would otherwise leave the underlying `mcp_servers` command fully functional with zero approval
gate and no error. Point the profile's MCP entry at this launcher instead of the real command
directly; it hard-stops (never execs the real command) unless a marker this plugin's `register()`
writes on every Hermes startup is present, matches the profile, and is fresh.

```yaml
# illustrative only — not written to any live config by this plugin
mcp_servers:
  tradingview-desktop:            # trader profile
    command: python3
    args: [/path/to/hermes_plugins/trading-consent/guarded_mcp_launcher.py, node, /path/to/tradingview-mcp/src/server.js]
    env:
      TRADING_CONSENT_PROFILE: trader

  tradingview-market-data:        # tradingresearch profile
    command: python3
    args: [/path/to/hermes_plugins/trading-consent/guarded_mcp_launcher.py, docker, run, --rm, -i, --read-only, --tmpfs, /tmp, --cap-drop, ALL, --security-opt, no-new-privileges, "tradingview-mcp-hermes:<tag>"]
    env:
      TRADING_CONSENT_PROFILE: tradingresearch
```

`TRADING_CONSENT_PROFILE_HOME` overrides the launcher's default `~/.hermes/profiles/<profile>`
guess for a non-default `HERMES_HOME`. Both env values are plain identifiers, never secrets.

The real command is `execvp`'d directly — no `shell=True`, no shell metacharacter interpretation
of any argument.

## Installing (not performed by this repository)

Install per-profile with Hermes's own plugin tooling. `hermes plugins install` does not accept a
local directory path; install from the Git URL with this directory as a `#` fragment, pinned to
an immutable commit:

```bash
hermes --profile trader plugins install \
  "https://github.com/RedGh0st1/tradingview-mcp-1.git#hermes_plugins/trading-consent" \
  --ref <COMMIT_SHA>
```

`<COMMIT_SHA>` is a placeholder for the full 40-character SHA of the reviewed commit containing
this plugin, as pushed to that remote (`--ref` rejects branches, tags, and short SHAs). Then
confirm with `hermes --profile trader plugins list` and
`hermes --profile trader plugins doctor trading-consent`. This repository does not install the
plugin into any live profile, write `mcp_servers` config, or call any trading tool.

**Note on `hermes plugins validate`/`doctor` run against a plain path** (e.g.
`hermes plugins validate hermes_plugins/trading-consent`, with no `--profile`): Hermes's own
"capability probe" calls this plugin's real `register()`, which writes its registration marker
under whatever `HERMES_HOME` is active at that moment -- the current default profile if none is
selected. This is a side effect of Hermes's validator, not of this repository. It's an inert
marker file (`.trading_consent/registered.json`) with no effect unless something else also wires
up `guarded_mcp_launcher.py` for that profile; safe to delete (`rm -rf ~/.hermes/.trading_consent`
or the equivalent under the profile actually used) after validating.
