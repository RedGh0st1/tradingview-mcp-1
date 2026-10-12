"""rule_key construction: date isolation, profile isolation, category isolation, and the
empty-block-message guard that prevents Hermes's own fail-open path (an empty block message is
silently treated as "no directive" by hermes_cli/plugins.py)."""

from __future__ import annotations

import datetime as dt

import pytest


def test_same_inputs_same_key(rule_key_mod):
    now = dt.datetime(2026, 6, 15, 10, 0, 0)
    a = rule_key_mod.build_rule_key("trader", "market-data", now=now)
    b = rule_key_mod.build_rule_key("trader", "market-data", now=now)
    assert a == b == "trader:market-data:2026-06-15"


def test_next_calendar_day_changes_the_key(rule_key_mod):
    """The entire point of the requirement: an 'always' choice must expire the next local day."""
    today = dt.datetime(2026, 6, 15, 23, 59, 59)
    tomorrow = dt.datetime(2026, 6, 16, 0, 0, 1)
    key_today = rule_key_mod.build_rule_key("trader", "market-data", now=today)
    key_tomorrow = rule_key_mod.build_rule_key("trader", "market-data", now=tomorrow)
    assert key_today != key_tomorrow
    assert key_today.endswith("2026-06-15")
    assert key_tomorrow.endswith("2026-06-16")


def test_same_day_different_time_same_key(rule_key_mod):
    """Two calls hours apart on the same calendar day must share one rule_key (one approval
    covers the rest of the day for that category), not re-key on every call."""
    morning = dt.datetime(2026, 6, 15, 8, 0, 0)
    evening = dt.datetime(2026, 6, 15, 22, 0, 0)
    assert (rule_key_mod.build_rule_key("trader", "alerts", now=morning)
            == rule_key_mod.build_rule_key("trader", "alerts", now=evening))


def test_different_profiles_never_share_a_key(rule_key_mod):
    now = dt.datetime(2026, 6, 15, 12, 0, 0)
    trader_key = rule_key_mod.build_rule_key("trader", "market-data", now=now)
    research_key = rule_key_mod.build_rule_key("tradingresearch", "market-data", now=now)
    assert trader_key != research_key


def test_different_categories_never_share_a_key(rule_key_mod):
    now = dt.datetime(2026, 6, 15, 12, 0, 0)
    a = rule_key_mod.build_rule_key("trader", "market-data", now=now)
    b = rule_key_mod.build_rule_key("trader", "alerts", now=now)
    assert a != b


@pytest.mark.parametrize("profile,category", [("", "market-data"), ("trader", "")])
def test_empty_profile_or_category_rejected(rule_key_mod, profile, category):
    with pytest.raises(ValueError):
        rule_key_mod.build_rule_key(profile, category)


def test_local_calendar_day_uses_local_not_utc_semantics(rule_key_mod):
    """local_calendar_day must key off the datetime it's given, not silently convert to UTC --
    build_rule_key/callers are responsible for passing a local-time datetime.now()."""
    naive_local = dt.datetime(2026, 1, 1, 0, 30, 0)  # 00:30 local on Jan 1
    assert rule_key_mod.local_calendar_day(naive_local) == "2026-01-01"


class TestApprovalRequestDirective:
    def test_approve_directive_shape(self, rule_key_mod):
        req = rule_key_mod.ApprovalRequest(action="approve", message="x", rule_key="trader:alerts:2026-06-15")
        directive = req.as_directive()
        assert directive == {"action": "approve", "message": "x", "rule_key": "trader:alerts:2026-06-15"}

    def test_block_directive_omits_rule_key(self, rule_key_mod):
        req = rule_key_mod.ApprovalRequest(action="block", message="nope")
        directive = req.as_directive()
        assert directive == {"action": "block", "message": "nope"}
        assert "rule_key" not in directive

    def test_empty_message_raises_rather_than_producing_a_fail_open_directive(self, rule_key_mod):
        """Hermes silently drops a block directive with no message (treated as no directive at
        all -- fail open). Constructing one must fail loudly here instead, long before it could
        ever reach Hermes's dispatcher."""
        req = rule_key_mod.ApprovalRequest(action="block", message="")
        with pytest.raises(ValueError):
            req.as_directive()


def test_redact_message_never_includes_raw_effect_beyond_the_static_description(rule_key_mod):
    msg = rule_key_mod.redact_message(
        profile="trader", category="pine-write-build-save", tool_name="pine_save",
        effect="writes, compiles, or saves a Pine script",
    )
    assert "trader" in msg and "pine-write-build-save" in msg and "pine_save" in msg
    assert "writes, compiles, or saves a Pine script" in msg


def test_redact_message_states_local_date_and_always_scope(rule_key_mod):
    """Item 8: the human-facing message must visibly say the local date and that choosing
    Always is scoped to this category/profile for that date -- not forever."""
    now = dt.datetime(2026, 6, 15, 9, 0, 0)
    msg = rule_key_mod.redact_message(
        profile="trader", category="alerts", tool_name="alert_create",
        effect="creates or lists a price alert", now=now,
    )
    assert "2026-06-15" in msg
    assert "Always" in msg
    assert "trader" in msg and "alerts" in msg


def test_redact_message_date_matches_build_rule_key_for_the_same_now(rule_key_mod):
    """The message's stated date and the rule_key's date must never drift apart -- both are
    derived from the SAME `now` the caller passes in."""
    now = dt.datetime(2026, 1, 1, 0, 0, 1)
    msg = rule_key_mod.redact_message(
        profile="trader", category="alerts", tool_name="alert_create", effect="x", now=now,
    )
    key = rule_key_mod.build_rule_key("trader", "alerts", now=now)
    assert "2026-01-01" in msg
    assert key.endswith("2026-01-01")


def test_block_message_includes_tool_and_reason(rule_key_mod):
    msg = rule_key_mod.block_message(tool_name="ui_evaluate", reason="hard-excluded")
    assert "ui_evaluate" in msg and "hard-excluded" in msg
