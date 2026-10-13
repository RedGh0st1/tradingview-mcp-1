"""Date-scoped rule_key construction and the redacted approval message.

Kept separate from categories.py and __init__.py so the date-isolation behavior (the whole
point of the "always expires the next day" requirement) can be unit tested as a pure function of
its inputs, without touching a clock, a profile, or Hermes itself.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


def local_calendar_day(now: _dt.datetime | None = None) -> str:
    """``YYYY-MM-DD`` for *now* (default: real local time) in the SYSTEM LOCAL timezone.

    Deliberately local, not UTC: the reviewed requirement is "each local calendar day," and an
    operator's day boundary is their own clock, not UTC's. Callers needing a specific instant
    (tests, date-isolation checks) pass an explicit ``now``.
    """
    moment = now if now is not None else _dt.datetime.now()
    return moment.date().isoformat()


def build_rule_key(profile: str, category: str, *, now: _dt.datetime | None = None) -> str:
    """``<profile>:<category>:<local-date>`` -- the exact string namespaced by Hermes as
    ``plugin_rule:<rule_key>`` when persisted via an "always" choice (tools/approval.py).

    Granularity is per (profile, category, day): every tool mapped to the same category on the
    same profile on the same day shares one rule_key, matching "first tool use in each category
    each local calendar day." A different profile, a different category, or the next day all
    produce a different string with no match in that profile's command_allowlist, so the human
    gate re-fires -- this is Hermes's own allowlist string-matching, not new logic.
    """
    if not profile:
        raise ValueError("profile must be non-empty")
    if not category:
        raise ValueError("category must be non-empty")
    return f"{profile}:{category}:{local_calendar_day(now)}"


@dataclass(frozen=True)
class ApprovalRequest:
    """Everything the pre_tool_call hook hands to Hermes's approval gate."""
    action: str  # always "approve" or "block" for this plugin -- never "modify"
    message: str
    rule_key: str | None = None

    def as_directive(self) -> dict:
        """The exact dict shape hermes_cli/plugins.py:_get_pre_tool_call_directive_details expects.

        A "block" directive with an empty/missing message is silently ignored by Hermes (treated
        as no directive at all -- fail OPEN), so ``message`` is asserted non-empty here rather
        than left to chance; every call site in __init__.py must go through this constructor.
        """
        if not self.message:
            raise ValueError("message must be non-empty -- an empty block message fails open in Hermes")
        directive: dict = {"action": self.action, "message": self.message}
        if self.action == "approve" and self.rule_key:
            directive["rule_key"] = self.rule_key
        return directive


def redact_message(
    *, profile: str, category: str, tool_name: str, effect: str,
    now: _dt.datetime | None = None,
) -> str:
    """Build the human-facing approval message.

    Redacted by design: profile, category, bare tool name, and a short static effect description
    only -- never raw tool arguments (a symbol, a Pine script body, an order-book size) which may
    be large, sensitive, or attacker/model-influenced text that shouldn't be rendered verbatim
    into a CLI/gateway approval card or a macOS notification body.

    Must visibly state the local date and what choosing "Always" actually means, matching the
    real rule_key scope exactly: this profile, this category, this date only -- not permanent.
    A human approving on faith that "Always" behaves the way it sounds (forever) would be
    approving something broader than what actually happens; the message says otherwise up front.
    """
    date = local_calendar_day(now)
    return (
        f"[{profile}/{category}] {tool_name} — {effect}. "
        f"Date: {date}. "
        f"Choosing Always approves {category} for {profile} on {date} only — "
        "it will ask again after this date, not permanently."
    )


def block_message(*, tool_name: str, reason: str) -> str:
    """Redacted block message: tool name and the classifier's own (already-generic) reason."""
    return f"BLOCKED by trading-consent: {tool_name} — {reason}"
