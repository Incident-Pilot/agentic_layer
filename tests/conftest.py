"""Shared pytest fixtures.

Autouse-disables the notifier's real Brevo call across the entire suite by
default -- FakeLLMClient-driven graph runs routinely reach the
CONFIRMED+actionable path that triggers agents/notifier.py, and a test
suite must never make a real external API call, same reasoning already
applied to LLM calls (FakeLLMClient) and to the Gateway (monkeypatched
GatewayContextProvider in test_investigation_trigger.py). Tests that want
to exercise the notifier itself (test_notifier.py) opt back in explicitly
by monkeypatching these config vars to real-looking values themselves.
"""

import pytest

from incident_pilot_agent import config


@pytest.fixture(autouse=True)
def _no_real_notifications(monkeypatch):
    monkeypatch.setattr(config, "BREVO_API_KEY", None)
    monkeypatch.setattr(config, "BREVO_SENDER_EMAIL", None)
    monkeypatch.setattr(config, "NOTIFICATION_EMAIL_TO", None)
