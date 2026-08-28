"""Regression tests for _build_llm's startup log line.

Context: OpenRouter's own dashboard showed a different model on every run
one night (gpt-4o-mini, claude-haiku-4.5, nemotron-3-ultra) even though the
developer believed a specific OPENROUTER_MODEL was set. Root cause was
`export OPENROUTER_MODEL=...` only persisting for that one shell session --
a fresh terminal silently fell back to config.py's hardcoded default. The
fix is not new selection logic, just an unmissable log line reporting
whichever model actually got resolved, emitted before any LLM API call.

These tests reload config.py with dotenv disabled so the "unset" case
reflects config.py's hardcoded fallback rather than whatever OPENROUTER_MODEL
happens to be in the developer's real local .env file.
"""

import importlib
import logging

import dotenv

from incident_pilot_agent import cli, config


def _reload_config_with_dotenv_disabled(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    importlib.reload(config)


def _restore_real_config(monkeypatch):
    # Undo env/attr patches *before* reloading, so the restore reflects the
    # developer's real environment and .env file again, not the test's.
    monkeypatch.undo()
    importlib.reload(config)


def test_build_llm_logs_hardcoded_fallback_when_env_unset(monkeypatch, caplog):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.OPENROUTER_MODEL == "openai/gpt-4o-mini"

        with caplog.at_level(logging.INFO, logger="incident_pilot_agent.cli"):
            cli._build_llm("openrouter")

        assert "provider=openrouter model=openai/gpt-4o-mini" in caplog.text
    finally:
        _restore_real_config(monkeypatch)


def test_build_llm_logs_configured_model_exactly(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.OPENROUTER_MODEL == "anthropic/claude-haiku-4.5"

        with caplog.at_level(logging.INFO, logger="incident_pilot_agent.cli"):
            cli._build_llm("openrouter")

        assert "provider=openrouter model=anthropic/claude-haiku-4.5" in caplog.text
    finally:
        _restore_real_config(monkeypatch)
