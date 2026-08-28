"""Regression tests for the per-graph-node model config (config.py's
INVESTIGATOR_MODEL/SYNTHESIZER_MODEL/VERIFIER_MODEL/REMEDIATION_MODEL) and
cli.py's _node_model gate that decides when they actually apply.

Same reload pattern as test_llm_resolution_logging.py: config.py reads
these via os.environ.get() at import time, so each test reloads the
module with dotenv disabled to isolate it from the developer's real .env.
"""

import importlib

import dotenv

from incident_pilot_agent import cli, config


def _reload_config_with_dotenv_disabled(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    importlib.reload(config)


def _restore_real_config(monkeypatch):
    monkeypatch.undo()
    importlib.reload(config)


def test_node_models_fall_back_to_openrouter_model_when_unset(monkeypatch):
    monkeypatch.delenv("INVESTIGATOR_MODEL", raising=False)
    monkeypatch.delenv("SYNTHESIZER_MODEL", raising=False)
    monkeypatch.delenv("VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("REMEDIATION_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.OPENROUTER_MODEL == "openai/gpt-4o-mini"
        assert config.INVESTIGATOR_MODEL == "openai/gpt-4o-mini"
        assert config.SYNTHESIZER_MODEL == "openai/gpt-4o-mini"
        assert config.VERIFIER_MODEL == "openai/gpt-4o-mini"
        assert config.REMEDIATION_MODEL == "openai/gpt-4o-mini"
    finally:
        _restore_real_config(monkeypatch)


def test_node_models_fall_back_to_a_custom_openrouter_model_when_unset(monkeypatch):
    """The fallback tracks whatever OPENROUTER_MODEL actually resolves to,
    not just its hardcoded default."""
    monkeypatch.delenv("INVESTIGATOR_MODEL", raising=False)
    monkeypatch.delenv("SYNTHESIZER_MODEL", raising=False)
    monkeypatch.delenv("VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("REMEDIATION_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.INVESTIGATOR_MODEL == "anthropic/claude-haiku-4.5"
        assert config.SYNTHESIZER_MODEL == "anthropic/claude-haiku-4.5"
        assert config.VERIFIER_MODEL == "anthropic/claude-haiku-4.5"
        assert config.REMEDIATION_MODEL == "anthropic/claude-haiku-4.5"
    finally:
        _restore_real_config(monkeypatch)


def test_node_models_override_independently_when_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("INVESTIGATOR_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("SYNTHESIZER_MODEL", "anthropic/claude-sonnet-5")
    monkeypatch.setenv("VERIFIER_MODEL", "anthropic/claude-sonnet-5")
    monkeypatch.setenv("REMEDIATION_MODEL", "anthropic/claude-sonnet-5")
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.INVESTIGATOR_MODEL == "openai/gpt-4o-mini"
        assert config.SYNTHESIZER_MODEL == "anthropic/claude-sonnet-5"
        assert config.VERIFIER_MODEL == "anthropic/claude-sonnet-5"
        assert config.REMEDIATION_MODEL == "anthropic/claude-sonnet-5"
        # each var is independent -- overriding one must not affect the others'
        # fallback to OPENROUTER_MODEL
        assert config.OPENROUTER_MODEL == "openai/gpt-4o-mini"
    finally:
        _restore_real_config(monkeypatch)


def test_node_model_only_applies_for_openrouter_and_bedrock_providers():
    """INVESTIGATOR_MODEL/etc. are OpenRouter-namespaced model ids (e.g.
    "openai/gpt-4o-mini") and BEDROCK_INVESTIGATOR_MODEL/etc. are
    Bedrock-namespaced (e.g. "anthropic.claude-sonnet-5") -- passing either
    straight to a different provider's SDK would just be an invalid model
    id there, so cli._node_model must only apply the override when that
    same provider is selected. Other providers keep their own single
    *_MODEL config for every node, unaffected."""
    assert cli._node_model("openrouter", "anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert cli._node_model("bedrock", "anthropic.claude-sonnet-5") == "anthropic.claude-sonnet-5"
    assert cli._node_model("anthropic", "anthropic/claude-sonnet-5") is None
    assert cli._node_model("openai", "anthropic/claude-sonnet-5") is None
    assert cli._node_model("gemini", "anthropic/claude-sonnet-5") is None
    assert cli._node_model("fake", "anthropic/claude-sonnet-5") is None


def test_bedrock_node_models_fall_back_to_bedrock_model_when_unset(monkeypatch):
    monkeypatch.delenv("BEDROCK_INVESTIGATOR_MODEL", raising=False)
    monkeypatch.delenv("BEDROCK_SYNTHESIZER_MODEL", raising=False)
    monkeypatch.delenv("BEDROCK_VERIFIER_MODEL", raising=False)
    monkeypatch.delenv("BEDROCK_REMEDIATION_MODEL", raising=False)
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.BEDROCK_MODEL == "moonshotai.kimi-k2.5"
        assert config.BEDROCK_INVESTIGATOR_MODEL == "moonshotai.kimi-k2.5"
        assert config.BEDROCK_SYNTHESIZER_MODEL == "moonshotai.kimi-k2.5"
        assert config.BEDROCK_VERIFIER_MODEL == "moonshotai.kimi-k2.5"
        assert config.BEDROCK_REMEDIATION_MODEL == "moonshotai.kimi-k2.5"
    finally:
        _restore_real_config(monkeypatch)


def test_bedrock_node_models_override_independently_when_set(monkeypatch):
    monkeypatch.setenv("BEDROCK_MODEL", "moonshotai.kimi-k2.5")
    monkeypatch.setenv("BEDROCK_INVESTIGATOR_MODEL", "anthropic.claude-haiku-4-5")
    monkeypatch.setenv("BEDROCK_SYNTHESIZER_MODEL", "anthropic.claude-sonnet-5")
    monkeypatch.setenv("BEDROCK_VERIFIER_MODEL", "anthropic.claude-sonnet-5")
    monkeypatch.setenv("BEDROCK_REMEDIATION_MODEL", "anthropic.claude-sonnet-5")
    _reload_config_with_dotenv_disabled(monkeypatch)
    try:
        assert config.BEDROCK_INVESTIGATOR_MODEL == "anthropic.claude-haiku-4-5"
        assert config.BEDROCK_SYNTHESIZER_MODEL == "anthropic.claude-sonnet-5"
        assert config.BEDROCK_VERIFIER_MODEL == "anthropic.claude-sonnet-5"
        assert config.BEDROCK_REMEDIATION_MODEL == "anthropic.claude-sonnet-5"
        # each var is independent -- overriding one must not affect the
        # others' fallback to BEDROCK_MODEL
        assert config.BEDROCK_MODEL == "moonshotai.kimi-k2.5"
    finally:
        _restore_real_config(monkeypatch)


def test_node_models_for_selects_provider_specific_config():
    assert cli._node_models_for("openrouter") == {
        "investigator": config.INVESTIGATOR_MODEL,
        "synthesizer": config.SYNTHESIZER_MODEL,
        "verifier": config.VERIFIER_MODEL,
        "remediation": config.REMEDIATION_MODEL,
    }
    assert cli._node_models_for("bedrock") == {
        "investigator": config.BEDROCK_INVESTIGATOR_MODEL,
        "synthesizer": config.BEDROCK_SYNTHESIZER_MODEL,
        "verifier": config.BEDROCK_VERIFIER_MODEL,
        "remediation": config.BEDROCK_REMEDIATION_MODEL,
    }
