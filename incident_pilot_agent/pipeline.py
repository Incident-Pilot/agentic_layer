"""Shared investigation pipeline — the one function that actually runs an
incident through the graph, plus the config-resolution helpers it needs.

`run` (cli.py), `watch` (cli.py), and `POST /investigations/{incident_id}`
(api/app.py) all call `run_incident()` here rather than each having their
own copy of "build a context provider, build tools, build node LLMs, build
the graph, invoke it" -- see the top-level docstring in cli.py and the
POST route's own docstring for why that matters (watch and the HTTP
trigger can run concurrently in the same process; two independent copies
of this logic would be two independent places to keep in sync).

Deliberately free of any printing/CLI concerns (no `print()`, no
`sys.exit()`) so it's safe to call from an HTTP handler: a resolution
failure raises a normal exception instead of killing the process.
cli.py's own `run`/`watch` commands are what turn a raised exception back
into stderr output + a process exit, at the one place that's actually
appropriate.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import config
from .context_provider.base import ContextProvider
from .context_provider.fixture_provider import FixtureContextProvider
from .context_provider.gateway_provider import GatewayContextProvider
from .graph.build import build_graph, finalize_status, initial_state
from .llm.base import LLMClient
from .llm.fake_client import FakeLLMClient
from .telemetry.fixture_backends import FixtureLokiBackend, FixturePrometheusBackend, FixtureTempoBackend
from .telemetry.loki_client import LokiClient
from .telemetry.prometheus_client import PrometheusClient
from .telemetry.tempo_client import TempoClient
from .tools.loki_tool import LokiTool
from .tools.prometheus_tool import PrometheusTool
from .tools.tempo_tool import TempoTool
from .trajectory.logger import TrajectoryLogger

logger = logging.getLogger(__name__)


def _build_llm(name: str, *, model: Optional[str] = None, node: str = "") -> LLMClient:
    # Logs the model actually resolved (post env-var/.env/hardcoded-default
    # fallback, or a `model` override from a caller) for whichever provider
    # was selected -- this is the one place in the codebase that always runs
    # before any LLM API call, so it's the single source of truth for
    # "which model ran" without having to infer it after the fact from a
    # provider's own dashboard. `node` (e.g. "investigator") is purely a log
    # label -- run_incident() calls this once per graph node so tiered
    # models (see config.INVESTIGATOR_MODEL/etc.) are distinguishable here.
    node_label = f"node={node} " if node else ""
    if name == "fake":
        logger.info("llm: %sprovider=fake model=n/a", node_label)
        return FakeLLMClient()
    if name == "anthropic":
        if not config.ANTHROPIC_API_KEY:
            print("error: --llm anthropic requires ANTHROPIC_API_KEY to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.anthropic_client import AnthropicLLMClient

        resolved_model = model or config.ANTHROPIC_MODEL
        logger.info("llm: %sprovider=anthropic model=%s", node_label, resolved_model)
        return AnthropicLLMClient(api_key=config.ANTHROPIC_API_KEY, model=resolved_model)
    if name == "openai":
        if not config.OPENAI_API_KEY:
            print("error: --llm openai requires OPENAI_API_KEY to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.openai_client import OpenAILLMClient

        resolved_model = model or config.OPENAI_MODEL
        logger.info("llm: %sprovider=openai model=%s", node_label, resolved_model)
        return OpenAILLMClient(api_key=config.OPENAI_API_KEY, model=resolved_model)
    if name == "gemini":
        if not config.GEMINI_API_KEY:
            print("error: --llm gemini requires GEMINI_API_KEY to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.gemini_client import GeminiLLMClient

        resolved_model = model or config.GEMINI_MODEL
        logger.info("llm: %sprovider=gemini model=%s", node_label, resolved_model)
        return GeminiLLMClient(api_key=config.GEMINI_API_KEY, model=resolved_model)
    if name == "openrouter":
        if not config.OPENROUTER_API_KEY:
            print("error: --llm openrouter requires OPENROUTER_API_KEY to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.openai_client import OpenAILLMClient

        resolved_model = model or config.OPENROUTER_MODEL
        logger.info("llm: %sprovider=openrouter model=%s", node_label, resolved_model)
        return OpenAILLMClient(
            api_key=config.OPENROUTER_API_KEY,
            model=resolved_model,
            base_url=config.OPENROUTER_BASE_URL,
        )
    if name == "bedrock":
        if not config.BEDROCK_API_KEY:
            print("error: --llm bedrock requires AWS_BEARER_TOKEN_BEDROCK to be set", file=sys.stderr)
            sys.exit(1)
        from .llm.openai_client import OpenAILLMClient

        resolved_model = model or config.BEDROCK_MODEL
        logger.info("llm: %sprovider=bedrock model=%s", node_label, resolved_model)
        return OpenAILLMClient(
            api_key=config.BEDROCK_API_KEY,
            model=resolved_model,
            base_url=config.BEDROCK_BASE_URL,
        )
    raise ValueError(f"unknown --llm {name!r}")


# config.INVESTIGATOR_MODEL/SYNTHESIZER_MODEL/VERIFIER_MODEL (OpenRouter) and
# config.BEDROCK_INVESTIGATOR_MODEL/etc. (Bedrock) are each namespaced to
# their own provider's model-id form (e.g. "openai/gpt-4o-mini" vs.
# "anthropic.claude-sonnet-5"), so they only make sense to apply when that
# same provider is selected via --llm -- passing one to a different
# provider's SDK directly would just be an invalid model id there. Other
# providers keep using their single existing *_MODEL config for every node,
# unaffected.
def _node_model(llm_name: str, node_model: str) -> Optional[str]:
    return node_model if llm_name in ("openrouter", "bedrock") else None


def _node_models_for(llm_name: str) -> Dict[str, str]:
    if llm_name == "bedrock":
        return {
            "investigator": config.BEDROCK_INVESTIGATOR_MODEL,
            "synthesizer": config.BEDROCK_SYNTHESIZER_MODEL,
            "verifier": config.BEDROCK_VERIFIER_MODEL,
            "remediation": config.BEDROCK_REMEDIATION_MODEL,
            "postmortem": config.BEDROCK_POSTMORTEM_MODEL,
        }
    return {
        "investigator": config.INVESTIGATOR_MODEL,
        "synthesizer": config.SYNTHESIZER_MODEL,
        "verifier": config.VERIFIER_MODEL,
        "remediation": config.REMEDIATION_MODEL,
        "postmortem": config.POSTMORTEM_MODEL,
    }


def _build_node_llms(llm_name: str) -> Dict[str, LLMClient]:
    node_models = _node_models_for(llm_name)
    return {
        node: _build_llm(llm_name, model=_node_model(llm_name, node_models[node]), node=node)
        for node in ("investigator", "synthesizer", "verifier", "remediation", "postmortem")
    }


def _default_llm_name() -> str:
    if config.OPENAI_API_KEY:
        return "openai"
    if config.ANTHROPIC_API_KEY:
        return "anthropic"
    if config.GEMINI_API_KEY:
        return "gemini"
    if config.OPENROUTER_API_KEY:
        return "openrouter"
    if config.BEDROCK_API_KEY:
        return "bedrock"
    return "fake"


def _build_context_provider(incident_id: str, fixtures_root: Path, source: str) -> ContextProvider:
    """`source` is "fixtures", "gateway", or "auto" (default): auto picks
    gateway only when incident_id doesn't match a local fixture, so existing
    fixture-based workflows/tests are unaffected unless a real incident_id
    is actually passed.

    Raises ValueError (never exits the process) when a gateway-sourced
    incident_id is requested but INCIDENT_GATEWAY_URL/INCIDENT_GATEWAY_API_KEY
    aren't configured -- callers that want the old CLI behavior (print to
    stderr + exit 1) catch this at their own entrypoint; an HTTP handler
    turns it into an error response instead."""
    use_gateway = source == "gateway" or (source == "auto" and not (fixtures_root / incident_id).exists())
    if not use_gateway:
        return FixtureContextProvider(fixtures_root)

    if not config.INCIDENT_GATEWAY_URL or not config.INCIDENT_GATEWAY_API_KEY:
        raise ValueError(
            f"incident_id {incident_id!r} not found under {fixtures_root} and "
            "INCIDENT_GATEWAY_URL/INCIDENT_GATEWAY_API_KEY are not set -- nothing to run against"
        )
    return GatewayContextProvider(config.INCIDENT_GATEWAY_URL, config.INCIDENT_GATEWAY_API_KEY)


def _build_tools(provider: ContextProvider, incident_id: str) -> list:
    if isinstance(provider, FixtureContextProvider):
        fixtures_dir = provider.incident_dir(incident_id)
        return [
            PrometheusTool(FixturePrometheusBackend(fixtures_dir)),
            LokiTool(FixtureLokiBackend(fixtures_dir)),
            TempoTool(FixtureTempoBackend(fixtures_dir)),
        ]

    # Gateway-sourced incident: live investigation still goes through this
    # repo's own telemetry clients (never the Gateway) -- only queried for
    # whichever backends are actually configured.
    tools: list = []
    if config.PROMETHEUS_BASE_URL:
        tools.append(PrometheusTool(PrometheusClient(config.PROMETHEUS_BASE_URL)))
    if config.LOKI_BASE_URL:
        tools.append(LokiTool(LokiClient(config.LOKI_BASE_URL)))
    if config.TEMPO_BASE_URL:
        tools.append(TempoTool(TempoClient(config.TEMPO_BASE_URL)))
    return tools


async def run_incident(
    incident_id: str,
    *,
    llm_name: str,
    fixtures_root: Path,
    trajectory_dir: Path,
    max_iterations: int,
    source: str = "auto",
) -> Tuple[dict, TrajectoryLogger]:
    """Given an incident_id, run the full investigation pipeline: resolve a
    context provider, fetch context, build tools/node LLMs/the graph,
    invoke it to a final verdict. Returns (result, trajectory) -- the
    trajectory file is written incrementally (TrajectoryLogger.log()
    flushes to disk on every node, not just at the end), so a concurrent
    GET /investigations/{incident_id} can observe live progress through
    this call, not just its final outcome."""
    provider = _build_context_provider(incident_id, fixtures_root, source)
    context = await provider.get_context(incident_id)

    node_llms = _build_node_llms(llm_name)
    tools = _build_tools(provider, incident_id)
    trajectory = TrajectoryLogger(incident_id, trajectory_dir)

    graph = build_graph(
        node_llms["investigator"],
        tools,
        trajectory,
        investigator_llm=node_llms["investigator"],
        synthesizer_llm=node_llms["synthesizer"],
        verifier_llm=node_llms["verifier"],
        remediation_llm=node_llms["remediation"],
        postmortem_llm=node_llms["postmortem"],
    )
    result = await graph.ainvoke(initial_state(context, max_iterations=max_iterations))
    result = finalize_status(result)

    return result, trajectory


def load_processed_incidents(state_file: Path) -> Set[str]:
    if not state_file.exists():
        return set()
    try:
        return set(json.loads(state_file.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()


def save_processed_incidents(state_file: Path, processed: Set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(sorted(processed), indent=2))


# Re-exported under their old cli.py names too, so existing call sites
# (cli.py itself, plus tests that reach into `cli._build_llm` etc.) keep
# working unchanged after this module became the single source of truth.
_load_processed_incidents = load_processed_incidents
_save_processed_incidents = save_processed_incidents
