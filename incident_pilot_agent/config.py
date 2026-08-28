"""Local configuration. No Postgres/pgvector, no Kubernetes config here --
this repo runs entirely on a developer machine against fixtures (or,
optionally, a real read-only Prometheus/Loki/Tempo endpoint) per the
top-level spec's "local-first, no deployment in this phase" constraint.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES_DIR = REPO_ROOT / "fixtures" / "incidents"
DEFAULT_TRAJECTORY_DIR = REPO_ROOT / "trajectories"

# Loads REPO_ROOT/.env into the process environment if present (never
# overrides a variable already set in the real environment). Safe to call
# even with no .env file -- this is the only place dotenv is used.
load_dotenv(REPO_ROOT / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

# OpenRouter speaks the OpenAI Chat Completions wire format, so it reuses
# OpenAILLMClient with a different base_url instead of its own client.
# Model ids are OpenRouter's own namespaced form, e.g. "openai/gpt-4o-mini"
# or "google/gemini-2.0-flash-exp:free" -- see https://openrouter.ai/models
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-graph-node OpenRouter model overrides -- see .env.example for the
# reasoning behind the suggested cheap/strong split. Each falls back to the
# single OPENROUTER_MODEL above when unset, so this is purely additive:
# nothing changes for a deployment that never sets these. Only meaningful
# for --llm openrouter (these are OpenRouter-namespaced model ids, e.g.
# "openai/gpt-4o-mini"); cli.py._build_llm only applies them when the
# openrouter provider is selected, since passing an OpenRouter-shaped id to
# the Anthropic/OpenAI/Gemini APIs directly would just be an invalid model.
INVESTIGATOR_MODEL = os.environ.get("INVESTIGATOR_MODEL", OPENROUTER_MODEL)
SYNTHESIZER_MODEL = os.environ.get("SYNTHESIZER_MODEL", OPENROUTER_MODEL)
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", OPENROUTER_MODEL)
# No remediation graph node exists yet (see graph/build.py) -- this is
# reserved so the config surface is already in place when one lands.
REMEDIATION_MODEL = os.environ.get("REMEDIATION_MODEL", OPENROUTER_MODEL)

# Bedrock, reached via the Mantle gateway, speaks the same OpenAI Chat
# Completions wire format as OpenRouter -- same OpenAILLMClient, different
# base_url/key/model, no separate client module. Model ids are Bedrock's
# own "provider.model" form, e.g. "moonshotai.kimi-k2.5" or
# "anthropic.claude-sonnet-5".
BEDROCK_API_KEY = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
# Observed with a stray leading/trailing space when pasted from the AWS
# console into .env -- stripped here since that silently breaks Bearer auth.
BEDROCK_API_KEY = BEDROCK_API_KEY.strip() if BEDROCK_API_KEY else BEDROCK_API_KEY
BEDROCK_BASE_URL = os.environ.get("BEDROCK_BASE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1")
BEDROCK_MODEL = os.environ.get("BEDROCK_MODEL", "moonshotai.kimi-k2.5")

# Per-graph-node Bedrock model overrides -- same reasoning and same
# cli.py._node_model gate as INVESTIGATOR_MODEL/etc. above, just scoped to
# --llm bedrock instead of --llm openrouter (these are Bedrock-namespaced
# model ids, e.g. "anthropic.claude-sonnet-5", invalid against the other
# SDKs directly). Each falls back to the single BEDROCK_MODEL when unset.
BEDROCK_INVESTIGATOR_MODEL = os.environ.get("BEDROCK_INVESTIGATOR_MODEL", BEDROCK_MODEL)
BEDROCK_SYNTHESIZER_MODEL = os.environ.get("BEDROCK_SYNTHESIZER_MODEL", BEDROCK_MODEL)
BEDROCK_VERIFIER_MODEL = os.environ.get("BEDROCK_VERIFIER_MODEL", BEDROCK_MODEL)
BEDROCK_REMEDIATION_MODEL = os.environ.get("BEDROCK_REMEDIATION_MODEL", BEDROCK_MODEL)

# Optional: point at a real read-only observability stack instead of
# fixtures. Unset by default -- FixtureContextProvider + fixture-backed
# tools are what this phase actually runs against.
PROMETHEUS_BASE_URL = os.environ.get("PROMETHEUS_BASE_URL")
LOKI_BASE_URL = os.environ.get("LOKI_BASE_URL")
TEMPO_BASE_URL = os.environ.get("TEMPO_BASE_URL")

# incident-pilot-ecommerce's observation-gateway -- unset by default, same
# "optional, real backend" pattern as PROMETHEUS_BASE_URL/etc. above.
# GatewayContextProvider (context_provider/gateway_provider.py) is only
# used when both are set; FixtureContextProvider remains the default.
INCIDENT_GATEWAY_URL = os.environ.get("INCIDENT_GATEWAY_URL")
INCIDENT_GATEWAY_API_KEY = os.environ.get("INCIDENT_GATEWAY_API_KEY")

DEFAULT_MAX_ITERATIONS = int(os.environ.get("INCIDENT_PILOT_MAX_ITERATIONS", "4"))

# Read-only investigation API (incident_pilot_agent/api/) -- serves the
# incident-pilot-dashboard repo. Runs in the same process as `watch`. Port
# defaults to 8100, distinct from the Gateway's 8000. AGENT_API_KEY is a
# locally-generated bearer token (e.g. `openssl rand -hex 32`), not a
# Kubernetes Secret -- this service isn't deployed to the cluster.
AGENT_API_HOST = os.environ.get("AGENT_API_HOST", "0.0.0.0")
AGENT_API_PORT = int(os.environ.get("AGENT_API_PORT", "8100"))
AGENT_API_KEY = os.environ.get("AGENT_API_KEY")

# `watch` subcommand (cli.py): polls GET {INCIDENT_GATEWAY_URL}/incidents on
# this cadence -- 30s matches Alertmanager's group_interval used elsewhere
# in this project.
DEFAULT_WATCH_POLL_INTERVAL_SECONDS = float(os.environ.get("INCIDENT_PILOT_WATCH_POLL_INTERVAL_SECONDS", "30"))
DEFAULT_STATE_DIR = REPO_ROOT / "state"
DEFAULT_PROCESSED_INCIDENTS_FILE = DEFAULT_STATE_DIR / "processed_incidents.json"
