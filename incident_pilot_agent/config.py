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

# Optional: point at a real read-only observability stack instead of
# fixtures. Unset by default -- FixtureContextProvider + fixture-backed
# tools are what this phase actually runs against.
PROMETHEUS_BASE_URL = os.environ.get("PROMETHEUS_BASE_URL")
LOKI_BASE_URL = os.environ.get("LOKI_BASE_URL")
TEMPO_BASE_URL = os.environ.get("TEMPO_BASE_URL")

DEFAULT_MAX_ITERATIONS = int(os.environ.get("INCIDENT_PILOT_MAX_ITERATIONS", "4"))
