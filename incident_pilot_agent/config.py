"""Local configuration. No Postgres/pgvector, no Kubernetes config here --
this repo runs entirely on a developer machine against fixtures (or,
optionally, a real read-only Prometheus/Loki/Tempo endpoint) per the
top-level spec's "local-first, no deployment in this phase" constraint.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURES_DIR = REPO_ROOT / "fixtures" / "incidents"
DEFAULT_TRAJECTORY_DIR = REPO_ROOT / "trajectories"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Optional: point at a real read-only observability stack instead of
# fixtures. Unset by default -- FixtureContextProvider + fixture-backed
# tools are what this phase actually runs against.
PROMETHEUS_BASE_URL = os.environ.get("PROMETHEUS_BASE_URL")
LOKI_BASE_URL = os.environ.get("LOKI_BASE_URL")
TEMPO_BASE_URL = os.environ.get("TEMPO_BASE_URL")

DEFAULT_MAX_ITERATIONS = int(os.environ.get("INCIDENT_PILOT_MAX_ITERATIONS", "4"))
