# Incident Pilot Agent

An autonomous incident-investigation agent. Given an incident (from a local
fixture or a live incident Gateway), it runs a LangGraph pipeline of
LLM-backed agents — Investigator, Synthesizer, Verifier, and a Remediation
Planner — that query telemetry (Prometheus/Loki/Tempo), form and verify a
root-cause hypothesis, and propose a remediation plan.

This repo is local-first: no Postgres/pgvector, no Kubernetes config, no
deployment target. It runs entirely on a developer machine against fixtures,
or optionally against a real read-only observability stack / incident
Gateway.

- **Repository:** https://github.com/vshwanilgv/agentic_layer
- **Main package:** [incident_pilot_agent/](incident_pilot_agent/)

## Repository layout

| Path | Purpose |
|---|---|
| [incident_pilot_agent/cli.py](incident_pilot_agent/cli.py) | CLI entrypoint: `run` (one incident) and `watch` (poll the Gateway forever) |
| [incident_pilot_agent/pipeline.py](incident_pilot_agent/pipeline.py) | Shared pipeline used by `run`, `watch`, and the API's trigger route |
| [incident_pilot_agent/config.py](incident_pilot_agent/config.py) | Env-driven configuration (loads `.env` via `python-dotenv`) |
| [incident_pilot_agent/graph/](incident_pilot_agent/graph/) | LangGraph state machine wiring the agents together |
| [incident_pilot_agent/agents/](incident_pilot_agent/agents/) | Investigator, Synthesizer, Verifier, Remediation Planner, orchestrator, prompts |
| [incident_pilot_agent/llm/](incident_pilot_agent/llm/) | LLM client adapters: Anthropic, OpenAI, Gemini, OpenRouter, Bedrock, and a fake client for tests |
| [incident_pilot_agent/context_provider/](incident_pilot_agent/context_provider/) | Loads `IncidentContext` from local fixtures or a live Gateway |
| [incident_pilot_agent/telemetry/](incident_pilot_agent/telemetry/) | Prometheus/Loki/Tempo clients (real and fixture-backed) |
| [incident_pilot_agent/tools/](incident_pilot_agent/tools/) | Tool wrappers over telemetry clients, exposed to the Investigator agent |
| [incident_pilot_agent/models/](incident_pilot_agent/models/) | Typed models: context, evidence, hypothesis, verification, remediation |
| [incident_pilot_agent/api/](incident_pilot_agent/api/) | Read-only FastAPI investigation API (serves a dashboard) |
| [incident_pilot_agent/trajectory/](incident_pilot_agent/trajectory/) | Trajectory logger — records each agent round to `trajectories/` |
| [fixtures/incidents/](fixtures/incidents/) | Local fixture incidents used for `--source fixtures` / `--llm fake` runs |
| [tests/](tests/) | Pytest suite |
| [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml) | Local containerized packaging |

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/vshwanilgv/agentic_layer.git
cd agentic_layer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the env template and fill in whichever provider(s) you'll use:

```bash
cp .env.example .env
```

`.env` is loaded automatically by [incident_pilot_agent/config.py](incident_pilot_agent/config.py) and is gitignored — never commit real keys.

Relevant variables (see [.env.example](.env.example) for the full, commented list):

- One of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, or `AWS_BEARER_TOKEN_BEDROCK` — required only for the corresponding `--llm` provider. Not needed for `--llm fake`.
- `INCIDENT_GATEWAY_URL` / `INCIDENT_GATEWAY_API_KEY` — optional, point at a real incident Gateway instead of local fixtures (see [Talking to the Gateway](#talking-to-the-gateway) below).
- `PROMETHEUS_BASE_URL` / `LOKI_BASE_URL` / `TEMPO_BASE_URL` — optional, point at a real read-only observability stack instead of fixture-backed telemetry.
- `AGENT_API_KEY` — bearer token for this repo's own investigation API (generate with `openssl rand -hex 32`); required to serve the API during `watch`.

## Running

### Run a single incident (fixture data, no API keys needed)

```bash
python -m incident_pilot_agent run inc-001-redis-cascade --llm fake
```

Available fixture incidents: `inc-001-redis-cascade`, `inc-002-db-pool-exhaustion`, `inc-003-bad-deploy-crashloop` (see [fixtures/incidents/](fixtures/incidents/)).

Prints the full agent trajectory and final root-cause verdict, and writes a
trajectory file to `trajectories/`.

Useful flags:

```bash
python -m incident_pilot_agent run <incident_id> \
  --llm {fake,anthropic,openai,gemini,openrouter,bedrock} \
  --fixtures-dir <path>       # default: fixtures/incidents
  --trajectory-dir <path>     # default: trajectories/
  --max-iterations <n>        # default: 4
  --source {auto,fixtures,gateway}   # default: auto
```

With a real LLM, e.g. Anthropic:

```bash
python -m incident_pilot_agent run inc-001-redis-cascade --llm anthropic
```

### Watch mode (poll the Gateway, long-running)

Requires `INCIDENT_GATEWAY_URL` and `INCIDENT_GATEWAY_API_KEY` to be set.
Polls the Gateway every 30s (configurable) for incidents in phase
`ready_for_investigation` and investigates each new one automatically:

```bash
python -m incident_pilot_agent watch --llm anthropic
```

If `AGENT_API_KEY` is set, `watch` also serves the read-only investigation
API (below) in the same process for as long as it runs.

### Via Docker

```bash
cp .env.docker.example .env.docker   # fill in real values
docker compose --env-file .env.docker up --build
```

Defaults to `watch`. For a one-shot run instead:

```bash
docker run <image> run <incident_id> --source fixture
```

On macOS/Windows, reach a Gateway running via `kubectl port-forward` with
`INCIDENT_GATEWAY_URL=http://host.docker.internal:8000` in `.env.docker`. On
Linux, uncomment `network_mode: host` in [docker-compose.yml](docker-compose.yml) and use `http://localhost:8000` instead.

## Communicating with Incident Pilot

There are two directions of communication: this agent *pulling* incidents
from a Gateway, and other services *querying* this agent's own investigation
results.

### Talking to the Gateway (inbound incident data)

This repo does not receive webhooks. Instead, `watch` polls
`GET {INCIDENT_GATEWAY_URL}/incidents` (bearer-authenticated with
`INCIDENT_GATEWAY_API_KEY`) every `INCIDENT_PILOT_WATCH_POLL_INTERVAL_SECONDS`
(default 30s, matching Alertmanager's `group_interval`), filters
client-side for `current_phase == "ready_for_investigation"`, and dispatches
each new incident through the same pipeline as `run`. See
[incident_pilot_agent/context_provider/gateway_provider.py](incident_pilot_agent/context_provider/gateway_provider.py)
for exactly which Gateway endpoints are called
(`/incidents/{id}`, `/evidence`, `/source-status`, `/timeline`, `/topology`).
Already-processed incidents are tracked in `state/processed_incidents.json`
so a restart doesn't re-investigate them.

### Talking to this agent's investigation API (outbound results)

When `watch` runs with `AGENT_API_KEY` set, it serves a read-only FastAPI
app on `http://<AGENT_API_HOST>:<AGENT_API_PORT>` (default `0.0.0.0:8100`).
Every route except `/health` requires `Authorization: Bearer <AGENT_API_KEY>`.
OpenAPI docs are disabled; see [incident_pilot_agent/api/app.py](incident_pilot_agent/api/app.py) for the source of truth.

```bash
# Health check (no auth)
curl http://localhost:8100/health

# List all investigations
curl -H "Authorization: Bearer $AGENT_API_KEY" \
  http://localhost:8100/investigations

# Get one investigation's detail
curl -H "Authorization: Bearer $AGENT_API_KEY" \
  http://localhost:8100/investigations/inc-001-redis-cascade

# Manually trigger an investigation (202 Accepted, runs in the background)
curl -X POST -H "Authorization: Bearer $AGENT_API_KEY" \
  http://localhost:8100/investigations/inc-001-redis-cascade
```

The `POST` route 409s if the incident is already mid-investigation, or (for
Gateway-sourced incidents) if its `current_phase` isn't yet
`ready_for_investigation`.

## Tests

```bash
pytest
```

Uses `pytest-asyncio` in auto mode (see [pytest.ini](pytest.ini)) and the fake LLM client — no API keys required.
