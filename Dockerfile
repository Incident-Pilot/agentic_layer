# Local containerized packaging only -- no AWS deployment artifacts here.
# Base matches incident-pilot-ecommerce's services (see e.g.
# services/observation-gateway/Dockerfile) -- no reason to diverge.
FROM python:3.11-slim

WORKDIR /app

# requirements.txt copied (and installed) before the rest of the source so
# code-only changes don't invalidate the dependency-install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Same non-root pattern as the Gateway's Dockerfile. /app is chowned to this
# user because `run`/`watch` create trajectories/ and state/ under
# REPO_ROOT (== /app here) at runtime -- see config.py.
RUN useradd --no-create-home --uid 10001 agent \
    && chown -R agent:agent /app
USER agent

ENV PYTHONUNBUFFERED=1

# Matches AGENT_API_PORT's default (config.py). If you override
# AGENT_API_PORT, publish the new port yourself (EXPOSE here is documentation
# only, not enforcement).
EXPOSE 8100

# Default to the long-running service. Override the command for one-shot
# manual testing, e.g.:
#   docker run <image> run <incident_id> --source fixture
#   docker run <image> run <incident_id> --source gateway
ENTRYPOINT ["python", "-m", "incident_pilot_agent"]
CMD ["watch"]
