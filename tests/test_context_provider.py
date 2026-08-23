from pathlib import Path

import pytest

from incident_pilot_agent.context_provider.fixture_provider import FixtureContextProvider
from incident_pilot_agent.context_provider.gateway_provider import GatewayContextProvider

FIXTURES_ROOT = Path(__file__).resolve().parent.parent / "fixtures" / "incidents"

ALL_INCIDENTS = [
    "inc-001-redis-cascade",
    "inc-002-db-pool-exhaustion",
    "inc-003-bad-deploy-crashloop",
]


@pytest.mark.parametrize("incident_id", ALL_INCIDENTS)
async def test_fixture_context_provider_loads_valid_context(incident_id):
    provider = FixtureContextProvider(FIXTURES_ROOT)
    context = await provider.get_context(incident_id)
    assert context.incident_id == incident_id
    assert context.affected_services


async def test_fixture_context_provider_missing_incident_raises():
    provider = FixtureContextProvider(FIXTURES_ROOT)
    with pytest.raises(FileNotFoundError):
        await provider.get_context("does-not-exist")


async def test_gateway_context_provider_not_implemented():
    provider = GatewayContextProvider(base_url="http://localhost:9999")
    with pytest.raises(NotImplementedError):
        await provider.get_context("inc-001-redis-cascade")
