"""Unit tests for agents/investigator.py's source-status-based tool
pruning (_select_tools / _empty_but_available_sources).

Confirmed pattern from real runs: query_loki and query_tempo returned
ok=False every time, matching the Gateway's own /source-status already
reporting observation_count=0 for those sources -- offering those tools
to the investigator anyway wastes tool-call turns. But a source being
AVAILABLE with observation_count=0 does not mean it will still be empty
by the time the investigator queries a possibly-different time window or
search term, so only a genuine UNAVAILABLE status (a real connectivity/
auth failure) is allowed to drop a tool outright -- see _select_tools's
own docstring for the full reasoning.
"""

from incident_pilot_agent.agents.investigator import _empty_but_available_sources, _select_tools
from incident_pilot_agent.models.context import IncidentContext, SourceAvailability, SourceStatusEntry
from incident_pilot_agent.tools.loki_tool import LokiTool
from incident_pilot_agent.tools.prometheus_tool import PrometheusTool
from incident_pilot_agent.tools.tempo_tool import TempoTool


def _context(source_status):
    return IncidentContext(
        incident_id="inc-test",
        title="test incident",
        detected_at="2026-08-24T09:00:00Z",
        source_status=source_status,
    )


def _all_tools():
    # _select_tools only ever inspects `.name` -- backend=None is fine,
    # execute() is never called by these tests.
    return [PrometheusTool(backend=None), LokiTool(backend=None), TempoTool(backend=None)]


def test_unavailable_source_drops_only_its_own_tool():
    context = _context(
        [
            SourceStatusEntry(source="loki", status=SourceAvailability.UNAVAILABLE, error="connection refused"),
            SourceStatusEntry(source="prometheus", status=SourceAvailability.AVAILABLE, observation_count=3),
            SourceStatusEntry(source="tempo", status=SourceAvailability.AVAILABLE, observation_count=1),
        ]
    )

    selected = _select_tools(_all_tools(), context)

    assert {t.name for t in selected} == {"query_prometheus", "query_tempo"}


def test_available_with_zero_observations_keeps_all_tools_offered():
    """Conservative scoping: observation_count == 0 alone must never drop
    a tool by itself -- only a genuine UNAVAILABLE status does. This is
    the exact shape seen in every real run tonight (loki/tempo AVAILABLE
    with observation_count 0)."""
    context = _context(
        [
            SourceStatusEntry(source="loki", status=SourceAvailability.AVAILABLE, observation_count=0),
            SourceStatusEntry(source="tempo", status=SourceAvailability.AVAILABLE, observation_count=0),
            SourceStatusEntry(source="prometheus", status=SourceAvailability.AVAILABLE, observation_count=3),
        ]
    )

    selected = _select_tools(_all_tools(), context)

    assert {t.name for t in selected} == {"query_prometheus", "query_loki", "query_tempo"}


def test_no_source_status_keeps_all_tools_offered():
    """Fixture-backed incidents (and any context predating this field)
    carry an empty source_status list by default -- must never be read as
    'nothing is available'."""
    selected = _select_tools(_all_tools(), _context([]))

    assert {t.name for t in selected} == {"query_prometheus", "query_loki", "query_tempo"}


def test_unavailable_sources_with_no_corresponding_tool_are_ignored():
    """alertmanager/kubernetes/deployment have no query_* tool -- their
    status must not affect pruning at all."""
    context = _context(
        [
            SourceStatusEntry(source="alertmanager", status=SourceAvailability.UNAVAILABLE, error="timeout"),
            SourceStatusEntry(source="kubernetes", status=SourceAvailability.UNAVAILABLE, error="rbac denied"),
            SourceStatusEntry(source="prometheus", status=SourceAvailability.AVAILABLE, observation_count=3),
        ]
    )

    selected = _select_tools(_all_tools(), context)

    assert {t.name for t in selected} == {"query_prometheus", "query_loki", "query_tempo"}


def test_empty_but_available_sources_lists_only_zero_observation_query_sources():
    context = _context(
        [
            SourceStatusEntry(source="loki", status=SourceAvailability.AVAILABLE, observation_count=0),
            SourceStatusEntry(source="tempo", status=SourceAvailability.AVAILABLE, observation_count=0),
            SourceStatusEntry(source="prometheus", status=SourceAvailability.AVAILABLE, observation_count=3),
            # AVAILABLE + zero observations, but no query_* tool exists for
            # it -- must not show up in the investigator's empty_sources hint.
            SourceStatusEntry(source="kubernetes", status=SourceAvailability.AVAILABLE, observation_count=0),
        ]
    )

    assert _empty_but_available_sources(context) == ["loki", "tempo"]


def test_empty_but_available_sources_excludes_unavailable():
    """An UNAVAILABLE source is pruned outright (see _select_tools) -- it
    must not also show up in the softer empty_sources hint, since its
    tool isn't even offered this round."""
    context = _context(
        [
            SourceStatusEntry(source="loki", status=SourceAvailability.UNAVAILABLE, error="connection refused"),
        ]
    )

    assert _empty_but_available_sources(context) == []
