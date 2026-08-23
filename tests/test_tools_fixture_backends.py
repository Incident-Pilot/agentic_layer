from datetime import datetime, timezone
from pathlib import Path

from incident_pilot_agent.telemetry.fixture_backends import (
    FixtureLokiBackend,
    FixturePrometheusBackend,
    FixtureTempoBackend,
)
from incident_pilot_agent.tools.loki_tool import LokiQueryInput, LokiTool
from incident_pilot_agent.tools.prometheus_tool import PrometheusQueryInput, PrometheusTool
from incident_pilot_agent.tools.tempo_tool import TempoQueryInput, TempoTool

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "incidents" / "inc-001-redis-cascade"

START = datetime(2026, 8, 22, 14, 0, tzinfo=timezone.utc)
END = datetime(2026, 8, 22, 14, 10, tzinfo=timezone.utc)


async def test_prometheus_tool_returns_series_summary_with_trend():
    tool = PrometheusTool(FixturePrometheusBackend(FIXTURE_DIR))
    result = await tool.execute(
        PrometheusQueryInput(promql='cpu_usage_seconds{namespace="cloudmart-prod"}', start=START, end=END)
    )
    assert result.ok
    series = result.data["series"]
    assert len(series) == 1
    assert series[0]["trend"] == "rising"
    assert series[0]["labels"]["__name__"] == "cpu_usage_seconds"


async def test_prometheus_tool_unmatched_query_falls_back_to_default_empty_series():
    tool = PrometheusTool(FixturePrometheusBackend(FIXTURE_DIR))
    result = await tool.execute(PrometheusQueryInput(promql="totally_unknown_metric", start=START, end=END))
    assert result.ok
    assert result.data["series"] == []


async def test_loki_tool_broad_query_excludes_redis_bucket():
    tool = LokiTool(FixtureLokiBackend(FIXTURE_DIR))
    result = await tool.execute(
        LokiQueryInput(logql='{namespace="cloudmart-prod"} |~ "(?i)error|exception|timeout|fail"', start=START, end=END)
    )
    assert result.ok
    messages = [e["message"] for e in result.data["entries"]]
    assert any("500 Internal Server Error" in m for m in messages)
    assert not any("redis" in m.lower() for m in messages)


async def test_loki_tool_targeted_redis_query_finds_smoking_gun():
    tool = LokiTool(FixtureLokiBackend(FIXTURE_DIR))
    result = await tool.execute(
        LokiQueryInput(logql='{namespace="cloudmart-prod"} |~ "(?i)redis"', start=START, end=END)
    )
    assert result.ok
    messages = [e["message"] for e in result.data["entries"]]
    assert any("ConnectionError" in m and "redis" in m.lower() for m in messages)
    # entries parsed via the ported LokiClient.parse_entries -- confirms
    # timestamps round-trip through real epoch-nanosecond parsing.
    assert all(e["timestamp"] for e in result.data["entries"])


async def test_tempo_tool_search_then_get_trace_round_trip():
    backend = FixtureTempoBackend(FIXTURE_DIR)
    tool = TempoTool(backend)

    search_result = await tool.execute(
        TempoQueryInput(operation="search", service_name="checkout-service", start=START, end=END)
    )
    assert search_result.ok
    traces = search_result.data["traces"]
    assert len(traces) == 1
    trace_id = traces[0]["trace_id"]

    trace_result = await tool.execute(TempoQueryInput(operation="get_trace", trace_id=trace_id))
    assert trace_result.ok
    spans = trace_result.data["spans"]
    assert len(spans) == 1
    assert spans[0]["service"] == "checkout-service"
    assert spans[0]["status"] == "error"
    # base64 -> hex decoding (ported from the OTLP parser) must agree with
    # the hex trace ID returned by search().
    assert spans[0]["trace_id"] == trace_id
