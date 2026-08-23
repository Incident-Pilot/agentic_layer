from datetime import datetime
from typing import List, Union

from pydantic import BaseModel, Field

from ..telemetry.fixture_backends import FixturePrometheusBackend
from ..telemetry.prometheus_client import PrometheusClient
from .base import Tool, ToolResult

PrometheusBackend = Union[PrometheusClient, FixturePrometheusBackend]


class PrometheusQueryInput(BaseModel):
    promql: str = Field(..., description="PromQL range-query expression, e.g. 'rate(container_cpu_usage_seconds_total{namespace=\"prod\",pod=~\"checkout.*\"}[5m])'")
    start: datetime = Field(..., description="Window start, ISO 8601 UTC")
    end: datetime = Field(..., description="Window end, ISO 8601 UTC")
    step: str = Field("30s", description="Resolution step, e.g. '30s'")


def _series_summary(matrix_data: dict) -> List[dict]:
    """Reduce a raw Prometheus matrix result down to one summary row per
    series: labels, first/last sample, min/max, so an LLM doesn't need to
    parse thousands of raw [timestamp, value] pairs to see the trend."""
    summaries = []
    for series in (matrix_data or {}).get("result", []) or []:
        values = series.get("values") or []
        if not values:
            continue
        floats = []
        for _, v in values:
            try:
                floats.append(float(v))
            except (TypeError, ValueError):
                continue
        if not floats:
            continue
        first, last = floats[0], floats[-1]
        trend = "rising" if last > first * 1.1 else "falling" if last < first * 0.9 else "stable"
        summaries.append(
            {
                "labels": series.get("metric", {}),
                "first_value": first,
                "last_value": last,
                "min_value": min(floats),
                "max_value": max(floats),
                "trend": trend,
                "sample_count": len(floats),
            }
        )
    return summaries


class PrometheusTool(Tool):
    name = "query_prometheus"
    description = (
        "Run a read-only PromQL range query against Prometheus for the incident window. "
        "Returns a per-series summary (labels, first/last/min/max value, trend), not raw samples."
    )
    input_schema = PrometheusQueryInput

    def __init__(self, backend: PrometheusBackend):
        self._backend = backend

    async def execute(self, tool_input: PrometheusQueryInput) -> ToolResult:
        result = await self._backend.query_range(
            tool_input.promql, tool_input.start, tool_input.end, step=tool_input.step
        )
        query_summary = f"PromQL `{tool_input.promql}` over {tool_input.start.isoformat()}..{tool_input.end.isoformat()}"
        if not result.ok:
            return ToolResult(tool_name=self.name, ok=False, error=result.error, query_summary=query_summary)
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={"series": _series_summary(result.data)},
            query_summary=query_summary,
        )
