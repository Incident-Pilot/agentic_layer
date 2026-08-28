import re
from datetime import datetime
from typing import List, Union

from pydantic import BaseModel, Field

from ..telemetry.fixture_backends import FixturePrometheusBackend
from ..telemetry.prometheus_client import PrometheusClient
from .base import Tool, ToolResult

PrometheusBackend = Union[PrometheusClient, FixturePrometheusBackend]

# A query that opens with a bare label selector -- e.g.
# `{namespace="x", pod=~"order-service.*"}` -- has no metric name in front
# of it, so PromQL matches it against every series of every metric that
# carries those labels, cluster-wide. Confirmed live: one such query
# returned 263,899 characters of series data in a single tool call, which
# is what pushed the next LLM call over the provider's prompt-size ceiling.
# `_series_summary`'s per-series compression doesn't help here -- there's
# nothing to bound the *number* of series a bare selector can match in the
# first place, so this is rejected before it ever reaches the backend.
_BARE_SELECTOR_PATTERN = re.compile(r"^\s*\{")

# Belt-and-suspenders cap for a different overly-broad-but-valid query (e.g.
# a real metric name with a missing pod filter) that still matches an
# unusually large number of series -- mirrors TempoTool._get_trace's
# span-count truncation (total_span_count/spans_shown).
_MAX_SERIES = 50


class PrometheusQueryInput(BaseModel):
    promql: str = Field(..., description="PromQL range-query expression, e.g. 'rate(container_cpu_usage_seconds_total{namespace=\"prod\",pod=~\"checkout.*\"}[5m])'")
    start: datetime = Field(..., description="Window start, ISO 8601 UTC")
    end: datetime = Field(..., description="Window end, ISO 8601 UTC")
    step: str = Field("30s", description="Resolution step, e.g. '30s'")


def _validate_promql(promql: str) -> None:
    if _BARE_SELECTOR_PATTERN.match(promql):
        raise ValueError(
            "PromQL query has no metric name before the label selector "
            "(e.g. '{namespace=\"x\"}') -- this matches every metric in "
            "the cluster and is not permitted. Specify a metric name, "
            "e.g. 'up{namespace=\"x\"}'."
        )


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
        query_summary = f"PromQL `{tool_input.promql}` over {tool_input.start.isoformat()}..{tool_input.end.isoformat()}"
        try:
            _validate_promql(tool_input.promql)
        except ValueError as exc:
            return ToolResult(tool_name=self.name, ok=False, error=str(exc), query_summary=query_summary)

        result = await self._backend.query_range(
            tool_input.promql, tool_input.start, tool_input.end, step=tool_input.step
        )
        if not result.ok:
            return ToolResult(tool_name=self.name, ok=False, error=result.error, query_summary=query_summary)

        all_series = _series_summary(result.data)
        # Most-changed series first, so a truncation still surfaces the
        # series most likely to matter rather than an arbitrary prefix.
        all_series.sort(key=lambda s: abs(s["last_value"] - s["first_value"]), reverse=True)
        shown_series = all_series[:_MAX_SERIES]
        return ToolResult(
            tool_name=self.name,
            ok=True,
            data={
                "series": shown_series,
                "total_series_count": len(all_series),
                "series_shown": len(shown_series),
            },
            query_summary=query_summary,
        )
