"""Fixture-backed stand-ins for PrometheusClient / LokiClient / TempoClient.

These return data in the exact raw wire shape the real backends return
(Prometheus `query_range` matrix data, Loki `streams` data, Tempo
`/api/search` and `/api/traces/{id}` OTLP data) so the *same* parsing code
in loki_client.py / tempo_client.py runs unmodified against fixtures and
against a live cluster — the fixture path exercises the real parser, it
doesn't reimplement a shortcut version of it.

Query matching is deliberately simple: each fixture file groups canned
responses into named "buckets", and a backend picks the first bucket whose
name appears as a substring of the incoming query string (case-insensitive),
falling back to a "default" bucket if present. This is not a PromQL/LogQL
interpreter — it's just enough realism for an agent's tool calls to get
query-appropriate data back without this repo needing to embed one.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .adapter_result import AdapterResult, SourceStatus


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _pick_bucket(buckets: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
    query_lower = query.lower()
    for name, payload in buckets.items():
        if name != "default" and name.lower() in query_lower:
            return payload
    return buckets.get("default")


class FixturePrometheusBackend:
    """Duck-type-compatible with PrometheusClient's query_range signature."""

    def __init__(self, fixtures_dir: Path):
        self._buckets: Dict[str, Any] = _load_json(fixtures_dir / "prometheus.json").get("buckets", {})

    async def query_range(self, promql: str, start, end, step: str = "30s") -> AdapterResult[Dict[str, Any]]:
        bucket = _pick_bucket(self._buckets, promql)
        if bucket is None:
            return AdapterResult(status=SourceStatus.AVAILABLE, data={"resultType": "matrix", "result": []})
        return AdapterResult(status=SourceStatus.AVAILABLE, data=bucket)

    async def query(self, promql: str, at=None) -> AdapterResult[Dict[str, Any]]:
        return await self.query_range(promql, at, at)


class FixtureLokiBackend:
    """Duck-type-compatible with LokiClient's query_range signature."""

    def __init__(self, fixtures_dir: Path):
        self._buckets: Dict[str, Any] = _load_json(fixtures_dir / "loki.json").get("buckets", {})

    async def query_range(
        self, logql: str, start, end, limit: int = 1000, direction: str = "backward"
    ) -> AdapterResult[Dict[str, Any]]:
        bucket = _pick_bucket(self._buckets, logql)
        if bucket is None:
            return AdapterResult(status=SourceStatus.AVAILABLE, data={"resultType": "streams", "result": []})
        return AdapterResult(status=SourceStatus.AVAILABLE, data=bucket)

    async def query(self, logql: str, limit: int = 100, at=None) -> AdapterResult[Dict[str, Any]]:
        return await self.query_range(logql, at, at, limit=limit)


class FixtureTempoBackend:
    """Duck-type-compatible with TempoClient's search/get_trace signatures."""

    def __init__(self, fixtures_dir: Path):
        self._search_buckets: Dict[str, Any] = _load_json(fixtures_dir / "tempo_search.json").get("buckets", {})
        self._traces: Dict[str, Any] = _load_json(fixtures_dir / "tempo_traces.json")

    async def search(self, params: Dict[str, Any]) -> AdapterResult[Dict[str, Any]]:
        query = " ".join(str(v) for v in params.values())
        bucket = _pick_bucket(self._search_buckets, query)
        if bucket is None:
            return AdapterResult(status=SourceStatus.AVAILABLE, data={"traces": []})
        return AdapterResult(status=SourceStatus.AVAILABLE, data=bucket)

    async def get_trace(self, trace_id: str) -> AdapterResult[Dict[str, Any]]:
        trace = self._traces.get(trace_id)
        if trace is None:
            return AdapterResult(status=SourceStatus.UNAVAILABLE, error=f"no fixture trace for id {trace_id!r}")
        return AdapterResult(status=SourceStatus.AVAILABLE, data=trace)
