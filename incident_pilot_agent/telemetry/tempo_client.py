"""Tempo client — ported verbatim (transport, OTLP-JSON parsing, and the
base64-trace-ID normalization) from incident-pilot-ecommerce's
app/collectors/tempo_adapter.py. See that file's original docstring for the
confirmed-against-live-cluster response-shape notes (OTLP `batches` with
base64 trace/span IDs vs. legacy Jaeger `data` shape; typed-union
attribute lists). This is a dumb transport/parse layer — no reasoning here.

Span tags/attributes are UNTRUSTED data (see models/context.py Provenance):
they're exporter/application-supplied strings, never instructions."""

import base64
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, ConfigDict

from .adapter_result import AdapterResult, SourceStatus


class Span(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    service: Optional[str] = None
    operation: str
    start_time: datetime
    duration_ms: float
    status: str  # "ok" or "error"
    tags: Dict[str, Any] = {}


class TraceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    root_service: Optional[str] = None
    root_operation: Optional[str] = None
    start_time: Optional[datetime] = None
    duration_ms: Optional[float] = None


def _b64_to_hex(value: str) -> str:
    try:
        return base64.b64decode(value).hex()
    except (ValueError, TypeError):
        return value


def _attr_value(value: Dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "arrayValue" in value:
        return [_attr_value(v) for v in value["arrayValue"].get("values", []) or []]
    if "kvlistValue" in value:
        return {
            kv.get("key"): _attr_value(kv.get("value", {}))
            for kv in value["kvlistValue"].get("values", []) or []
            if "key" in kv
        }
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def _attrs_to_dict(attributes: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for attr in attributes or []:
        key = attr.get("key")
        if key is None:
            continue
        result[key] = _attr_value(attr.get("value", {}) or {})
    return result


def _is_error_span(tags: Dict[str, Any], status_obj: Dict[str, Any]) -> bool:
    status_code = (status_obj or {}).get("code")
    if status_code in ("STATUS_CODE_ERROR", 2):
        return True
    if bool(tags.get("error")):
        return True
    try:
        return int(tags.get("http.status_code")) >= 400
    except (TypeError, ValueError):
        return False


class TempoClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 5.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._injected_client = client

    async def get_trace(self, trace_id: str) -> AdapterResult[Dict[str, Any]]:
        if not trace_id or not trace_id.strip():
            raise ValueError("trace_id is required")
        return await self._get(f"/api/traces/{trace_id}", {})

    async def search(self, params: Dict[str, Any]) -> AdapterResult[Dict[str, Any]]:
        return await self._get("/api/search", params)

    @staticmethod
    def parse_spans(data: Dict[str, Any]) -> List[Span]:
        if not data:
            return []
        if "batches" in data:
            return TempoClient._parse_spans_otlp(data)
        if "data" in data:
            return TempoClient._parse_spans_jaeger(data)
        return []

    @staticmethod
    def _parse_spans_otlp(data: Dict[str, Any]) -> List[Span]:
        spans: List[Span] = []
        for batch in data.get("batches", []) or []:
            resource = batch.get("resource", {}) or {}
            resource_attrs = _attrs_to_dict(resource.get("attributes"))
            service = resource_attrs.get("service.name")

            spans_key = "scopeSpans" if "scopeSpans" in batch else "instrumentationLibrarySpans"
            for scope_span in batch.get(spans_key, []) or []:
                for raw_span in scope_span.get("spans", []) or []:
                    try:
                        span_id = _b64_to_hex(raw_span["spanId"])
                        trace_id = _b64_to_hex(raw_span.get("traceId", ""))
                        operation = raw_span.get("name", "")
                        start_ns = int(raw_span["startTimeUnixNano"])
                        end_ns = int(raw_span.get("endTimeUnixNano", start_ns))
                        start_time = datetime.fromtimestamp(start_ns / 1_000_000_000, tz=timezone.utc)
                        duration_ms = (end_ns - start_ns) / 1_000_000
                    except (KeyError, TypeError, ValueError, OverflowError):
                        continue

                    raw_parent = raw_span.get("parentSpanId")
                    parent_span_id = _b64_to_hex(raw_parent) if raw_parent else None

                    tags = _attrs_to_dict(raw_span.get("attributes"))
                    status_obj = raw_span.get("status") or {}
                    status = "error" if _is_error_span(tags, status_obj) else "ok"

                    spans.append(
                        Span(
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_span_id=parent_span_id,
                            service=service,
                            operation=operation,
                            start_time=start_time,
                            duration_ms=duration_ms,
                            status=status,
                            tags=tags,
                        )
                    )
        return spans

    @staticmethod
    def _parse_spans_jaeger(data: Dict[str, Any]) -> List[Span]:
        spans: List[Span] = []
        for trace in data.get("data", []) or []:
            processes: Dict[str, Any] = trace.get("processes", {}) or {}
            for raw_span in trace.get("spans", []) or []:
                try:
                    span_id = raw_span["spanID"]
                    trace_id = raw_span.get("traceID", trace.get("traceID", ""))
                    operation = raw_span.get("operationName", "")
                    start_time = datetime.fromtimestamp(raw_span["startTime"] / 1_000_000, tz=timezone.utc)
                    duration_ms = raw_span.get("duration", 0) / 1_000
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue

                process_id = raw_span.get("processID")
                service = None
                if process_id and process_id in processes:
                    service = processes[process_id].get("serviceName")

                parent_span_id = None
                for ref in raw_span.get("references", []) or []:
                    if ref.get("refType") == "CHILD_OF":
                        parent_span_id = ref.get("spanID")
                        break

                tags = {t.get("key"): t.get("value") for t in raw_span.get("tags", []) or [] if "key" in t}
                status = "error" if _is_error_span(tags, {}) else "ok"

                spans.append(
                    Span(
                        trace_id=trace_id,
                        span_id=span_id,
                        parent_span_id=parent_span_id,
                        service=service,
                        operation=operation,
                        start_time=start_time,
                        duration_ms=duration_ms,
                        status=status,
                        tags=tags,
                    )
                )
        return spans

    @staticmethod
    def parse_search_results(data: Dict[str, Any]) -> List[TraceSummary]:
        summaries: List[TraceSummary] = []
        for raw in data.get("traces", []) or []:
            trace_id = raw.get("traceID")
            if not trace_id:
                continue

            start_time = None
            raw_start = raw.get("startTimeUnixNano")
            if raw_start is not None:
                try:
                    start_time = datetime.fromtimestamp(int(raw_start) / 1_000_000_000, tz=timezone.utc)
                except (ValueError, TypeError, OverflowError):
                    start_time = None

            duration_ms = raw.get("durationMs")

            summaries.append(
                TraceSummary(
                    trace_id=trace_id,
                    root_service=raw.get("rootServiceName"),
                    root_operation=raw.get("rootTraceName"),
                    start_time=start_time,
                    duration_ms=float(duration_ms) if duration_ms is not None else None,
                )
            )
        return summaries

    async def _get(self, path: str, params: Dict[str, Any]) -> AdapterResult[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(timeout=self.timeout_seconds)

        try:
            try:
                response = await client.get(url, params=params)
                if response.status_code == 404:
                    return AdapterResult(
                        status=SourceStatus.UNAVAILABLE, error=f"Tempo returned 404 (trace/route not found) for {url}"
                    )
                response.raise_for_status()
                payload = response.json()
            except httpx.TimeoutException as exc:
                return AdapterResult(status=SourceStatus.TIMEOUT, error=f"Tempo request timed out: {exc}")
            except httpx.HTTPStatusError as exc:
                return AdapterResult(
                    status=SourceStatus.UNAVAILABLE,
                    error=f"Tempo returned HTTP {exc.response.status_code}: {exc.response.text[:300]}",
                )
            except httpx.RequestError as exc:
                return AdapterResult(
                    status=SourceStatus.UNAVAILABLE, error=f"Could not reach Tempo at {self.base_url}: {exc}"
                )
        finally:
            if owns_client:
                await client.aclose()

        return AdapterResult(status=SourceStatus.AVAILABLE, data=payload)
