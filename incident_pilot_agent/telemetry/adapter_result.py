"""Common result type for every telemetry client (Prometheus, Loki, Tempo).

Ported from incident-pilot-ecommerce's app/collectors/base.py: a backend
being temporarily unavailable must degrade to a typed error, not an
exception, so an agent's tool call gets a clean signal to work around
rather than a stack trace.
"""

from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class SourceStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    PARTIAL = "partial"


class AdapterResult(BaseModel, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: SourceStatus
    data: Optional[T] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == SourceStatus.AVAILABLE
