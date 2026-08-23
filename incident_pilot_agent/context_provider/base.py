"""ContextProvider — the one seam between this repo and the (not-yet-built)
Gateway/Normalizer/Context Builder. No agent code may import the Gateway,
a database, or any telemetry backend directly to build an IncidentContext;
everything goes through this interface, so swapping FixtureContextProvider
for GatewayContextProvider later is a one-line change at the call site.
"""

from abc import ABC, abstractmethod

from ..models.context import IncidentContext


class ContextProvider(ABC):
    @abstractmethod
    async def get_context(self, incident_id: str) -> IncidentContext: ...
