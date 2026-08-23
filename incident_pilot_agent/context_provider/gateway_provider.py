"""GatewayContextProvider — extension point for when the real
incident-pilot-ecommerce Gateway, its Normalizer, and its Incident Context
Builder are ready.

NOT IMPLEMENTED. The Gateway's API is still under active development and
its nine incident endpoints are not final (see app/api/incidents.py in
that repo: GET /incidents, /incidents/{id}, /incidents/{id}/observations,
/incidents/{id}/evidence, /incidents/{id}/source-status,
/incidents/{id}/timeline). When that stabilizes, implement get_context()
here to call it (likely GET /incidents/{id} plus the observations/evidence/
timeline endpoints) and map its Observation/Evidence records onto this
repo's IncidentContext shape (models/context.py) — including setting
`provenance` per item (structured Kubernetes/deployment/Alertmanager data
-> TRUSTED, raw log/trace content -> UNTRUSTED, matching FixtureContextProvider's
fixtures). No other agent code should change: everything upstream consumes
the ContextProvider interface, not this class directly.
"""

from ..models.context import IncidentContext
from .base import ContextProvider


class GatewayContextProvider(ContextProvider):
    def __init__(self, base_url: str):
        self._base_url = base_url

    async def get_context(self, incident_id: str) -> IncidentContext:
        raise NotImplementedError(
            "GatewayContextProvider is not implemented yet: incident-pilot-ecommerce's "
            "Normalizer and Incident Context Builder are still under active development "
            "and their API is not stable. Use FixtureContextProvider until that lands."
        )
