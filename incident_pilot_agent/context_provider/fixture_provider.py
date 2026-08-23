"""FixtureContextProvider — loads IncidentContext from local JSON fixtures
under fixtures/incidents/<incident_id>/context.json. This is the only
ContextProvider implementation this phase actually runs against."""

import json
from pathlib import Path

from ..models.context import IncidentContext
from .base import ContextProvider


class FixtureContextProvider(ContextProvider):
    def __init__(self, fixtures_root: Path):
        self._fixtures_root = fixtures_root

    async def get_context(self, incident_id: str) -> IncidentContext:
        context_path = self._fixtures_root / incident_id / "context.json"
        if not context_path.exists():
            available = sorted(p.name for p in self._fixtures_root.iterdir() if p.is_dir())
            raise FileNotFoundError(
                f"No fixture context for incident_id={incident_id!r} at {context_path}. "
                f"Available fixture incidents: {available}"
            )
        with context_path.open() as f:
            raw = json.load(f)
        return IncidentContext.model_validate(raw)

    def incident_dir(self, incident_id: str) -> Path:
        """Directory holding this incident's context.json plus its
        prometheus.json/loki.json/tempo_*.json telemetry fixtures — used
        by the tool layer's fixture backends, not by get_context itself."""
        return self._fixtures_root / incident_id
