"""FastAPI app for the investigation API. Serves whatever trajectory files
(trajectory/logger.py's output) happen to exist on disk -- written by
`run`, `watch`, or this app's own POST route, doesn't care which.

Wired into the same process as `watch` (see cli.py); reachable from `run`
purely because it reads the same trajectory directory `run` writes to.

Bearer-token auth (same pattern as GatewayContextProvider's client side,
just the server side of it here) gates every route except /health.
OpenAPI docs are disabled outright (docs_url=None etc.) rather than shipped
unauthenticated.

POST /investigations/{incident_id} is this API's first write action --
see that route's own docstring below for how it coexists with `watch`'s
own dispatch loop running in the same process.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Set

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import config
from ..context_provider.gateway_provider import GatewayContextProvider
from ..pipeline import _build_context_provider, _save_processed_incidents, run_incident
from . import reader
from .schemas import InvestigationDetail, InvestigationListItem, InvestigationTriggerResponse

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


async def _require_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> None:
    expected = request.app.state.api_key
    if not expected:
        raise HTTPException(status_code=500, detail="AGENT_API_KEY is not configured")
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(
    trajectory_dir: Path,
    api_key: Optional[str],
    *,
    llm_name: str = "fake",
    fixtures_root: Optional[Path] = None,
    max_iterations: Optional[int] = None,
    state_file: Optional[Path] = None,
    processed_incidents: Optional[Set[str]] = None,
) -> FastAPI:
    """`llm_name`/`fixtures_root`/`max_iterations` are whatever `watch` was
    itself started with (see cli.py's watch_incidents) -- a manual trigger
    runs with the same config `watch` would have used for that incident,
    not a separate one.

    `state_file`/`processed_incidents`: POST /investigations/{incident_id}
    marks an incident processed the same way `watch`'s own dispatch loop
    does, so `watch`'s next poll cycle doesn't redundantly re-dispatch an
    incident a manual trigger just started. `processed_incidents`, when
    passed, must be the *same* set object `watch`'s poll loop is holding a
    reference to (not a copy) -- watch_incidents passes its own `processed`
    set through for exactly this reason. Left as None (a fresh empty set)
    for standalone/test use, where there's no poll loop to coordinate
    with."""
    app = FastAPI(
        title="incident-pilot-agent investigation API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.trajectory_dir = trajectory_dir
    app.state.api_key = api_key
    app.state.llm_name = llm_name
    app.state.fixtures_root = fixtures_root or config.DEFAULT_FIXTURES_DIR
    app.state.max_iterations = max_iterations or config.DEFAULT_MAX_ITERATIONS
    app.state.state_file = state_file or config.DEFAULT_PROCESSED_INCIDENTS_FILE
    app.state.processed_incidents = processed_incidents if processed_incidents is not None else set()
    # Guards against two concurrent investigations of the same incident --
    # a double-clicked trigger, or a manual trigger landing in the same
    # window as `watch`'s own dispatch (before the processed-incidents
    # state file update above has had a chance to make watch's *next* poll
    # skip it). Checked/added/removed only by the POST route below.
    app.state.investigations_in_progress = set()
    # asyncio.create_task()'s result must be held onto by something, or the
    # task can be garbage-collected mid-run (a well-known asyncio footgun)
    # -- this set is that "something"; each task removes itself when done.
    app.state.background_tasks = set()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get(
        "/investigations",
        response_model=List[InvestigationListItem],
        dependencies=[Depends(_require_api_key)],
    )
    async def list_investigations(request: Request) -> List[InvestigationListItem]:
        return reader.list_investigations(request.app.state.trajectory_dir)

    @app.get(
        "/investigations/{incident_id}",
        response_model=InvestigationDetail,
        dependencies=[Depends(_require_api_key)],
    )
    async def get_investigation(incident_id: str, request: Request) -> InvestigationDetail:
        detail = reader.get_investigation(request.app.state.trajectory_dir, incident_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="No investigation found for this incident")
        return detail

    @app.post(
        "/investigations/{incident_id}",
        status_code=202,
        response_model=InvestigationTriggerResponse,
        dependencies=[Depends(_require_api_key)],
    )
    async def trigger_investigation(incident_id: str, request: Request) -> InvestigationTriggerResponse:
        """Manually trigger a real investigation -- the same run_incident()
        pipeline `run`/`watch` use (pipeline.py), not a third code path.
        Fires the graph run as a background asyncio task and returns 202
        immediately; a real investigation takes 30s-several minutes, far
        too long to hold an HTTP request open for.

        409s in two cases, both without starting anything: the incident is
        already mid-investigation (this route re-triggered, or watch's own
        dispatch loop got there first), or the Gateway's current_phase for
        this incident isn't ready_for_investigation yet (nothing complete
        enough to investigate against). A fixture-sourced incident_id (used
        by the test suite) has no Gateway current_phase to check at all --
        skipped in that case, same as `run` just runs it."""
        state = request.app.state

        if incident_id in state.investigations_in_progress:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "already_in_progress",
                    "message": "investigation already in progress for this incident",
                },
            )

        provider = _build_context_provider(incident_id, state.fixtures_root, "auto")
        if isinstance(provider, GatewayContextProvider):
            current_phase = await provider.get_current_phase(incident_id)
            if current_phase != "ready_for_investigation":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "not_ready",
                        "message": f"incident is not ready for investigation (current_phase={current_phase!r})",
                    },
                )

        state.investigations_in_progress.add(incident_id)
        # Marked processed immediately (not after the run finishes) so
        # watch's *next* poll cycle -- which reads this same set object,
        # see create_app's docstring -- doesn't also dispatch it.
        state.processed_incidents.add(incident_id)
        _save_processed_incidents(state.state_file, state.processed_incidents)

        async def _dispatch() -> None:
            try:
                await run_incident(
                    incident_id,
                    llm_name=state.llm_name,
                    fixtures_root=state.fixtures_root,
                    trajectory_dir=state.trajectory_dir,
                    max_iterations=state.max_iterations,
                    source="auto",
                )
            except Exception:
                logger.exception("manually triggered investigation failed for %s", incident_id)
            finally:
                state.investigations_in_progress.discard(incident_id)

        task = asyncio.create_task(_dispatch())
        state.background_tasks.add(task)
        task.add_done_callback(state.background_tasks.discard)

        return InvestigationTriggerResponse(incident_id=incident_id, status="investigation_started")

    return app
