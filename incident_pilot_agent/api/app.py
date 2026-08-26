"""FastAPI app for the read-only investigation API. Serves whatever
trajectory files (trajectory/logger.py's output) happen to exist on disk --
written by `run` or `watch`, this app doesn't care which. No write
endpoints of any kind.

Wired into the same process as `watch` (see cli.py); reachable from `run`
purely because it reads the same trajectory directory `run` writes to.

Bearer-token auth (same pattern as GatewayContextProvider's client side,
just the server side of it here) gates every route except /health.
OpenAPI docs are disabled outright (docs_url=None etc.) rather than shipped
unauthenticated.
"""

from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import reader
from .schemas import InvestigationDetail, InvestigationListItem

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


def create_app(trajectory_dir: Path, api_key: Optional[str]) -> FastAPI:
    app = FastAPI(
        title="incident-pilot-agent investigation API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.trajectory_dir = trajectory_dir
    app.state.api_key = api_key

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

    return app
