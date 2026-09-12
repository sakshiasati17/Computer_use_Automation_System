"""Minimal FastAPI server the human operator uses during an escalation.

Runs embedded in the same asyncio event loop as the replay engine and the
Playwright browser session (`EscalationHub.ensure_server_started` schedules
it as a task on the caller's running loop, rather than spinning up its
own). Every route below is `async def` for that reason, not just style:
FastAPI executes `async def` handlers directly on the loop that received the
request, so state on the shared `EscalationHub` is only ever touched from
one thread, with no locking required.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from .escalation import ControlState, InterventionRequest, get_hub

__all__ = ["app"]

REPO_ROOT = Path(__file__).resolve().parents[2]
OPERATOR_UI_PATH = REPO_ROOT / "operator_ui" / "index.html"

app = FastAPI(title="Automation Operator Console")


class InterventionStatus(BaseModel):
    """Response shape for every endpoint below: the current pause (if any) plus who's in control."""

    intervention: InterventionRequest | None
    control_state: ControlState


def _status() -> InterventionStatus:
    hub = get_hub()
    return InterventionStatus(intervention=hub.current_intervention, control_state=hub.control_state)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not OPERATOR_UI_PATH.exists():
        raise HTTPException(status_code=500, detail=f"operator UI not found at {OPERATOR_UI_PATH}")
    return HTMLResponse(OPERATOR_UI_PATH.read_text())


@app.get("/api/intervention", response_model=InterventionStatus)
async def get_intervention() -> InterventionStatus:
    return _status()


@app.get("/api/screenshot")
async def get_screenshot() -> FileResponse:
    hub = get_hub()
    if hub.current_screenshot_path is None or not hub.current_screenshot_path.exists():
        raise HTTPException(status_code=404, detail="no screenshot available")
    return FileResponse(hub.current_screenshot_path, media_type="image/png")


@app.post("/api/take-control", response_model=InterventionStatus)
async def take_control() -> InterventionStatus:
    hub = get_hub()
    if hub.current_intervention is None:
        raise HTTPException(status_code=409, detail="no active intervention to take control of")
    if hub.control_state is not ControlState.PAUSED:
        raise HTTPException(status_code=409, detail=f"cannot take control from state {hub.control_state.value!r}")
    hub.take_control()
    return _status()


@app.post("/api/resume", response_model=InterventionStatus)
async def resume() -> InterventionStatus:
    hub = get_hub()
    if hub.current_intervention is None:
        raise HTTPException(status_code=409, detail="no active intervention to resume from")
    hub.signal_resume()
    return _status()
