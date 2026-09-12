"""Manual end-to-end check for observe_and_decide against the demo banking app.

Starts the Flask app from demo_app/, opens the login page in a real browser,
and asks Claude for the single next action toward logging in. This makes a
live Anthropic API call, so it needs ANTHROPIC_API_KEY set - see the
module docstring at the bottom / README instructions for how to run it.

Not a unit test: it depends on a live LLM call and a real browser, so there
is nothing meaningful to assert beyond "we got a well-formed AgentAction
back". Run it directly rather than through a pytest suite.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_APP_DIR = REPO_ROOT / "demo_app"
HOST = "127.0.0.1"
PORT = 5000
LOGIN_URL = f"http://{HOST}:{PORT}/login"
STARTUP_TIMEOUT_SECONDS = 15

sys.path.insert(0, str(REPO_ROOT))

from src.agent import observe_and_decide  # noqa: E402


def _wait_for_server(url: str, timeout: float) -> None:
    # A bare TCP connect isn't enough here: the listening socket can accept
    # connections briefly before Flask is actually ready to route requests
    # (observed as a transient 403), so poll with a real HTTP GET instead.
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(
        f"Flask app at {url} did not become ready within {timeout}s "
        f"(last error: {last_error}). Check that demo_app/app.py runs cleanly on its own."
    )


def _start_flask_app() -> subprocess.Popen:
    # Runs app.run() directly instead of `python app.py` so debug=True's
    # reloader (which forks a second werkzeug process) never gets involved -
    # that second process would survive proc.terminate() and leak port 5000.
    return subprocess.Popen(
        [sys.executable, "-c", f"import app; app.app.run(host={HOST!r}, port={PORT})"],
        cwd=DEMO_APP_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _observe_login_page():
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(LOGIN_URL)
            return await observe_and_decide(
                page=page,
                goal="log in with username admin and password admin123",
                step_history=[],
            )
        finally:
            await browser.close()


def _run_observation():
    """Start Flask, observe the login page, print the action, tear down. Returns the AgentAction."""
    print(f"Starting Flask app from {DEMO_APP_DIR} ...")
    flask_process = _start_flask_app()
    try:
        _wait_for_server(LOGIN_URL, STARTUP_TIMEOUT_SECONDS)
        print(f"Flask app is up at {LOGIN_URL}")

        action = asyncio.run(_observe_login_page())

        print("\nAgentAction returned by observe_and_decide:")
        print(action.model_dump_json(indent=2))
        return action
    finally:
        print("\nShutting down Flask app ...")
        flask_process.terminate()
        try:
            flask_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            flask_process.kill()
            flask_process.wait()


def test_observe_and_decide():
    """Pytest entry point. Skips (rather than fails) if no API key is configured.

    Run with `-s` to see the printed AgentAction - pytest captures stdout by
    default and only surfaces it on failure.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set")
    action = _run_observation()
    assert action is not None


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. Export it before running this script, e.g.:\n"
            '  export ANTHROPIC_API_KEY="sk-ant-..."\n'
            "  python3 tests/test_observer.py"
        )
    _run_observation()


if __name__ == "__main__":
    main()
