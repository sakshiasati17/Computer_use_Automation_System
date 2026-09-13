# Computer-Use Automation System

A two-stage browser automation system built for the interface.ai take-home assignment. **Discovery** uses Claude (with vision) to explore an unfamiliar web app and record a natural-language goal as a structured, reusable artifact. **Replay** then executes that artifact deterministically against a live browser — no LLM calls, no non-determinism — so the same flow can be run again and again at production scale, safely, cheaply, and predictably. The split exists because the two stages have opposite requirements: discovery must *understand* a page it has never seen; replay must be fast and reliable enough to run unattended against regulated data.

## Architecture Overview

- **`src/agent/`** — Discovery loop. `observer.py` makes one observe-then-decide call to Claude per step (screenshot + accessibility snapshot in, a validated `AgentAction` out); `discovery.py` owns the Playwright browser and drives the observe → act → log loop until the goal completes, gets stuck, or hits `max-steps`.
- **`src/artifacts/`** — `emitter.py` converts a successful discovery run into a schema-validated, parameterized `Artifact` saved under `saved_artifacts/`.
- **`src/models/`** — Pydantic schema for artifacts, actions, and execution results, including the locator fallback chain and the three-way outcome taxonomy (`success` / `business_outcome` / `hard_failure`).
- **`src/replay/`** — `engine.py` walks a saved artifact's steps with **zero LLM calls**, resolving each step's locator fallback chain against the live DOM and substituting `{{input.x}}` parameters.
- **`src/safety/`** — Domain/action allowlist, a risk classifier (`safe`/`risky`/`irreversible`), and a redactor (SSNs, account numbers, passwords, etc.) applied identically on both the discovery and replay paths before anything is written to disk.
- **`src/handoff/`** — Human escalation for the replay path: pauses on risky/irreversible steps or unresolved failures and serves a small operator console (`operator_ui/`) so a human can approve, or take control of, the same live browser window.
- **`src/observability/`** — Shared `RunLogger`/`RunEvidence` used by both stages to write structured, redacted evidence to `evidence/`.
- **`demo_app/`** — A small mock credit-union member portal (Flask) used as the target app for both discovery and replay demos.

See [REPORT.md](REPORT.md) for the full design write-up and rationale.

## Prerequisites

- Python 3.10+
- Playwright (Chromium browser)
- An Anthropic API key — **only needed for discovery**; replay runs saved artifacts deterministically and needs no API key

## Quick Start

```bash
git clone https://github.com/sakshiasati17/Computer_use_Automation_System.git
cd Computer_use_Automation_System
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Demo Path

**a. Start the demo app** (in its own terminal — it must keep running for discovery/replay):

```bash
cd demo_app && python app.py
# Serves the mock member portal at http://127.0.0.1:5000
```

**b. Run discovery** — Claude explores the app live from the goal text alone and, on success, writes a new artifact to `saved_artifacts/` plus a step-by-step log and screenshots to `evidence/discovery/`:

```bash
python -m src.main discover \
  --goal "Log in to the member portal using username 'admin' and password 'admin123'. Then find member M-1001 and open a new Savings sub-account for them with a .00 initial deposit, confirming creation." \
  --target http://127.0.0.1:5000/login
```

**c. Run replay — success path** (member `M-1001`, an existing member — replays the hand-written reference artifact and extracts confirmation data):

```bash
python -m src.main replay \
  --artifact saved_artifacts/open_new_account.json \
  --params '{"username": "admin", "password": "admin123", "member_id": "M-1001", "deposit_amount": "100.00"}'
```

Compare against the recorded evidence: `evidence/replay/replay_success_result.json` (`outcome.type == "success"`).

**d. Run replay — business outcome** (member `M-9999`, which does not exist — the engine correctly classifies this as an expected business result, not an error):

```bash
python -m src.main replay \
  --artifact saved_artifacts/open_new_account.json \
  --params '{"username": "admin", "password": "admin123", "member_id": "M-9999", "deposit_amount": "100.00"}'
```

Compare against `evidence/failure/replay_business_outcome_result.json` (`outcome.type == "business_outcome"`, `outcome_name == "member_not_found"`).

**e. Run replay — hard failure** (stop the Flask demo app first with `Ctrl+C`, then run the same replay command as in step c against the now-unreachable target):

```bash
python -m src.main replay \
  --artifact saved_artifacts/open_new_account.json \
  --params '{"username": "admin", "password": "admin123", "member_id": "M-1001", "deposit_amount": "100.00"}'
```

Compare against `evidence/failure/replay_hard_failure_result.json` (`outcome.type == "hard_failure"`, with a screenshot and the failing step's expected/observed state). Restart `demo_app/app.py` before running any further demos.

**f. Run replay with human escalation** — pauses on risky/irreversible steps (or unresolved failures) and opens an operator console instead of proceeding unattended. Forces a visible browser window:

```bash
python -m src.main replay \
  --artifact saved_artifacts/open_new_account.json \
  --params '{"username": "admin", "password": "admin123", "member_id": "M-1001", "deposit_amount": "100.00"}' \
  --enable-escalation
```

Open the operator UI at **http://127.0.0.1:8080** to approve each pause (or take manual control of the same browser window) and resume the run.

## Running Without an API Key

Only the `discover` command calls the Anthropic API. The `replay` command makes **no LLM calls at all** — it deterministically executes a saved artifact's recorded steps against a live browser. This means steps **c**, **d**, **e**, and **f** above (and any of the pre-recorded artifacts in [`saved_artifacts/`](saved_artifacts/)) can be run and verified with no `ANTHROPIC_API_KEY` set. Only step **b** (discovery) requires a real key.

## Project Structure

```
.
├── demo_app/           # Mock credit-union member portal (Flask) — the demo target app
├── src/
│   ├── agent/          # Discovery loop: observer (Claude call) + driver
│   ├── artifacts/      # Converts a discovery run into a saved Artifact
│   ├── models/         # Pydantic schema: artifacts, actions, results
│   ├── replay/         # Deterministic replay engine (no LLM calls)
│   ├── safety/         # Allowlist, risk classifier, redactor
│   ├── handoff/        # Human escalation + operator server
│   ├── observability/  # Structured evidence/logging shared by both stages
│   └── main.py         # CLI entry point (`python -m src.main ...`)
├── operator_ui/        # Static operator console served during escalation
├── config/             # allowlist.json, risk_rules.json
├── saved_artifacts/    # Recorded, reusable flow artifacts (JSON)
├── evidence/           # Redacted run evidence: discovery/, replay/, failure/, safety/
├── tests/              # Test suite
├── REPORT.md           # Full design write-up
└── requirements.txt
```

## Design Decisions

See [REPORT.md](REPORT.md) for the full write-up: architecture rationale, the artifact schema, the determinism/error-handling model, multi-tenant and heterogeneous-surface extensibility, escalation/handoff design, safety layers, and known cuts.
