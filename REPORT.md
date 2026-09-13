# Computer Use Automation System — Design Report

## 1. Architecture

The system splits into two stages sharing one artifact contract: **discovery** (LLM-driven, records a flow once) and **replay** (deterministic, executes it many times, no LLM in the loop). The split exists because the two have opposite requirements — discovery must *understand* an unfamiliar page; replay must be fast, cheap, and predictable. Paying an LLM's cost and non-determinism on every run would be wasteful; recording once and replaying deterministically amortizes it.

Components: `src/agent/observer.py` (one observe→decide call to Claude per step), `src/agent/discovery.py` (owns the Playwright browser, runs the observe→act→log loop, screenshots and logs every step to `evidence/discovery/`), `src/artifacts/emitter.py` (converts a recorded run into a structured `Artifact`), `src/replay/engine.py` (walks a saved `Artifact`'s steps with no LLM calls), `src/safety/` (allowlist, risk classifier, redactor — used identically by both stages), `src/handoff/` (human escalation, replay-only), and `src/observability/` (one `RunLogger`/`RunEvidence` shape shared by both stages).

**Why Python + Playwright:** Playwright's async API gives first-class multi-frame support (`page.frames`, frame-scoped locators) and an accessibility-tree snapshot API, both used directly to handle the demo app's iframe and table-based markup. Python pairs naturally with the Anthropic SDK's async client and Pydantic, which the entire schema layer (`src/models/`) is built on.

**Why Claude Sonnet 4.6:** the observe/decide step needs vision (a real screenshot, since layout often carries meaning a DOM dump won't) and reliable structured output (`AgentAction` is a Pydantic model with `extra="forbid"` and cross-field validation). `client.messages.parse(..., output_format=AgentAction)` gets both without a separate JSON-repair layer.

**Why a single process:** each run owns one Playwright browser, and escalation's operator server is deliberately scheduled on the *same* asyncio event loop as the replay engine, not a separate process. Because everything runs on one loop, the shared `EscalationHub` state needs no locks or IPC — a real simplicity win, at the cost of not scaling past one browser per process (see §7).

**Data flow:** `discover --goal --target` runs observe/decide/act until `goal_complete`, `stuck`, or `max_steps` → on success, `emit_artifact` converts the raw log into a schema-validated JSON file under `saved_artifacts/` → `replay --artifact --params` loads it, substitutes `{{input.x}}` parameters, and executes with Playwright, no Anthropic API call anywhere in this path → every run also produces a redacted `RunEvidence` record under `evidence/{discovery,replay,failure}/`.

## 2. Artifact schema

The schema (`src/models/artifact.py`, `actions.py`) treats an artifact as a typed function: declared `inputs` in, `outputs` out, plus what replay needs to survive a browser it doesn't fully control.

**Multiple locators with fallback chains.** `ElementTarget` is an ordered list of `Locator` variants (`text`, `aria_role`, `xpath`, `css`, `text_near`, `coordinates`) — order *is* the fallback policy, no separate priority field. Replay tries each until one resolves to exactly one element. This exists because these apps have no stable selectors; a single CSS selector would break on the vendor's next table reflow. `text_near` (the workhorse locator in `open_new_account.json`) matches a legacy layout where a label and its input are only related by document-order proximity. `coordinates` is a deliberate last resort — its validator *forces* `confidence=LOW` regardless of input, since pixel coordinates break under any layout change.

**Parameterized inputs.** Concrete values typed during discovery become `{{input.x}}` template references, not literals, via `InputParameter`. `Artifact._validate_cross_references` statically checks every reference against declared inputs at save time — a typo fails immediately, not mid-flow in production months later.

**Typed outputs.** `OutputField` pairs a declared name/type with its own extractor locator, so "what this flow returns" is explicit and checkable rather than whatever happened to land in a free-form dict.

**Business outcomes kept separate from errors.** The central decision, and it's structural, not conventional: `business_outcomes` and `error_handlers` are distinct lists. "Member not found" is a `BusinessOutcome` — the flow ran correctly and reached a known, valid, non-happy-path state. A stale-session redirect is an `ErrorHandler` — an infrastructure hiccup to recover from. Conflating the two is the most common design mistake in this space, so `ExecutionResult.outcome` (§3) makes it structurally impossible to conflate at the result layer too.

**Versioning.** `schema_version` (semver) tracks the schema format; `version` (a plain int, bumped by `emitter._next_version` on re-recording) tracks the flow's own revision history, independently.

## 3. Determinism & error handling

Replay makes zero LLM calls. Every decision — which locator to try, whether a checkpoint held, which of three outcomes a run hit — comes purely from the artifact's data plus live DOM state, which is what makes it safe to run unattended at scale.

**Locator fallback order.** `_resolve_element_target` tries each locator across every frame (main frame first, then iframes), splitting the step's timeout evenly across strategies so an early wrong one can't starve a reliable one of time on a slow page. This mirrors discovery's own `resolve_element` deliberately, so what discovery records and replay executes stay consistent.

**Three-category taxonomy** (`ExecutionOutcome`, discriminated on `type` — a caller must branch before reaching any field):
- **`SuccessOutcome`** — every step checkpoint passed and the overall `success_condition` held. Carries `outputs`.
- **`BusinessOutcomeOutcome`** — a step fails, but the page also matches a declared business-outcome condition (checked first, before anything else, once a step throws). Not an error: carries `outcome_name`, `severity`, best-effort `extracted_data`.
- **`HardFailureOutcome`** — everything else: an unresolved locator chain, a checkpoint timeout with no matching outcome/handler, or a safety violation. Carries the failing step, `expected`/`observed`, a screenshot, and every locator attempt tried.

Detection order is deliberate: business outcomes checked before error handlers, hard failure only if neither matches. `SafetyViolationError` bypasses this taxonomy entirely — an allowlist violation is a safety gate, not a condition the artifact anticipated, so it always hard-fails immediately (§6). This is demonstrated end-to-end in `evidence/`: the same `open_new_account` artifact against `M-1001` succeeds with three extracted outputs; against `M-9999` yields `business_outcome: member_not_found`; with Flask stopped yields a `hard_failure` (`CheckpointFailedError`, screenshot attached).

## 4. Heterogeneity & multi-tenant

The schema is surface-agnostic by design. `Target.surface_type` is `web | desktop | native`, and the locator vocabulary generalizes: `aria_role` is UI-Automation-native on Windows desktop apps just as it's ARIA-native on the web; only `xpath`/`css` are web-specific. Extending to legacy desktop apps means adding a **surface abstraction layer** beneath `ElementTarget` — the current `_build_locator`/`_resolve_element_target` pair *is* the web resolver; a desktop resolver would back `aria_role` with UI Automation/`pywinauto` and need a geometry-based `text_near` equivalent instead of DOM XPath. `Condition`'s `url_matches` would need a desktop substitute (window title/process state), but the rest of the vocabulary — and every consumer of it (checkpoints, outcome/handler detection, success condition) — is already surface-neutral, since none of those call sites touch Playwright directly.

For **multi-tenant reuse**, the schema already separates structure (steps, action sequence) from tenant-specific detail (input values, and per-locator rationale strings noting which label text is or isn't stable across tenant branding). The natural override point is a tenant-scoped locator override table keyed by `(artifact.name, step_id)` that substitutes an alternate locator into a step's chain without re-recording the flow — e.g. a rebranded portal's "Client ID" instead of "Member ID." Not built (§7), but every locator's `confidence`/`rationale` fields exist specifically to make "why this locator, and is it tenant-specific" legible to whoever builds that layer.

## 5. Escalation & handoff

Escalation (`src/handoff/`) exists only on the **replay** path — discovery blocks any `irreversible` action outright rather than pausing, since a person is already watching the terminal during a one-off discovery run.

**Stuck detection.** Two triggers call `escalate`: (1) a step's *effective* risk level — the more severe of the artifact's recorded `risk_level` and a fresh reclassification against live `risk_rules.json` — is `risky`/`irreversible`, checked before the step runs; (2) a step exhausts recoverable-error handling and would otherwise hard-fail, checked after, re-escalated up to 3 times per step so a human can't spin the run forever without resolving it.

**Intervention context:** a fresh screenshot, current URL, escalation reason, human-readable description, full step history, and — when the browser launched with `--remote-debugging-port` — a CDP websocket URL and window bounds, so a human can attach DevTools without disturbing the running session.

**CDP handoff.** Engine and human share the *same visible browser window*; escalation never navigates or closes the page. `escalate()` freezes the page, starts the operator console at `127.0.0.1:8080`, and blocks with **no timeout by design** — an unattended timeout that silently resumes an irreversible action defeats the point of asking a human. A human clicks **Resume** directly (approval; automation still performs the step) or **Take Control** first and acts manually, then **Resume**.

**Control state machine:** `AUTOMATION → PAUSED → (HUMAN_CONTROL, optional) → AUTOMATION`, `COMPLETED` terminal at run end. The load-bearing detail is `human_took_control`: if true, the engine does **not** re-perform the pending action, only re-verifies the checkpoint — avoiding a double-submitted payment.

**What's recorded:** every escalation becomes an `Escalation` record (step, reason, risk level, resolution, resolved_by, notes) on the run's result and evidence, plus a best-effort before/after screenshot and page-text diff for the audit trail — non-fatal to the run if capture fails.

## 6. Safety

Three independent layers, applied identically on both paths, since this drives a browser against regulated financial data and no single check should be the only thing standing between an LLM's decision (or a stale artifact) and an irreversible action.

**Allowlist enforcement:** every `navigate` checked against `permitted_domains`/`permitted_ports` and denied on `blocked_url_patterns`; every click/type/select checked against `blocked_action_keywords` (`delete`, `wire_transfer`, `close_account`, ...) against the action's identifying text. Every rejection is logged both through the standard logger and as an append-only audit record at `evidence/safety/violations.jsonl` — a silently-dropped action is as dangerous as one that ran unchecked.

**Risk classification:** one shared rule set (`risk_rules.json`) classifies `safe`/`risky`/`irreversible` by keyword/URL pattern, irreversible always winning over risky so a "Confirm Wire Transfer" is never under-classified. Checked twice per risky replay step by design: once at recording time (`Step.risk_level`), once live against the *current* rules — defense in depth against a step that was safe when recorded but matches a newly-added pattern today.

**Redaction before any disk write:** pattern-based redaction for SSNs, account numbers, card numbers, emails, and passwords — including a sibling-context rule that redacts a value next to a password-labeled field even with no recognizable pattern. Applied before every write across the system (discovery logs, artifacts, evidence), never after, and idempotent so double-redaction is safe. One deliberate carve-out: a `{{input.x}}` template reference is exempt from the password override, since redacting it would corrupt the artifact rather than protect anything.

This matters for regulated data because none of the three layers depends on the LLM behaving correctly — discovery's `AgentAction` is untrusted model output, checked exactly like a recorded artifact's steps, and redaction runs regardless of which path produced the data or whether the run succeeded.

## 7. Cuts

- **No desktop/native resolver.** The schema supports it (`SurfaceType`, surface-neutral `Condition`/`ElementTarget`), but only the Playwright/web resolver is implemented.
- **No per-tenant locator override layer.** The intended shape is described in §4; nothing currently reads or applies one — a differently-branded tenant currently needs a re-recorded artifact, not an override.
- **No retry/backoff beyond the SDK's own transient-error handling and one local retry** for a malformed structured response; a sustained API outage during discovery just surfaces as `stuck`.
- **No horizontal scaling.** Single process, single browser, per run; the escalation hub is a process-wide singleton correct for one run at a time. Running several replays with escalation enabled concurrently would need one hub per run and an operator console that can address multiple pending interventions.
- **No automated test coverage for the replay engine, safety layer, or emitter.** `tests/test_observer.py` is a manual/live check, not a unit suite; correctness evidence here is the three recorded end-to-end runs under `evidence/`. Unit tests for locator-fallback ordering, the redaction sibling-context rule, and the outcome/handler/failure triage boundary would be the highest-value next addition.
- **No artifact diffing tool.** `version` increments on re-recording, but nothing compares two versions to flag what changed before trusting a re-recorded artifact in production.
- **Business-outcome data extraction is best-effort.** It can only fill a requested field that happens to match a declared input name exactly — there's no locator on a `BusinessOutcome` the way there is on an `OutputField`, so a business outcome needing a value not already known as an input can't currently extract it.
