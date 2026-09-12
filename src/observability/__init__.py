"""Structured evidence/logging for discovery and replay runs.

`evidence`: `RunLogger` accumulates a run's steps as they happen;
`RunEvidence` is the resulting Pydantic record; `save_evidence` redacts and
files it under `evidence/<discovery|replay|failure>/`. See
`src/agent/discovery.py` and `src/replay/engine.py` for the two call sites.
"""

from .evidence import RunEvidence, RunLogger, RunOutcome, RunType, StepEvidence, save_evidence

__all__ = ["RunEvidence", "RunLogger", "RunOutcome", "RunType", "StepEvidence", "save_evidence"]
