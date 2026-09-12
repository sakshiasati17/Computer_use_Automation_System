"""Artifact production: turning a raw discovery recording into a replayable Artifact.

- `emitter`: converts the step log produced by `src.agent.discovery.run_discovery`
  into the `Artifact` schema (`src.models.artifact`) and saves it under
  `saved_artifacts/`.
"""

from .emitter import emit_artifact

__all__ = ["emit_artifact"]
