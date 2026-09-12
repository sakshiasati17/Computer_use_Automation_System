"""Deterministic replay of saved Artifacts: the production execution path.

`src.agent.discovery` uses an LLM once, to record an Artifact.
`src.replay.engine.replay_artifact` runs it afterward, thousands of times a
day, with no LLM in the loop.
"""

from .engine import ReplayConfig, replay_artifact

__all__ = ["ReplayConfig", "replay_artifact"]
