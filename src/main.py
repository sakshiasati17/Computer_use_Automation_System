"""CLI entry point: `python -m src.main <command> ...`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

__all__ = ["build_arg_parser", "main"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.main")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser(
        "discover", help="Run the discovery agent loop against a target web app."
    )
    discover.add_argument("--goal", required=True, help="Natural-language goal for the agent to accomplish.")
    discover.add_argument("--target", required=True, help="URL to start the discovery session from.")
    discover.add_argument(
        "--max-steps", type=int, default=20, help="Maximum steps before giving up (default: 20)."
    )

    replay = subparsers.add_parser(
        "replay", help="Deterministically replay a saved artifact against a live browser. No LLM calls."
    )
    replay.add_argument("--artifact", required=True, help="Path to the saved artifact JSON file.")
    replay.add_argument(
        "--params", default="{}", help="JSON object of input parameters, e.g. '{\"member_id\": \"M-1001\"}'."
    )
    replay.add_argument(
        "--headed", action="store_true", help="Run with a visible browser window (default: headless)."
    )
    replay.add_argument(
        "--permitted-domain",
        action="append",
        dest="permitted_domains",
        help="Domain allowed for navigate steps. Repeatable. Defaults to the artifact's own recorded domain.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)

    if args.command == "discover":
        from .agent.discovery import run_discovery

        log = asyncio.run(run_discovery(goal=args.goal, target=args.target, max_steps=args.max_steps))
        print(
            json.dumps(
                {
                    "outcome": log["outcome"],
                    "log_path": log.get("log_path"),
                    "artifact_path": log.get("artifact_path"),
                    "failure_log_path": log.get("failure_log_path"),
                },
                indent=2,
            )
        )
        return 0 if log["outcome"] == "goal_complete" else 1

    if args.command == "replay":
        from .replay import ReplayConfig, replay_artifact

        params = json.loads(args.params)
        config = ReplayConfig(permitted_domains=args.permitted_domains, headless=not args.headed)
        result = asyncio.run(replay_artifact(args.artifact, params, config))
        print(result.model_dump_json(indent=2))
        return 0 if result.outcome.type != "hard_failure" else 1

    raise ValueError(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
