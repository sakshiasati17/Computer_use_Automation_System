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

    raise ValueError(f"unknown command: {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
