from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .core import ResearchController, read_json
from .server import serve


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_controller(config_path: str) -> ResearchController:
    root = project_root()
    config = read_json(root / config_path)
    return ResearchController(root, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="KuaiRand autonomous research agent")
    parser.add_argument("--config", default="configs/default.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Load the immutable baseline artifact")
    init_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run the autonomous research loop")
    run_parser.add_argument("--reset", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Serve the dashboard API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    subparsers.add_parser("state", help="Print current machine-readable state")

    args = parser.parse_args()
    controller = build_controller(args.config)
    if args.command == "init":
        print(json.dumps(controller.initialize(force=args.force), indent=2))
    elif args.command == "state":
        print(json.dumps(controller.initialize(), indent=2))
    elif args.command == "serve":
        serve(controller, args.host, args.port)
    elif args.command == "run":
        if args.reset:
            controller.initialize(force=True)
        controller.start()
        seen = 0
        while True:
            state = controller.store.load()
            events = state.get("events", [])
            for event in events[seen:]:
                print(f"[{event['kind']}] {event['message']}")
            seen = len(events)
            if state["run"]["status"] in {"completed", "converged", "budget_exhausted", "stopped", "error"}:
                print(json.dumps({"status": state["run"]["status"], "best": state["best"]}, indent=2))
                break
            time.sleep(0.2)


if __name__ == "__main__":
    main()
