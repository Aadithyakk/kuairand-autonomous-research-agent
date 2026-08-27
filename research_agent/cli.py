from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .autonomous import GenericResearchAgent, ScriptedValidationModel
from .benchmark import ToyRankingBenchmark
from .core import ResearchController, read_json
from .core import LiteratureIndex
from .kuairand_contract import validate_kuairand_inputs
from .safety import write_validation_report
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

    run_parser = subparsers.add_parser("run", help="Run the legacy catalog controller")
    run_parser.add_argument("--reset", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Serve the dashboard API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)

    subparsers.add_parser("state", help="Print current machine-readable state")
    validation_parser = subparsers.add_parser("validate-agent", help="Validate the autonomous outer loop end to end")
    validation_parser.add_argument("--output", default="runtime/agent-validation")
    kuairand_parser = subparsers.add_parser("validate-kuairand", help="Validate real KuaiRand data and baseline inputs")
    kuairand_parser.add_argument("--dataset", default="work/kuairand/KuaiRand-Pure.tar.gz")
    kuairand_parser.add_argument("--baseline", default="artifacts/iteration_000_baseline_artifacts.zip")
    kuairand_parser.add_argument("--output", default="runtime/kuairand-readiness.json")

    args = parser.parse_args()
    if args.command == "init":
        controller = build_controller(args.config)
        print(json.dumps(controller.initialize(force=args.force), indent=2))
    elif args.command == "state":
        controller = build_controller(args.config)
        print(json.dumps(controller.initialize(), indent=2))
    elif args.command == "serve":
        controller = build_controller(args.config)
        serve(controller, args.host, args.port)
    elif args.command == "run":
        controller = build_controller(args.config)
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
    elif args.command == "validate-agent":
        validation_root = project_root() / args.output
        benchmark = ToyRankingBenchmark(validation_root / "benchmark")
        literature = LiteratureIndex.from_path(project_root() / "knowledge/literature.json")
        agent = GenericResearchAgent(
            benchmark=benchmark,
            research_model=ScriptedValidationModel(),
            literature=literature,
            workspace=validation_root / "agent",
            max_experiments=2,
            budget_seconds=60,
            convergence_patience=3,
        )
        state = agent.run(force=True)
        experiments = state["experiments"]
        experiment_files = list((validation_root / "agent/experiments").rglob("*"))
        checks = {
            "baseline_reproduced": state["baseline"]["status"] == "passed",
            "planner_created_experiments": len(experiments) == 2 and len(state["decisions"]) == 2,
            "unsafe_code_rejected": experiments[0].get("stage") == "safety" and experiments[0]["status"] == "failed",
            "agent_recovered": experiments[1]["status"] == "completed",
            "champion_improved": state["best"]["metrics"]["primary"] > state["baseline"]["metrics"]["primary"],
            "memory_updated": len(state["memory"]) == 2,
            "audit_complete": {"research", "decision", "recovery", "result", "reflection", "complete"}.issubset({event["kind"] for event in state["events"]}),
            "hidden_labels_isolated": not any("validation_labels" in str(path) for path in experiment_files),
            "zero_manual_interventions": state["run"]["manual_interventions"] == 0,
        }
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "baseline": state["baseline"]["metrics"],
            "champion": state["best"],
            "experiments": [
                {
                    "id": item["experiment_id"], "title": item["title"], "status": item["status"],
                    "stage": item.get("stage"), "metrics": item.get("metrics"), "improved": item.get("improved"),
                    "lesson": item.get("reflection", {}).get("reusable_lesson"),
                }
                for item in experiments
            ],
            "report_path": str(validation_root / "validation-report.json"),
        }
        write_validation_report(validation_root / "validation-report.json", report)
        print(json.dumps(report, indent=2))
        if report["status"] != "passed":
            sys.exit(1)
    elif args.command == "validate-kuairand":
        root = project_root()
        report = validate_kuairand_inputs(root / args.dataset, root / args.baseline)
        write_validation_report(root / args.output, report)
        print(json.dumps(report, indent=2))
        if report["status"] != "passed":
            sys.exit(1)


if __name__ == "__main__":
    main()
