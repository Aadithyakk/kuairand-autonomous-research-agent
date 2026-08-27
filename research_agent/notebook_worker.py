"""Execute one agent-selected model branch through the Kaggle-ready research notebook.

This bridge lets the controller remain adaptive while reusing the validated model
implementations in outputs/kuairand_autonomous_research_lab.ipynb. Each worker run
enables only the selected branch and returns the registry record as result.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .core import read_json, write_json_atomic


ACTION_MAP = {
    "history_affinity_lambdarank": ("lgb_lambdarank_history", "run_lgb_ranker"),
    "bpr_exposed_negatives": ("bpr_matrix_factorization", "run_bpr"),
    "din_causal_history": ("din_lite_sequence", "run_din"),
}


def patch_notebook(source: Path, destination: Path, spec: dict[str, Any], log_path: Path) -> str:
    action_id = spec.get("id") or spec.get("action_id")
    if action_id not in ACTION_MAP:
        raise ValueError(f"No notebook branch is registered for {action_id!r}")
    target_record, enabled_flag = ACTION_MAP[action_id]
    notebook = read_json(source)
    override = "\n".join([
        "# Injected by research_agent.notebook_worker; selection was made by the controller.",
        "CFG.run_fm = False",
        "CFG.run_lgb_binary = False",
        "CFG.run_lgb_ranker = False",
        "CFG.run_catboost_ranker = False",
        "CFG.run_bpr = False",
        "CFG.run_din = False",
        f"CFG.{enabled_flag} = True",
        f"CFG.experiment_log = {str(log_path)!r}",
    ])
    patched = False
    for cell in notebook.get("cells", []):
        raw = cell.get("source", "")
        text = "".join(raw) if isinstance(raw, list) else raw
        if "CFG = Config()" in text and "TASK = TaskSpec.from_mode" in text:
            text = text.replace("CFG = Config()", "CFG = Config()\n" + override)
            cell["source"] = text
            patched = True
            break
    if not patched:
        raise ValueError("Could not locate the notebook configuration cell")
    write_json_atomic(destination, notebook)
    return target_record


def run(spec_path: Path, result_path: Path, timeout: int) -> None:
    spec = read_json(spec_path)
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/kuairand_autonomous_research_lab.ipynb"
    work = result_path.parent
    work.mkdir(parents=True, exist_ok=True)
    notebook_path = work / "experiment.ipynb"
    executed_path = work / "experiment.executed.ipynb"
    log_path = work / "registry.jsonl"
    target_record = patch_notebook(source, notebook_path, spec, log_path)

    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={timeout}", "--output", str(executed_path), str(notebook_path),
    ]
    process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout + 60, check=False)
    (work / "worker.stdout.log").write_text(process.stdout, encoding="utf-8")
    (work / "worker.stderr.log").write_text(process.stderr, encoding="utf-8")
    records = []
    if log_path.exists():
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    record = next((item for item in reversed(records) if item.get("name") == target_record), None)
    if process.returncode != 0 or not record:
        write_json_atomic(result_path, {
            "status": "failed",
            "metrics": {},
            "error": record.get("error") if record else f"Notebook execution failed with code {process.returncode}",
            "artifacts": [str(executed_path), str(work / "worker.stderr.log")],
        })
        return
    status = "completed" if record.get("status") == "ok" else "failed"
    write_json_atomic(result_path, {
        "status": status,
        "metrics": record.get("metrics", {}),
        "error": record.get("error"),
        "notes": record.get("change_summary"),
        "worker_record": record,
        "artifacts": [str(executed_path), str(log_path)],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    try:
        run(args.spec, args.result, args.timeout)
    except Exception as exc:
        write_json_atomic(args.result, {"status": "failed", "metrics": {}, "error": str(exc), "artifacts": []})
        raise


if __name__ == "__main__":
    main()
