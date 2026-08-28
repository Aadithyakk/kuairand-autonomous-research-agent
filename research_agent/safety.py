from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class CodeSafetyGate:
    """Fail-closed static checks for model-authored experiment programs."""

    allowed_import_roots = {
        "collections", "functools", "itertools", "json", "math", "pathlib",
        "statistics", "typing",
    }
    forbidden_calls = {
        "eval", "exec", "compile", "__import__", "breakpoint", "input",
    }
    forbidden_names = {
        "socket", "subprocess", "requests", "urllib", "http", "ftplib",
        "shutil", "tempfile", "ctypes", "multiprocessing",
    }
    forbidden_fragments = {
        "validation_labels", "hidden_test", "hidden-test", "test_labels",
        "http://", "https://", "../", "~", "/users/", "/etc/", "/var/",
    }

    def __init__(self, extra_allowed_import_roots: set[str] | None = None):
        self.allowed_import_roots = set(type(self).allowed_import_roots)
        self.allowed_import_roots.update(extra_allowed_import_roots or set())

    def inspect(self, source: str) -> dict[str, Any]:
        findings: list[str] = []
        if not source.strip():
            findings.append("Generated program is empty")
        if len(source.encode("utf-8")) > 40_000:
            findings.append("Generated program exceeds the 40 KB safety limit")
        lowered = source.lower()
        for fragment in sorted(self.forbidden_fragments):
            if fragment in lowered:
                findings.append(f"Forbidden source fragment: {fragment}")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            findings.append(f"Syntax error: {exc.msg} at line {exc.lineno}")
            return {"passed": False, "findings": findings}

        has_main = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for module in modules:
                    root = module.split(".")[0]
                    if root not in self.allowed_import_roots:
                        findings.append(f"Import is not allowed: {module}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.forbidden_calls:
                    findings.append(f"Dynamic or interactive call is forbidden: {node.func.id}")
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"unlink", "rmdir"}:
                    findings.append(f"Destructive filesystem call is forbidden: {node.func.attr}")
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"absolute", "cwd", "glob", "iterdir", "resolve", "rglob"}:
                    findings.append(f"Filesystem discovery call is forbidden: {node.func.attr}")
            elif isinstance(node, ast.Attribute) and node.attr == "parent":
                findings.append("Parent-directory traversal is forbidden")
            elif isinstance(node, ast.Name) and node.id in self.forbidden_names:
                findings.append(f"Forbidden capability referenced: {node.id}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal = node.value.strip()
                if literal.startswith(("/", "~")):
                    findings.append("Absolute or home-relative path literal is forbidden")
                if ".." in Path(literal).parts:
                    findings.append("Parent path traversal is forbidden")
            elif isinstance(node, ast.FunctionDef) and node.name == "main":
                has_main = True
        if not has_main:
            findings.append("Program must define main()")
        return {"passed": not findings, "findings": sorted(set(findings))}


class IsolatedPythonRunner:
    """Executes approved standard-library experiment code in a bounded directory."""

    def __init__(self, timeout_seconds: int = 20):
        self.timeout_seconds = timeout_seconds

    def run(self, experiment_dir: Path, source: str) -> dict[str, Any]:
        program_path = experiment_dir / "experiment.py"
        stdout_path = experiment_dir / "stdout.log"
        stderr_path = experiment_dir / "stderr.log"
        program_path.write_text(source, encoding="utf-8")
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        started = time.monotonic()
        try:
            process = subprocess.run(
                [sys.executable, "-I", "experiment.py"],
                cwd=experiment_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            return {
                "status": "failed", "stage": "execution", "error": "Experiment timed out",
                "elapsed_seconds": round(time.monotonic() - started, 4),
            }
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
        if process.returncode != 0:
            return {
                "status": "failed", "stage": "execution",
                "error": f"Experiment exited with code {process.returncode}",
                "stderr_tail": process.stderr[-1200:],
                "elapsed_seconds": round(time.monotonic() - started, 4),
            }
        return {
            "status": "completed", "stage": "execution",
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "artifacts": [str(program_path), str(stdout_path), str(stderr_path)],
        }


def write_validation_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
