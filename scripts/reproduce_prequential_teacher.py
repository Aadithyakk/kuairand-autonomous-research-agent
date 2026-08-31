#!/usr/bin/env python3
"""One entry point for the locked 37-stage prequential-teacher result.

There are two deliberately separate reproducibility levels:

* ``reproduce`` downloads (optionally), hashes, and replays every published
  promotion artifact, then recomputes the final public-development metrics.
* ``retrain-source`` reruns an individual logistic/CatBoost source generator
  from its locked parameters. CatBoost archives can differ byte-for-byte
  across platforms, so numerical arrays are compared instead of ZIP bytes.

The reserved 29 April--8 May period is never evaluated here. The only command
that touches it reads an explicit label-free column allowlist.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "prequential_teacher.lock.json"
CHECKSUM_PATH = ROOT / "results" / "prequential-online-teacher" / "checksums.json"
DEFAULT_ARTIFACT_ROOT = ROOT / "runtime" / "prequential-teacher-release"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_root(value: str | None) -> Path:
    configured = value or os.environ.get("KUAI_PREQUENTIAL_ARTIFACT_ROOT")
    return Path(configured).resolve() if configured else DEFAULT_ARTIFACT_ROOT


def validate_lock(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    stages = config.get("stages", [])
    if len(stages) != 37:
        errors.append(f"expected 37 stages, found {len(stages)}")
    if [stage.get("id") for stage in stages] != list(range(1, 38)):
        errors.append("stage IDs are not the ordered sequence 1..37")
    required = {
        "source", "transform", "gate", "weight", "expected_metrics",
        "source_artifact", "output_artifact", "report_artifact", "generator",
    }
    for stage in stages:
        missing = sorted(required - set(stage))
        if missing:
            errors.append(f"stage {stage.get('id')} missing {missing}")
    development = config.get("periods", {}).get("development", {})
    holdout = config.get("periods", {}).get("final_holdout", {})
    if development.get("end_date", 99999999) >= holdout.get("start_date", 0):
        errors.append("final holdout is not strictly later than development")
    if holdout.get("status") != "reserved_not_evaluated":
        errors.append("final holdout is not locked as reserved_not_evaluated")
    return errors


def verify_entries(
    entries: list[dict[str, Any]], base: Path, label: str
) -> tuple[int, int, list[str]]:
    present = 0
    errors: list[str] = []
    for entry in entries:
        path = base / entry["path"]
        if not path.is_file():
            continue
        present += 1
        size = path.stat().st_size
        if size != entry["size"]:
            errors.append(
                f"{label} size mismatch: {entry['path']} ({size} != {entry['size']})"
            )
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            errors.append(f"{label} SHA-256 mismatch: {entry['path']}")
    return present, len(entries), errors


def command_doctor(args: argparse.Namespace) -> int:
    config = load_json(CONFIG_PATH)
    checksums = load_json(CHECKSUM_PATH)
    errors = validate_lock(config)
    root = artifact_root(args.artifact_root)
    dataset_present, dataset_total, dataset_errors = verify_entries(
        checksums["dataset"], ROOT, "dataset"
    )
    artifact_present, artifact_total, artifact_errors = verify_entries(
        checksums["artifacts"], root, "artifact"
    )
    errors.extend(dataset_errors)
    errors.extend(artifact_errors)

    interpreter_match = platform.python_version() == config["environment"]["python"]
    platform_match = (
        platform.system() == config["environment"]["system"]
        and platform.machine() == config["environment"]["machine"]
    )
    package_mismatches = []
    package_lock = ROOT / config["environment"]["package_lock"]
    locked_packages = {}
    for line in package_lock.read_text().splitlines():
        if line and not line.startswith("#") and "==" in line:
            name, version = line.split("==", 1)
            locked_packages[name] = version
    for name, expected in locked_packages.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            package_mismatches.append({"name": name, "actual": actual, "expected": expected})
    if args.strict_environment and not interpreter_match:
        errors.append(
            f"Python mismatch: {platform.python_version()} != "
            f"{config['environment']['python']}"
        )
    if args.strict_environment and not platform_match:
        errors.append("platform differs from the locked training platform")
    if args.strict_environment and package_mismatches:
        errors.append(f"package lock mismatch: {len(package_mismatches)} package(s)")
    if args.require_dataset and dataset_present != dataset_total:
        errors.append(f"dataset incomplete: {dataset_present}/{dataset_total}")
    if args.require_artifacts and artifact_present != artifact_total:
        errors.append(f"release artifacts incomplete: {artifact_present}/{artifact_total}")

    result = {
        "lock": "valid" if not validate_lock(config) else "invalid",
        "stages": len(config["stages"]),
        "python": {
            "actual": platform.python_version(),
            "expected": config["environment"]["python"],
            "match": interpreter_match,
        },
        "platform": {
            "actual": f"{platform.system()} {platform.machine()}",
            "expected": (
                f"{config['environment']['system']} "
                f"{config['environment']['machine']}"
            ),
            "match": platform_match,
        },
        "packages": {
            "locked": len(locked_packages),
            "mismatches": package_mismatches,
        },
        "dataset_files": f"{dataset_present}/{dataset_total}",
        "release_artifacts": f"{artifact_present}/{artifact_total}",
        "artifact_root": str(root),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if resolved != target and resolved not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not permitted in release bundle: {member.name}")
        bundle.extractall(destination, filter="data")


def command_download(args: argparse.Namespace) -> int:
    config = load_json(CONFIG_PATH)
    release = config["release"]
    url = args.url or release["asset_url"]
    root = artifact_root(args.artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="prequential-teacher-", suffix=".tar.gz", delete=False
    ) as temp:
        temp_path = Path(temp.name)
    try:
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, temp_path)
        expected = release.get("asset_sha256")
        actual = sha256(temp_path)
        if expected and actual != expected:
            raise ValueError(f"release SHA-256 mismatch: {actual} != {expected}")
        safe_extract(temp_path, root)
    finally:
        temp_path.unlink(missing_ok=True)
    print(json.dumps({"downloaded": url, "artifact_root": str(root)}, indent=2))
    return 0


def deterministic_bundle(
    output: Path,
    entries: list[tuple[Path, str]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as bundle:
                for source, logical_path in sorted(entries, key=lambda item: item[1]):
                    info = bundle.gettarinfo(str(source), arcname=logical_path)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with source.open("rb") as handle:
                        bundle.addfile(info, handle)


def command_bundle(args: argparse.Namespace) -> int:
    checksums = load_json(CHECKSUM_PATH)
    work_root = Path(args.work_root).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    entries: list[tuple[Path, str]] = []
    missing: list[str] = []
    for item in checksums["artifacts"]:
        logical = item["path"]
        if logical.startswith("work/"):
            source = work_root / logical.removeprefix("work/")
        elif logical.startswith("runtime/"):
            source = runtime_root / logical.removeprefix("runtime/")
        else:
            raise ValueError(f"unknown artifact namespace: {logical}")
        if not source.is_file():
            missing.append(f"{logical} <- {source}")
            continue
        if source.stat().st_size != item["size"] or sha256(source) != item["sha256"]:
            raise ValueError(f"source does not match checksum lock: {source}")
        entries.append((source, logical))
    if missing:
        raise FileNotFoundError("missing bundle inputs:\n" + "\n".join(missing))
    output = Path(args.output).resolve()
    deterministic_bundle(output, entries)
    result = {
        "output": str(output),
        "files": len(entries),
        "size": output.stat().st_size,
        "sha256": sha256(output),
    }
    print(json.dumps(result, indent=2))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    import numpy as np

    config = load_json(CONFIG_PATH)
    checksums = load_json(CHECKSUM_PATH)
    root = artifact_root(args.artifact_root)
    present, total, errors = verify_entries(checksums["artifacts"], root, "artifact")
    if present != total:
        errors.append(f"release artifacts incomplete: {present}/{total}")

    previous = None
    for stage in config["stages"]:
        report_path = root / stage["report_artifact"]
        output_path = root / stage["output_artifact"]
        if not report_path.is_file() or not output_path.is_file():
            continue
        report = load_json(report_path)
        selected = report.get("selected") or {}
        for key in ("source", "gate", "weight"):
            if selected.get(key) != stage[key]:
                errors.append(
                    f"stage {stage['id']} {key}: {selected.get(key)!r} != {stage[key]!r}"
                )
        with np.load(output_path) as archive:
            champion = np.asarray(archive["champion"])
            score = np.asarray(archive["selected"])
        if previous is not None and not np.array_equal(previous, champion):
            errors.append(f"stage {stage['id']} champion does not equal stage {stage['id']-1}")
        previous = score

    metrics = None
    dataset = ROOT / config["dataset"]["standard_log"]
    if previous is not None and dataset.is_file():
        import pandas as pd

        sys.path.insert(0, str(ROOT / "scripts"))
        from joint_terminal_gate_search import exact_metrics

        development = config["periods"]["development"]
        rows = pd.read_csv(
            dataset,
            usecols=["user_id", "date", "long_view"],
            dtype={"user_id": "string"},
        )
        rows = rows.loc[
            (rows["date"] >= development["start_date"])
            & (rows["date"] <= development["end_date"])
        ].reset_index(drop=True)
        if len(rows) != len(previous):
            errors.append(f"score length {len(previous)} != development rows {len(rows)}")
        else:
            metrics = exact_metrics(
                rows["user_id"].astype(str).to_numpy(),
                rows["long_view"].to_numpy(dtype=np.float64),
                previous.astype(np.float64),
            )
            expected = config["expected_final_metrics"]
            for key, value in expected.items():
                if abs(float(metrics[key]) - float(value)) > 5e-8:
                    errors.append(f"final {key}: {metrics[key]} != {value}")

    print(json.dumps({
        "stages_replayed": len(config["stages"]),
        "artifact_files": f"{present}/{total}",
        "final_metrics": metrics,
        "hidden_holdout_accessed": False,
        "errors": errors,
    }, indent=2))
    return 1 if errors else 0


def command_retrain_source(args: argparse.Namespace) -> int:
    import numpy as np

    config = load_json(CONFIG_PATH)
    root = artifact_root(args.artifact_root)
    stage = next((item for item in config["stages"] if item["id"] == args.stage), None)
    if stage is None:
        raise ValueError(f"unknown stage: {args.stage}")
    generator = stage["generator"]
    if generator["kind"] == "artifact_replay":
        raise ValueError(
            f"stage {args.stage} source is a locked assembly; use `verify` to replay it"
        )
    published = root / stage["source_artifact"]
    if not published.is_file():
        raise FileNotFoundError(f"download the release artifacts first: {published}")
    with np.load(published) as archive:
        original_champion = np.asarray(archive["champion"], dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix=f"prequential-stage-{args.stage:02d}-") as temp:
        temp_root = Path(temp)
        champion_path = temp_root / "champion.npz"
        output_path = temp_root / "source.npz"
        report_path = temp_root / "source.json"
        np.savez_compressed(champion_path, selected=original_champion)
        environment = os.environ.copy()
        environment["KUAI_PREQUENTIAL_WORKDIR"] = str(root / "work")
        environment["KUAI_STATIC_MODEL_ROOT"] = str(root / "runtime")
        environment.update({key: str(value) for key, value in generator["env"].items()})
        if generator["kind"] == "pairwise_logistic":
            environment.update({
                "KUAI_PAIRWISE_CHAMPION": str(champion_path),
                "KUAI_PAIRWISE_OUTPUT": str(output_path),
                "KUAI_PAIRWISE_REPORT": str(report_path),
            })
        elif generator["kind"] == "catboost":
            environment.update({
                "KUAI_CATBOOST_CHAMPION": str(champion_path),
                "KUAI_CATBOOST_OUTPUT": str(output_path),
                "KUAI_CATBOOST_REPORT": str(report_path),
            })
        else:
            raise ValueError(f"unsupported generator: {generator['kind']}")
        subprocess.run(
            [sys.executable, str(ROOT / generator["script"])],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        with np.load(output_path) as actual, np.load(published) as expected:
            source_key = stage["source_key"]
            np.testing.assert_allclose(
                actual[source_key], expected[source_key], rtol=args.rtol, atol=args.atol
            )
    print(json.dumps({
        "stage": args.stage,
        "source": stage["source"],
        "result": "numerically reproduced",
        "rtol": args.rtol,
        "atol": args.atol,
    }, indent=2))
    return 0


def command_causality(args: argparse.Namespace) -> int:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_prequential_causality",
            "tests.test_prequential_repro_config",
        ],
        cwd=ROOT,
    )
    return completed.returncode


def command_reserve_holdout(args: argparse.Namespace) -> int:
    import pandas as pd

    config = load_json(CONFIG_PATH)
    protocol = config["holdout_protocol"]
    holdout = config["periods"]["final_holdout"]
    safe = protocol["safe_input_columns"]
    forbidden = set(protocol["forbidden_columns"])
    if forbidden.intersection(safe):
        raise ValueError("holdout safe-input allowlist includes an outcome column")
    source = ROOT / config["dataset"]["standard_log"]
    # Crucially, pandas is asked to parse only the explicit non-outcome allowlist.
    rows = pd.read_csv(source, usecols=safe, dtype={"user_id": "string", "video_id": "string"})
    rows = rows.loc[
        (rows["date"] >= holdout["start_date"])
        & (rows["date"] <= holdout["end_date"])
    ].copy()
    rows.insert(0, "source_row_id", rows.index.to_numpy())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output, index=False)
    print(json.dumps({
        "output": str(output),
        "rows": len(rows),
        "columns": list(rows.columns),
        "sha256": sha256(output),
        "labels_read": False,
        "labels_scored": False,
        "holdout_status": "reserved_not_evaluated",
    }, indent=2))
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    root = artifact_root(args.artifact_root)
    if args.download:
        download_args = argparse.Namespace(
            artifact_root=str(root), url=None
        )
        result = command_download(download_args)
        if result:
            return result
    doctor_args = argparse.Namespace(
        artifact_root=str(root),
        require_dataset=True,
        require_artifacts=True,
        strict_environment=args.strict_environment,
    )
    for action in (
        lambda: command_doctor(doctor_args),
        lambda: command_causality(args),
        lambda: command_verify(args),
    ):
        result = action()
        if result:
            return result
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="inspect lock, environment, inputs, and artifacts")
    doctor.add_argument("--artifact-root")
    doctor.add_argument("--require-dataset", action="store_true")
    doctor.add_argument("--require-artifacts", action="store_true")
    doctor.add_argument("--strict-environment", action="store_true")
    doctor.set_defaults(func=command_doctor)

    download = sub.add_parser("download", help="download and verify the release artifact bundle")
    download.add_argument("--artifact-root")
    download.add_argument("--url")
    download.set_defaults(func=command_download)

    bundle = sub.add_parser("bundle", help="build the deterministic GitHub Release asset")
    bundle.add_argument("--work-root", required=True)
    bundle.add_argument("--runtime-root", required=True)
    bundle.add_argument("--output", required=True)
    bundle.set_defaults(func=command_bundle)

    verify = sub.add_parser("verify", help="verify all 37 promotions and final metrics")
    verify.add_argument("--artifact-root")
    verify.set_defaults(func=command_verify)

    retrain = sub.add_parser("retrain-source", help="rerun one locked source generator")
    retrain.add_argument("--stage", type=int, required=True)
    retrain.add_argument("--artifact-root")
    retrain.add_argument("--rtol", type=float, default=1e-6)
    retrain.add_argument("--atol", type=float, default=1e-6)
    retrain.set_defaults(func=command_retrain_source)

    causality = sub.add_parser("test-causality", help="mutate future labels and run guards")
    causality.set_defaults(func=command_causality)

    reserve = sub.add_parser(
        "reserve-holdout", help="export label-free rows from the untouched final period"
    )
    reserve.add_argument("--output", required=True)
    reserve.set_defaults(func=command_reserve_holdout)

    reproduce = sub.add_parser("reproduce", help="run the complete published reproduction")
    reproduce.add_argument("--artifact-root")
    reproduce.add_argument("--download", action="store_true")
    reproduce.add_argument("--strict-environment", action="store_true")
    reproduce.set_defaults(func=command_reproduce)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, AssertionError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
