from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_MARKER = "__CANDIDATE_SOURCE_REPR__"
PROPOSAL_MARKER = "__PROPOSAL_REPR__"


def package(template: Path, candidate: Path, proposal: Path, destination: Path) -> None:
    template_text = template.read_text(encoding="utf-8")
    if template_text.count(SOURCE_MARKER) != 1 or template_text.count(PROPOSAL_MARKER) != 1:
        raise ValueError("Worker template markers are missing or ambiguous")
    source = candidate.read_text(encoding="utf-8")
    proposal_value = json.loads(proposal.read_text(encoding="utf-8"))
    rendered = template_text.replace(f'"{SOURCE_MARKER}"', repr(source)).replace(
        f'"{PROPOSAL_MARKER}"', repr(json.dumps(proposal_value, ensure_ascii=False))
    )
    if SOURCE_MARKER in rendered or PROPOSAL_MARKER in rendered:
        raise ValueError("Worker template rendering left unresolved markers")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a safety-approved generated program in the trusted Kaggle worker")
    parser.add_argument("--template", type=Path, default=Path("outputs/kuairand_candidate_worker.py"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    package(args.template, args.candidate, args.proposal, args.destination)


if __name__ == "__main__":
    main()
