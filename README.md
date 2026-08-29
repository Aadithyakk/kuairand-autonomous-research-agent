# KuaiLab v2

KuaiLab is a local-first autonomous research control room for the KuaiRand-Pure `long_view` recommendation task. It uses the OpenAI Responses API with `gpt-5.6-sol` to design one falsifiable experiment at a time, runs experiments through a sealed benchmark adapter, retains the validation-best champion, and records the evidence behind every decision.

The previous repository is not a dependency. This implementation was rebuilt from scratch.

## What works now

- Six visible stages per iteration: inspect, hypothesize, implement, train, evaluate, reflect.
- GPT-5.6 Sol provider with high reasoning and strict JSON output.
- Deterministic no-cost demo provider and synthetic benchmark for end-to-end smoke testing.
- Real KuaiRand-Pure adapter built around the supplied organizer starter kit, plus a fail-closed external-adapter contract for alternative runners.
- Typed pointwise, positive-weighted, multi-seed, BPR pairwise, pointwise/pairwise-blend, and DeepFM-blend experiment executors.
- Atomic `state.json`, append-only `events.jsonl`, proposal source, unified diff, runner logs, metrics, failures, and recovery evidence.
- Champion promotion, 50-iteration and six-hour budgets, and convergence after three gains below 0.002.
- Live dashboard controls: start, pause, resume, stop, and steer the next hypothesis.
- Token accounting split into input, output, reasoning, and total tokens.
- Per-run compute accounting: training and wall time, CPU time/hours, average CPU utilization, peak RAM, GPU-hours, and peak VRAM.

## Verified real runs

On 29 August 2026, KuaiLab completed a GPT-5.6 Sol campaign against the supplied KuaiRand-Pure validation split. It first improved the retained FM champion from `0.601470` to `0.603781` primary score, then added a within-user BPR executor and reached `0.605366`. A final controlled extension verified a compact three-way ensemble at **`0.605809` primary** (`+0.004340`, or about `+0.72%` relative, over the reproduced baseline), with `0.672898` GAUC and `0.538721` nDCG@5.

The retained blend averages positive-weighted pointwise FMs from seeds 0 and 2, mixes in a seed-1 BPR FM at weight `0.455`, then mixes a seed-0 DeepFM into that score at weight `0.23`. Its clean executor run took `45.674` seconds, used `0.014031` CPU-hours, peaked at `892.312` MB RAM, and used no GPU. The method is now a typed `fm_deep_blend` option available to the autonomous planner rather than a one-off analysis script.

One ensemble attempt was deliberately invalidated after its runner process was terminated: inspection showed that the ensemble path would have ignored the proposed positive-example weight. The result was not scored or promoted, and the runner was corrected before the campaign continued. The hidden test was never accessed.

The original campaign evidence is in [`results/run-9ecfd2aa09`](results/run-9ecfd2aa09), the pairwise milestone is in [`results/verified-pairwise-blend`](results/verified-pairwise-blend), and the retained DeepFM-blend verification is in [`results/verified-deep-blend`](results/verified-deep-blend). Checkpoints, raw runner requests, logs containing local paths, and the dataset are excluded from Git.

## Security first

The API key pasted into chat should be revoked. It is not present anywhere in this project. Create a fresh key and expose it only to the backend process:

```bash
export OPENAI_API_KEY='your-newly-rotated-key'
```

Do not put a real key in `.env.example`, source control, proposals, runner logs, or dashboard requests. The browser never receives the key.

## Start locally

Requirements: Python 3.11+, Node 22.13+, and npm. Install the core Python dependency with `python3 -m pip install -r requirements.txt`. The `fm_deep_blend` executor additionally needs PyTorch from `python3 -m pip install -r requirements-deep.txt`; all other executors remain NumPy-only.

```bash
npm install
npm run local
```

Then open the dashboard URL shown in the terminal (normally `http://localhost:3000`). Start with **Demo planner + Synthetic smoke test**; it exercises the entire campaign lifecycle without API usage.

Useful checks:

```bash
npm run test:backend
npm run lint
npm run build
```

## Use GPT-5.6 Sol

With a fresh `OPENAI_API_KEY` exported, restart `npm run local`. The dashboard readiness strip will show **GPT key ready**, and GPT-5.6 Sol becomes selectable. Defaults:

- Model: `gpt-5.6-sol`
- Reasoning effort: `high`
- API: Responses API
- Output: strict experiment-proposal JSON schema

Override these with `KUAILAB_MODEL` and `KUAILAB_REASONING_EFFORT` if needed.

## Connect the real KuaiRand-Pure benchmark

When the two supplied archives are extracted under `external/`, `npm run local` detects them automatically. For a different location, set both variables explicitly:

```bash
export KUAIRAND_DATA_PATH='/absolute/path/to/KuaiRand-Pure'
export KUAI_EXPERIMENT_COMMAND='/absolute/path/to/your-organizer-adapter'
```

The command is split into arguments without a shell and runs inside an iteration workspace. It receives `KUAI_RUNNER_REQUEST`, the path to JSON containing:

- `action` (`baseline` or `experiment`)
- `iteration`
- `dataset_path`
- `proposal_path` (`null` for the baseline)
- `metrics_path`
- target `long_view`

Your adapter must use the organizer starter kit to train/validate the candidate, then write the requested `metrics.json`:

```json
{
  "primary": 0.6078,
  "gauc": 0.6731,
  "ndcg5": 0.5425,
  "runtime_seconds": 412.8,
  "resource_usage": {
    "wall_seconds": 412.8,
    "train_seconds": 391.4,
    "cpu_seconds": 1460.2,
    "peak_rss_mb": 2840,
    "gpu_count": 1,
    "gpu_seconds": 391.4,
    "peak_gpu_memory_mb": 6120,
    "device": "gpu"
  }
}
```

All three metrics must be finite values in `[0, 1]`. `resource_usage` is optional for third-party adapters; KuaiLab falls back to wall time, child-process CPU time, and peak resident memory when it is absent. GPU-hours and peak VRAM must be supplied by GPU-backed adapters because the controller cannot reliably infer accelerator use across containers. A non-zero exit, timeout, missing file, or invalid metric fails the iteration, retains the prior champion, and still records the compute spent before failure. Generated candidate code is saved for review but is never executed directly by KuaiLab; isolation is the adapter's responsibility (normally a pinned container built around the official starter kit).

## Evidence layout

Full runtime evidence is intentionally untracked. A compact sanitized real-run record is checked into `results/`:

```text
results/run-9ecfd2aa09/
  summary.json
  baseline/{metrics.json,training-history.json}
  iteration-001/{proposal.json,candidate.py,candidate.diff,metrics.json,training-history.json}
  ...

results/verified-pairwise-blend/
  summary.json
  proposal.json
  metrics.json
  resource-usage.json

results/verified-deep-blend/
  summary.json
  proposal.json
  metrics.json
  resource-usage.json

runtime/
  state.json
  events.jsonl
  campaigns/run-.../iteration-001/
    proposal.json
    candidate.py
    candidate.diff
    runner-request.json       # real mode
    runner.stdout.log         # real mode
    runner.stderr.log         # real mode
    metrics.json              # real mode
    resource-usage.json       # normalized per-run compute evidence
  resource-summary.json       # campaign totals and per-iteration usage
```

The hidden test is not part of the autonomous loop. Only the retained validation-best checkpoint should be evaluated once under the challenge's final-submission procedure.

## Local API

- `GET /api/health`
- `GET /api/state`
- `GET /api/events`
- `POST /api/run/start` with `{ "provider": "demo|gpt", "mode": "demo|kuairand" }`
- `POST /api/run/pause`, `/api/run/resume`, `/api/run/stop`, `/api/run/reset`
- `POST /api/steer` with `{ "instruction": "..." }`

The API binds to `127.0.0.1:8787` by default and only permits the local dashboard origins through CORS.
