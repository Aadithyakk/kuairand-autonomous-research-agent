# KuaiLab judging report

## Technical execution (35%)

KuaiLab implements the complete autonomous research loop—inspect, hypothesize, implement, train, evaluate, and reflect—against the official KuaiRand-Pure temporal split and metric. Candidate code is evidence only; a typed trusted executor owns training and label access. Invalid metrics, missing output, non-zero exits, and timeouts fail closed. A worker failure is retried once through an automatically reduced batch/thread/epoch route, and compute from both attempts remains accounted for. Atomic state, append-only events, per-iteration logs, diffs, metrics, stderr/stdout, and resource summaries support restart and audit.

The current validation-best score is 0.612858 primary (0.682354 GAUC, 0.543362 nDCG@5), +0.011389 absolute over the reproduced 0.601470 FM baseline. The official hidden split remains untouched.

## Innovation and problem insight (20%)

The research found that KuaiRand's top-five metric rewards calibration and punishes rare overconfident swaps. The final method combines calibrated pointwise models with training-only preference priors and label-free slate structure, then routes only structurally defined user regimes toward or away from diverse ranking experts. Four disjoint actual-user-ID folds guard every terminal scalar. This is materially different from blind hyperparameter search: the agent reasons about exposure, sessions, uncertainty, and ranking stability while the evaluator enforces causal boundaries.

## Impact, relevance, and autonomy (20%)

The system reduces manual experiment orchestration to optional steering. It accounts for interventions, tokens, wall time, CPU, GPU, and failures; retains the champion automatically; resumes after backend restarts; and stops on the official budget or convergence rule. The same sealed-adapter pattern can be reused for other recommender datasets without giving generated code direct host execution.

## Feasibility and practicality (15%)

The core runner is CPU-first and works on Apple arm64. NumPy handles FM/BPR, PyTorch is optional for DeepFM, and research-only tree packages are isolated in a separate requirements file. The dashboard and deterministic demo run without an API key or dataset. Real mode requires local data, the official starter kit, and a fresh server-side API key. The final exporter validates the organizer schema and guards the one-time hidden-test action.

## Presentation (10%)

The control room shows champion metrics, official budget, compute, current hypothesis, six-stage progress, append-only trace, recovery failures, and iteration history. `docs/DEVPOST.md` is submission-ready written copy; `docs/FINAL_SUBMISSION.md` is the final runbook; all claimed scores point to checked-in evidence.

## Resource use to autonomous convergence

| Field | Recorded value |
| --- | ---: |
| LLM input tokens | 6,800 |
| LLM output tokens | 18,002 |
| Total billed input + output tokens | 24,802 |
| Reasoning tokens (subset of output accounting) | 12,238 |
| Agent wall-clock | 479.67 seconds |
| Iterations used | 5 / 50 |
| Manual interventions | 1 |
| GPU-hours | 0.000000 |

The later offline validation sweep tracked experiment-level CPU/RAM/wall time in its accepted and rejected cycle ledgers. The canonical final verification took 17.493 wall-seconds, 0.004858 CPU-hours, 1,108.156 MB peak RAM, and zero GPU-hours; the accepted shallow-Yeti training run took 122.022 wall-seconds, 0.206093 CPU-hours, 3,625.344 MB peak RAM, and zero GPU-hours.

## Team contributions

- **Aadithyakk / Aadithya:** original autonomous-agent architecture, campaign hardening, trusted KuaiRand runner, challenge integration, and baseline research.
- **milkksthetic:** compute tracking, pairwise/DeepFM executors, campaign continuation, extensive leak-free ablations, consensus/routing research, verified evidence, and final completion pass.
