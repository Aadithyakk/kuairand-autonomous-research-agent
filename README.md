# KuaiLab v2

KuaiLab is a local-first autonomous research control room for the KuaiRand-Pure `long_view` recommendation task. It uses the OpenAI Responses API with `gpt-5.6-sol` to design one falsifiable experiment at a time, runs experiments through a sealed benchmark adapter, retains the validation-best champion, and records the evidence behind every decision.

The previous repository is not a dependency. This implementation was rebuilt from scratch.

## What works now

- Six visible stages per iteration: inspect, hypothesize, implement, train, evaluate, reflect.
- GPT-5.6 Sol provider with high reasoning and strict JSON output.
- Deterministic no-cost demo provider and synthetic benchmark for end-to-end smoke testing.
- Real KuaiRand-Pure adapter built around the supplied organizer starter kit, plus a fail-closed external-adapter contract for alternative runners.
- Typed pointwise, positive-weighted, multi-seed, BPR pairwise, pointwise/pairwise-blend, DeepFM, clock-context, and frozen-champion residual experiment executors.
- Atomic `state.json`, append-only `events.jsonl`, proposal source, unified diff, runner logs, metrics, failures, and recovery evidence.
- Champion promotion plus the official global 50-iteration, six-hour, ε=0.002 / three-iteration convergence limits, preserved across continuation.
- Automatic planner retry and one lower-resource worker retry with both failures, routes, diffs, and compute retained in the evidence log.
- Live dashboard controls: start, continue from the retained champion, pause, resume, stop, and steer the next hypothesis.
- Token accounting split into input, output, reasoning, and total tokens.
- Per-run compute accounting: training and wall time, CPU time/hours, average CPU utilization, peak RAM, GPU-hours, and peak VRAM, plus campaign-level manual-intervention/failure/recovery counts.

## Official benchmark contract

- Target: `long_view`.
- Metrics: user-grouped GAUC and nDCG@5; primary = `0.5 * (GAUC + nDCG@5)`.
- Train: 8–21 April 2022, 1,141,112 rows.
- Validation: 22–28 April 2022, 124,909 rows.
- Hidden test: 29 April–8 May 2022, 170,588 rows; never used by the development loop.
- Campaign ceiling: 50 experiments and six wall-clock hours.
- Convergence: stop after three consecutive experiments without a primary gain greater than 0.002.
- Submission header: `row_id,user_id,video_id,score` with zero-based strictly increasing row IDs.

## Champion-mounted autonomous training

The 0.612858 validation champion is no longer metrics-only bootstrap evidence. Real campaigns expose a trusted `champion_residual_blend` experiment family to the LLM. Each such iteration:

1. verifies the frozen score archive against `results/final-model/manifest.json` and its SHA-256 checksum;
2. retrains a fresh pointwise FM, pairwise FM, DeepFM blend, or temporal DeepFM blend on the official April 8–21 training block;
3. converts the candidate and champion to stable within-user ranks;
4. blends or extrapolates the candidate at a typed weight in `[-0.25, 0.25]`;
5. evaluates on April 22–28, saves the candidate checkpoints and blended scores in the isolated iteration workspace, and promotes only a positive primary-score gain.

The hidden split remains unavailable to this adapter. Set **Mount verified 0.612858 champion** in the dashboard; the agent can then choose this family autonomously. Operator steering is optional, for example: `Try champion_residual_blend with a pairwise_fm candidate and a conservative positive weight.`

The real arm64 integration smoke retrained a fresh FM in `7.001` seconds, used `0.002496` CPU-hours and `541.359` MB peak RAM, and correctly retained the `0.612858` champion when the candidate reached only `0.596280`. Its auditable result is checked in at [`results/final-model/autonomous-worker-smoke.json`](results/final-model/autonomous-worker-smoke.json).

## Verified real runs

On 29 August 2026, KuaiLab completed a GPT-5.6 Sol campaign against the supplied KuaiRand-Pure validation split. It first improved the retained FM champion from `0.601470` to `0.603781` primary score, then added a within-user BPR executor and reached `0.605366`. A controlled DeepFM extension reached `0.605809`; the current clean checkpoint adds a small label-free clock-context FM and verifies **`0.605885` primary** (`+0.004415`, or about `+0.73%` relative, over the reproduced baseline), with `0.672964` GAUC and `0.538805` nDCG@5.

A subsequent leak-free offline research sweep adds outcome-free session position, repeat-fatigue, and time-gap rerankers, then combines candidates by within-user ordinal rank. An ordered categorical full-history candidate adds a small independent correction. Continuous watch-ratio supervision and a lightly anchored same-session margin add complementary corrections, while a full-metadata CatBoost classifier supplies a final tree-based residual. A final slate correction aligns the recipe with the official evaluator's seven-day within-user ranking unit: label-free content and temporal neighborhoods remove locally anomalous scores, while three Bayesian-smoothed cross-conditional priors use only April 8-21 training outcomes. A temporally trained April 14-21 batch-slate CatBoost then combines full-slate score neighborhoods with a continuous repeat penalty; an outer ordinal consensus retains the same positive direction across four disjoint user folds. Score-only confidence gates correct high-margin predictions, active-day and slate-size gates route sparse structural regimes, two training-coverage gates select frozen experts, a training-only user-neighbor graph adds a collaborative correction, and a jointly selected user-balanced-tree/watch-ratio pair supplies a conservative residual. Final session-count gates move one-session users toward YetiRank and two/three-session users slightly away from broad median consensus; a high-recent-coverage gate extrapolates away from a shallow regularized YetiRank expert. The current reproduced validation result is **`0.612858` primary** (`0.682354` GAUC, `0.543362` nDCG@5), a gain of `+0.011389` (about `+1.89%`) over the baseline. Multi-view neighbors, uncertainty smoothing, causal Transformers, published CWM likelihood, xDeepFM compressed interactions, and XGBoost LambdaMART found no stable promotion; the best LambdaMART seed reached `0.611949` standalone and selected zero global residual. This is recorded under `results/verified-slate-consensus`, and the autonomous engine now restores it as the validation-best starting champion. The `0.700000` stretch target has not been reached, and the remaining gap is `0.087142`.

The retained blend averages positive-weighted pointwise FMs from seeds 0 and 2, mixes in a seed-1 BPR FM at weight `0.455`, mixes a seed-0 DeepFM into that score at weight `0.23`, then adds a globally standardized clock-context FM at weight `0.024`. Its clean executor run took `69.070` seconds, used `0.020507` CPU-hours, peaked at `1042.172` MB RAM, and used no GPU. The method is available to the autonomous planner as typed `fm_temporal_deep_blend` evidence rather than a one-off analysis script.

The first 30 August follow-up cycle tested training-aggregate transfer, 205 retained model artifacts, random-exposure transfer, pointwise/pairwise stackers, public-feedback priors, expert routing, and localized top-k reranking. None survived four held-out user folds, so that cycle retained `0.612781`. Successful runs used `1044.430` wall-seconds, `1.045439` CPU-hours, at most `3224.984` MB RAM, and no GPU; the full rejection ledger is in [`results/verified-slate-consensus/rejected-cycle-2026-08-30.json`](results/verified-slate-consensus/rejected-cycle-2026-08-30.json).

A second 30 August cycle tested censored mixture-of-lognormals watch survival, nonparametric threshold survival priors, and shared plus independent duration-regime experts. The best apparent fixed score was `0.612828`, but it regressed on a held-out fold and was not promoted. Tracked runs used `705.294` wall-seconds, `0.365852` CPU-hours, at most `15868.000` MB RAM, and no GPU; details are in [`results/verified-slate-consensus/rejected-cycle-2026-08-30-survival-duration.json`](results/verified-slate-consensus/rejected-cycle-2026-08-30-survival-duration.json).

A third 30 August cycle rejected whole-user PairLogitPairwise and QuerySoftMax residual rankers, then found a small stable user-neighbor correction. A 60-neighbor exposure-overlap graph, trained only on April 8-21 interactions, selected weight `0.002` in every held-out user fold and improved all four folds. The promoted score is `0.612805`; the tracked accepted trainer takes `6.943` wall-seconds, `0.002244` CPU-hours, peaks at `4239.000` MB RAM, and uses no GPU. The complete cycle ledger is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-user-neighbor.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-user-neighbor.json).

A later 30 August cycle temporally selected a centered-outcome item co-exposure graph. It improved substantially over the old positive-only ItemKNN (`0.581189` versus `0.551791` after within-user ranking), but all four residual folds selected zero weight. Re-audited DCN and AutoInt predictions also selected zero, while numeric-bilinear interactions regressed in three folds at their fixed mean weight. The champion therefore remains `0.612805`; the successful checks used `187.700` wall-seconds, `0.052184` CPU-hours, at most `1221.422` MB RAM, and no GPU. Details are in [`results/verified-slate-consensus/rejected-cycle-2026-08-30-item-graph-interactions.json`](results/verified-slate-consensus/rejected-cycle-2026-08-30-item-graph-interactions.json).

A follow-up paired the strongest unused interaction families with the continuous watch-ratio auxiliary objective. AutoInt peaked at `0.607850` and regressed to `0.612791` cross-fitted; the low-rank Deep Cross Network peaked at `0.607275` and selected zero residual weight in every fold. These tracked runs used `117.321` wall-seconds, `0.078589` CPU-hours, at most `15970.719` MB RAM, and no GPU. The evidence is in [`results/verified-slate-consensus/rejected-cycle-2026-08-30-interaction-watch.json`](results/verified-slate-consensus/rejected-cycle-2026-08-30-interaction-watch.json).

The next cycle rejected causal randomized-feedback transfer, weekly recurrence priors, focal BCE, and several multi-task/user-balance controls. A later audit found that the apparent tree-only micro-gain used ordinal row codes instead of actual user IDs for held-out reporting; the corrected tree-only OOF score regressed, so that claim is explicitly invalidated in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-user-balanced-tree.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-user-balanced-tree.json). A corrected two-dimensional audit then selected the user-balanced CatBoost and watch-ratio DeepFM together at weights `(0.001875, 0.001875)` in all four user folds. Every held-out fold improved, raising the verified champion to `0.612830`; the joint audit used `19.204` wall-seconds, `0.005328` CPU-hours, `1210.094` MB peak RAM, and no GPU. Full evidence is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-joint-terminal-residual.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-joint-terminal-residual.json).

A YetiRankPairwise batch-slate model reached `0.612307` standalone and selected zero global residual weight, but four user folds selected weights `[0.78, 0.78, 0.78, 0.40]` toward it for users whose supplied slate forms exactly one 30-minute session. Their fixed mean `0.685` changes three pair orderings across three users, improves one held-out fold, leaves three unchanged, and raises the champion to `0.612834`. Training used `212.028` wall-seconds, `0.406794` CPU-hours, `3746.594` MB peak RAM, and no GPU; evidence is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-single-session-yeti.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-single-session-yeti.json).

A corrected zero-preferring scan of 20 frozen model families across 32 outcome-free structural regimes then selected a `-0.0275` median-consensus extrapolation for users with two or three sessions. Fold weights `[-0.03, -0.03, -0.02, -0.03]` are directionally consistent, the fixed recipe improves every fold or leaves it unchanged, and the champion rises to `0.612839`. The broad scan used `342.668` wall-seconds, `0.095170` CPU-hours, `1520.953` MB peak RAM, and no GPU; evidence is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-session-median.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-session-median.json).

Two extra deep YetiRank seeds and their rank ensemble improved diversity but failed the held-out fold gate. A shallower depth-6, strongly regularized YetiRank reached `0.612236` standalone, then supplied a stable `-0.0925` extrapolation for users in the highest quartile of April 19–21 training-row coverage. All four folds improve and the champion reaches `0.612858`. Training used `122.022` wall-seconds, `0.206093` CPU-hours, `3625.344` MB peak RAM, and no GPU; evidence is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-recent-coverage-yeti.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-recent-coverage-yeti.json).

The dashboard's deterministic demo can display scores up to `0.6250`; those values are synthetic workflow checks, not model-training evidence. The UI labels the entire demo campaign accordingly. Only results under `results/verified-*` are claimed as reproduced validation scores.

One ensemble attempt was deliberately invalidated after its runner process was terminated: inspection showed that the ensemble path would have ignored the proposed positive-example weight. The result was not scored or promoted, and the runner was corrected before the campaign continued. The hidden test was never accessed.

The original campaign evidence is in [`results/run-9ecfd2aa09`](results/run-9ecfd2aa09), the pairwise milestone is in [`results/verified-pairwise-blend`](results/verified-pairwise-blend), the DeepFM milestone is in [`results/verified-deep-blend`](results/verified-deep-blend), the retained lightweight checkpoint is in [`results/verified-temporal-deep-blend`](results/verified-temporal-deep-blend), and the current research-best evidence is in [`results/verified-slate-consensus`](results/verified-slate-consensus). Checkpoints, raw runner requests, logs containing local paths, and the dataset are excluded from Git.

The frozen recipe manifest, compressed validation scores, and 124,909-row organizer-schema validation export are checked into [`results/final-model`](results/final-model). The export is alignment evidence, not a hidden-test result. See [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) for the guarded one-time final procedure.

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
python3 -m pip install -r requirements-research.txt  # research-only tree stack
python3 scripts/train_batch_slate_meta.py            # rebuild final tree artifact
python3 scripts/verify_slate_consensus.py  # requires retained runtime score artifacts
python3 scripts/verify_slate_consensus.py --scores-output results/final-model/validation-scores.npz
python3 scripts/export_submission.py --help
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

results/verified-temporal-deep-blend/
  summary.json
  proposal.json
  metrics.json
  resource-usage.json

results/verified-slate-consensus/
  summary.json
  proposal.json
  metrics.json
  resource-usage.json

results/verified-session-consensus/
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
- `POST /api/run/start` with `{ "provider": "demo|gpt", "mode": "demo|kuairand", "limits": { "max_iterations": 50, "max_hours": 6, "convergence_epsilon": 0.002, "convergence_patience": 3 }, "bootstrap_verified": true }`
- `POST /api/run/continue` with a new `limits` object to retain the champion, evidence history, and cumulative compute accounting while opening another experiment/time budget.
- `POST /api/run/pause`, `/api/run/resume`, `/api/run/stop`, `/api/run/reset`
- `POST /api/steer` with `{ "instruction": "..." }`

The API binds to `127.0.0.1:8787` by default and only permits the local dashboard origins through CORS.

## Submission materials

- Devpost-ready project copy: [`docs/DEVPOST.md`](docs/DEVPOST.md)
- Judging report and resource totals: [`docs/JUDGING_REPORT.md`](docs/JUDGING_REPORT.md)
- Final output runbook: [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md)
- Requirement-by-requirement audit: [`docs/REQUIREMENTS_CHECKLIST.md`](docs/REQUIREMENTS_CHECKLIST.md)
- Machine-readable score/resource summary: [`results/project-summary.json`](results/project-summary.json)
- Validation-best recipe and output artifacts: [`results/final-model`](results/final-model)

## Limitations and future improvements

- The 0.612858 champion is a large offline ensemble with several frozen prerequisite artifacts; it is reproducible but not yet a compact production model. Distillation into one deployable scorer is the highest-value follow-up.
- Full-slate label-free corrections assume the complete candidate slate is available at inference, which matches the official batch evaluator but not every online serving environment.
- GPU telemetry must be supplied by a GPU-backed adapter; this CPU-only run used zero GPU-hours.
- The hidden test is intentionally untouched. Its guarded export should be run once only after the validation recipe and Git commit are frozen.
- KuaiRand-1k/27k remains an optional follow-up benchmark.

## Team contributions

- **Aadithyakk / Aadithya:** original autonomous-agent architecture, campaign hardening, trusted KuaiRand runner, challenge integration, and baseline research.
- **milkksthetic:** compute tracking, pairwise/DeepFM executors, campaign continuation, extensive leak-free ablations, consensus/routing research, verified evidence, and final completion pass.
