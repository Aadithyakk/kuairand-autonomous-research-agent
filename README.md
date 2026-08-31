# KuaiLab v2

KuaiLab is a local-first autonomous research control room for the KuaiRand-Pure `long_view` recommendation task. It uses the OpenAI Responses API with `gpt-5.6-sol` to design one falsifiable experiment at a time, runs experiments through a sealed benchmark adapter, retains the validation-best champion, and records the evidence behind every decision.

The previous repository is not a dependency. This implementation was rebuilt from scratch.

## Judge quickstart

```bash
npm install
npm run verify:demo
npm run local
```

Open `http://localhost:3000` and select **3-minute walkthrough**. This path presents the checked-in **0.612858** validation champion, **+0.011389 absolute** lift over the reproduced baseline, the separate five-iteration autonomous run, a real recovered failure, a rejected deployment decision, compute telemetry, and the exact source artifacts. It does not need an OpenAI key or dataset download. The live campaign controls underneath use the local backend and optional real-mode dependencies.

The timed narration and recording checklist are in [`docs/JUDGE_DEMO.md`](docs/JUDGE_DEMO.md). Run `npm run verify:demo` before presenting; it fails if the judge-facing numbers drift from the champion manifest, experiment ledger, worker smoke, or frozen-score checksum.

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

## Online/prequential research teacher

A separate completion-safe online experiment reaches **`0.723415` primary**
(`0.821572` GAUC and `0.625259` nDCG@5) on the 124,909 public-validation
impressions. It retrains on time blocks and exposes an interaction outcome only
after `time_ms + play_time_ms`, with same-timestamp feedback excluded. Candidate
corrections are retained only when primary, GAUC, and nDCG@5 are nonnegative in
all four actual-user-ID folds.

This number is intentionally not mounted as the frozen static champion: it is a
validation-selected online/prequential teacher rather than an untouched hidden-
test estimate. The portable feature builders, block trainers, pairwise trainer,
gated residual search, exact verifier, and run instructions are documented in
[`docs/PREQUENTIAL_TEACHER.md`](docs/PREQUENTIAL_TEACHER.md). The accepted
37-stage lineage and final score hash are in
[`results/prequential-online-teacher/manifest.json`](results/prequential-online-teacher/manifest.json).

## Champion-mounted autonomous training

The 0.612858 validation champion is no longer metrics-only bootstrap evidence. Real campaigns expose a trusted `champion_residual_blend` experiment family to the LLM. Each such iteration:

1. verifies the frozen score archive against `results/final-model/manifest.json` and its SHA-256 checksum;
2. retrains a fresh pointwise FM, pairwise FM, DeepFM blend, or temporal DeepFM blend on the official April 8–21 training block;
3. converts the candidate and champion to stable within-user ranks;
4. blends or extrapolates the candidate at a typed weight in `[-0.25, 0.25]`;
5. evaluates on April 22–28, saves the candidate checkpoints and blended scores in the isolated iteration workspace, and promotes only a positive primary-score gain.

The hidden split remains unavailable to this adapter. Set **Mount verified 0.612858 champion** in the dashboard; the agent can then choose this family autonomously. Operator steering is optional, for example: `Try champion_residual_blend with a pairwise_fm candidate and a conservative positive weight.`

The worker also exposes a `slate_context_deepfm` champion candidate. Unlike the
row-wise residuals, it batches complete user slates, pools a permutation-invariant
DeepSets context, and includes label-free session/repeat structure. It selects an
epoch on matched seven-day slates (April 8–14 → April 15–21), refits both weeks,
then computes the April 22–28 confirmation score, reducing repeated confirmation-
split tuning.

The `rad_deepfm` family implements the paper-guided Relative Advantage
Debiasing ablation. It estimates smoothed watch-time quantiles only inside the
training window: by video ID, and by user ID within four duration bins. Support-
weighted probit fusion supplies an auxiliary target to DeepFM, while only the
`long_view` head produces ranking scores by default. `rad_score_weight` exposes
a controlled paper-faithful ablation from the binary head toward the predicted
relative-advantage head. Epoch selection and refitting use the same nested
temporal protocol as the slate candidate.

The first RAD audit rejected promotion. Its auxiliary version passed the
train-only screen at `0.615556`, but the 5% confirmation residual reached only
`0.612856567`; an otherwise identical α=0 control was stronger standalone
(`0.604018` versus `0.603648`). Pure RAD-head ranking scored `0.583969` on the
fast screen, and every held-out residual fold regressed. The five runs consumed
`72.076` wall-seconds, `0.026406` CPU-hours, at most `1132.125` MB RAM, and no
GPU. Evidence is in
[`results/final-model/rad-auxiliary-audit.json`](results/final-model/rad-auxiliary-audit.json).

A separate paper-faithful DVR/Watch-Time-Gain audit kept the official
`long_view` target and added only a train-derived duration-conditioned reward
head plus gradient-reversal duration adversary. It gained `+0.000148` against
its matched control on the April 15–21 screen, but reversed to `-0.000063` on
April 22–28. The fixed 5% champion residual scored `0.612857640` and failed two
of four actual-user-ID folds, so the `0.612858057` champion remains unchanged.
The runnable experiment and decision record are
[`scripts/train_dvr_wtg.py`](scripts/train_dvr_wtg.py) and
[`results/duration-debiasing/summary.json`](results/duration-debiasing/summary.json).

A Kuaishou [Contextual Distillation Model](https://arxiv.org/abs/2406.09021)
follow-up then tested whether learned candidate-set context could stabilize the
earlier MMR micro-gain. A broad exploratory grid reached `0.612908304` but
reduced nDCG@5. Its best all-fold-safe rule reached only `0.612860441` and did
not survive leave-one-fold-out selection; an independently trained context gate
selected zero at its locked screen threshold. The frozen champion therefore
remains `0.612858057`. Full evidence and recovery compute are recorded in
[`results/context-distillation/summary.json`](results/context-distillation/summary.json).

Promotion has a stability guard: any primary gain below `0.0001` must also leave
both GAUC and nDCG@5 non-decreasing. Larger primary gains still follow the
organizer's combined metric. This prevents top-five regressions from being
promoted as floating-point-scale average improvements.

Real campaigns use a two-level evaluator. The fast loop trains only on April
8–14 and screens on April 15–21. A weak candidate is recorded as `screened_out`
without reading April 22–28 labels. Only candidates that match or beat the fixed
train-only FM screen baseline enter the slow confirmation loop; ensemble
operators receive a narrow `0.002` diversity allowance. Both tiers' resource
usage is retained in the same experiment journal.

The controller now keeps an AIDE-style experiment tree, applies MLE-STAR-style
single-component refinement, asks for one exploit/explore/innovate alternative
before every selection, and retrieves DS-Agent-style method cards with explicit
attempt status and risk. See [`docs/PAPER_METHODS.md`](docs/PAPER_METHODS.md) and
[`backend/kuailab/method_cards.json`](backend/kuailab/method_cards.json).

The real arm64 integration smoke retrained a fresh FM in `7.001` seconds, used `0.002496` CPU-hours and `541.359` MB peak RAM, and correctly retained the `0.612858` champion when the candidate reached only `0.596280`. Its auditable result is checked in at [`results/final-model/autonomous-worker-smoke.json`](results/final-model/autonomous-worker-smoke.json).

The launcher also selects the most capable available runner interpreter instead of the first NumPy-only runtime. On this arm64 host it selected PyTorch `2.10.0` and completed the formerly blocked DeepFM residual in `32.833` seconds; its `0.612568` result was correctly rejected.

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

A later diagnostic localized 53.1% of recoverable nDCG loss to users with seven
or more sessions. A new full-slate DeepSets/DeepFM executor therefore modeled
permutation-invariant user context plus outcome-free session and repeat features.
Matched-week temporal selection and refitting improved a single seed from
`0.602125` to `0.603570` standalone; its apparent fixed residual was only
`+0.0000063` and three held-out user folds regressed. A three-seed rank ensemble
reached `0.603915` standalone but its fixed residual was `-0.0000001`, so the
`0.612858` champion remains frozen. The four tracked trainers used `113.919`
wall-seconds, `0.044665` CPU-hours, at most `1369.594` MB RAM, and no GPU.
Evidence is recorded in `results/final-model/*slate-context*.json`.

A corrected zero-preferring scan of 20 frozen model families across 32 outcome-free structural regimes then selected a `-0.0275` median-consensus extrapolation for users with two or three sessions. Fold weights `[-0.03, -0.03, -0.02, -0.03]` are directionally consistent, the fixed recipe improves every fold or leaves it unchanged, and the champion rises to `0.612839`. The broad scan used `342.668` wall-seconds, `0.095170` CPU-hours, `1520.953` MB peak RAM, and no GPU; evidence is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-session-median.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-session-median.json).

Two extra deep YetiRank seeds and their rank ensemble improved diversity but failed the held-out fold gate. A shallower depth-6, strongly regularized YetiRank reached `0.612236` standalone, then supplied a stable `-0.0925` extrapolation for users in the highest quartile of April 19–21 training-row coverage. All four folds improve and the champion reaches `0.612858`. Training used `122.022` wall-seconds, `0.206093` CPU-hours, `3625.344` MB peak RAM, and no GPU; evidence is in [`results/verified-slate-consensus/accepted-cycle-2026-08-30-recent-coverage-yeti.json`](results/verified-slate-consensus/accepted-cycle-2026-08-30-recent-coverage-yeti.json).

A parallel paper-guided sweep then evaluated UMRE-lite, SetRank attention,
MaskNet, FinalMLP, outcome-free MMR diversity, near-tie hard-negative mining,
and a bounded NeuralNDCG residual. MaskNet and FinalMLP passed their train-only
matched screens but reversed at confirmation. MMR moved the global frozen score
to `0.612871`, yet failed the predeclared all-metric gate in two actual-user-ID
folds and was not promoted. The NeuralNDCG term improved its matched alpha-zero
control by `+0.000307` primary, but its complete residual remained below the
untouched base. The exact `0.612858057` champion therefore remains frozen; the
scripts, compute telemetry, and rejection evidence are in
[`results/parallel-methods`](results/parallel-methods).

The isolated NeuralNDCG objective differential was also tested separately. It
replicated against its own locked April 22–28 base (`+0.000221` primary), but a
predeclared 5% blend into the actual champion lost `0.0000019` primary and
regressed in two user folds. This confirms that the differentiable top-five
gradient contains signal, but not yet one that complements the much stronger
frozen ensemble.

A further parallel wave tested ten calibrated-ranking, debiasing, reward, and
interaction methods against exact matched controls: RCR, personalized direct
GAUC, SBCR-lite, position-aware KD, conditional watch-time quantiles, JRC,
confidence-aware ranking, AFN, behavior-bias projection, and EulerNet. RCR was
the only method to improve all three metrics on both the April 15–21 screen
(`+0.000186` primary) and locked April 22–28 confirmation (`+0.000117`). Its
fixed 5% residual into the actual champion still lost `0.000002384` primary and
regressed in three of four user folds, so the exact `0.612858057` champion is
unchanged. The other nine methods were rejected without opening confirmation.
Tracked trainers used `346.197` aggregate wall-seconds, `0.139206` CPU-hours,
no GPU, and at most `1490.812` MB RAM in one process. Full metrics, compute, and
integrity evidence are in
[`results/calibrated-ranking/summary.json`](results/calibrated-ranking/summary.json).

One aborted FinalMLP preflight parsed a single April 29 row into transient
memory before its date-boundary assertion stopped the process. That value was
never printed, saved, scored, trained on, or used. The corrected clean process
parsed outcome fields only for April 22–28, and the incident remains explicitly
recorded in the checked-in confirmation report.

The dashboard's deterministic demo can display scores up to `0.6250`; those values are synthetic workflow checks, not model-training evidence. The UI labels the entire demo campaign accordingly. Only results under `results/verified-*` are claimed as reproduced validation scores.

One ensemble attempt was deliberately invalidated after its runner process was terminated: inspection showed that the ensemble path would have ignored the proposed positive-example weight. The result was not scored or promoted, and the runner was corrected before the campaign continued. No hidden-test outcome was ever scored, trained on, selected against, or used for promotion; the single transient parse from the later disclosed aborted preflight was discarded immediately.

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

- `action` (`baseline`, `screen_baseline`, `screen`, or `experiment`)
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
