# Devpost project description

## Inspiration

Recommendation research is usually a manual loop: inspect a dataset, invent a model change, wait for training, compare metrics, diagnose failures, and repeat. That process is slow, difficult to audit, and especially risky on KuaiRand-Pure, where exposure bias, sequential behavior, duplicate impressions, and a strict temporal split make accidental leakage easy. We built KuaiLab to turn that loop into a bounded, evidence-first autonomous system.

## What it does

KuaiLab is an LLM-powered research agent and live control room for the KuaiRand-Pure `long_view` task. On every iteration it reads the current champion and prior failures, proposes one falsifiable experiment, writes an auditable typed configuration and code diff, runs it through a sealed organizer-compatible evaluator, measures GAUC and nDCG@5, tracks time/CPU/RAM/GPU/token use, promotes only a genuine validation improvement, reflects on the result, and chooses the next experiment.

The campaign obeys the official limits: at most 50 experiments, at most six wall-clock hours, and convergence after three consecutive iterations without a primary-score gain greater than 0.002. The hidden test is excluded from research. Worker failures trigger one automatically routed lower-resource retry; both attempts and their compute are retained in the run log. Pause, resume, stop, continuation, and operator steering are available from the dashboard, and every intervention is counted.

## How we built it

- OpenAI Responses API with `gpt-5.6-sol`, high reasoning, and strict JSON-schema proposals.
- Python campaign engine with atomic state, append-only events, isolated workspaces, retry/routing, convergence enforcement, and a fail-closed subprocess adapter.
- The official KuaiRand starter-kit loader and evaluator, with exact train/validation date and row-count checks.
- NumPy FM and BPR executors, PyTorch DeepFM variants, and offline CatBoost/XGBoost ranking research.
- A Vinext/React dashboard that exposes the campaign state, metrics, budget, compute, trace, and iteration table in real time.
- Reproducible validation evidence, a final recipe manifest, and a guarded exporter for the required `row_id,user_id,video_id,score` schema.

## Dataset and task

KuaiRand-Pure only; no external training data or pretrained weights trained on challenge outcomes were used. Training is 8–21 April 2022 (1,141,112 rows), validation is 22–28 April (124,909 rows), and the hidden test is 29 April–8 May (170,588 rows). Relevance is `long_view`; the primary metric is the mean of user-grouped GAUC and nDCG@5.

## Results

The official validation baseline is 0.6016 primary (0.6674 GAUC, 0.5357 nDCG@5). Our independently reproduced baseline is 0.601470. The current validation-best recipe reaches **0.612858 primary**, with **0.682354 GAUC** and **0.543362 nDCG@5**: +0.011389 absolute and +1.89% relative over the reproduced baseline. The hidden test remains untouched.

## Challenges

The hardest lesson was that apparent micro-gains are easy to manufacture accidentally. Listwise objectives, reciprocal-rank fusion, causal Transformers, CWM, xDeepFM, Monte Carlo dropout, propensity weighting, continuous margin loss, and many graph/sequence variants either regressed or failed a disjoint-user stability gate. We invalidated one earlier tree claim after discovering an incorrect fold identifier and kept the correction in the public ledger. The final ensemble is conservative because every accepted terminal correction had to improve all four held-out user folds or leave them unchanged.

## What we learned

On this task, globally calibrated pointwise models are strong, while nDCG@5 is sensitive to a tiny number of unstable top-item swaps. Outcome-free full-slate structure, training-only coverage, and carefully gated expert disagreement helped more reliably than aggressive rank losses. Autonomy is valuable only when its evidence and stopping rules are stricter than its optimism.

## What's next

The next practical step is to package the large research ensemble into a smaller distillation target, add container-level GPU telemetry, and run the guarded hidden-test export exactly once after the organizers authorize final submission. KuaiRand-1k/27k is a natural bonus benchmark after the Pure submission is frozen.
