# Completion-safe prequential teacher

The public-validation teacher reaches:

| Metric | Value |
| --- | ---: |
| GAUC | `0.821571529` |
| nDCG@5 | `0.625259161` |
| Mean | `0.723415375` |

This is a validation-selected **online/prequential** result. It is not a frozen
static-model score and is not an untouched hidden-test estimate. Each block
model may use earlier feedback only after that feedback becomes observable.

## Causality rule

For row `i`, its outcome becomes available at:

```text
availability_i = time_ms_i + max(play_time_ms_i, 0)
```

A model scoring a block beginning at time `t` trains only on rows satisfying
`availability_i < t`. Events at the same timestamp are invisible to each
other. Rows after 28 April 2022 are excluded from the public experiment.

## Setup

Install the research dependencies and place the KuaiRand-Pure files under
`external/KuaiRand-Pure/data`:

```bash
python -m pip install -r requirements-research.txt
export KUAI_PREQUENTIAL_WORKDIR="$PWD/runtime/prequential-teacher"
mkdir -p "$KUAI_PREQUENTIAL_WORKDIR"
```

Generated feature matrices, model scores, and reports remain under the ignored
runtime directory. Override `KUAI_PREQUENTIAL_WORKDIR` to use another location.

## Build the causal feature caches

```bash
python scripts/build_causal_streaming_random_features.py
python scripts/build_causal_random_watch_features.py
python scripts/build_causal_random_user_state.py
python scripts/build_causal_random_action_state.py
python scripts/build_causal_random_transition_features.py
python scripts/build_causal_decayed_random_features.py
python scripts/build_prequential_standard_feedback_features.py
python scripts/build_prequential_immediate_feedback_features.py
```

The random-panel builders impose their documented lag before exposing random
feed outcomes. The immediate standard-feed builder uses the stricter
completion-time rule above.

## Train a block-retrieved CatBoost challenger

The champion input must be an NPZ file containing a `selected` score array in
public-validation row order. Frozen ranker outputs used by
`KUAI_PAIRWISE_STATIC_MODELS=1` must already exist in `runtime`; they are
produced by the repository's existing base-ranker scripts.

```bash
KUAI_CATBOOST_CHAMPION=runtime/prequential-teacher/current_champion.npz \
KUAI_CATBOOST_OUTPUT=runtime/prequential-teacher/daily_pairlogit_scores.npz \
KUAI_CATBOOST_REPORT=runtime/prequential-teacher/daily_pairlogit_report.json \
KUAI_CATBOOST_MODELS=ranker \
KUAI_CATBOOST_RANK_LOSS=PairLogitPairwise \
KUAI_CATBOOST_BLOCK_HOURS=24 \
KUAI_CATBOOST_DEPTH=5 \
KUAI_CATBOOST_ITERATIONS=220 \
KUAI_PAIRWISE_IMMEDIATE=1 \
python scripts/prequential_block_catboost.py
```

Supported experiments also include pointwise `classifier` models and ranking
losses such as `YetiRankPairwise`, `QueryRMSE`, and `QuerySoftMax` when the
installed CatBoost backend supports them.

## Train the pairwise logistic challenger

```bash
KUAI_PAIRWISE_CHAMPION=runtime/prequential-teacher/current_champion.npz \
KUAI_PAIRWISE_OUTPUT=runtime/prequential-teacher/pairwise_scores.npz \
KUAI_PAIRWISE_REPORT=runtime/prequential-teacher/pairwise_report.json \
KUAI_PAIRWISE_BLOCK_HOURS=6 \
KUAI_PAIRWISE_ALPHAS=0.003 \
KUAI_PAIRWISE_IMMEDIATE=1 \
python scripts/prequential_daily_pairwise_logistic.py
```

## Screen a residual with four user folds

Create an NPZ residual cache whose keys are candidate names and whose arrays
follow public-validation row order, then run:

```bash
KUAI_TWIN_CHAMPION=runtime/prequential-teacher/current_champion.npz \
KUAI_RESIDUAL_CACHE=runtime/prequential-teacher/candidate_sources.npz \
KUAI_RESIDUAL_TRANSFORMS=raw,z,rank \
KUAI_TWIN_EXPANDED_GATES=1 \
KUAI_TWIN_OUTPUT=runtime/prequential-teacher/gated_scores.npz \
KUAI_TWIN_REPORT=runtime/prequential-teacher/gated_report.json \
python scripts/gated_twin_residual_search.py
```

A correction is promoted only when primary, GAUC, and nDCG@5 are nonnegative
relative to the current champion in all four `user_id % 4` folds. The search
supports raw predictions, within-user z-scores, and within-user ordinal ranks.

## Verify the frozen score artifact

The generated score file is intentionally not committed. If it is placed at
`runtime/prequential-teacher/best_verified_online_scores.npz`, verify its hash
and exact metrics with:

```bash
python scripts/verify_prequential_teacher.py
```

The accepted 37-stage lineage, final hash, routing weights, and evaluation
caveat are recorded in
`results/prequential-online-teacher/manifest.json`.
