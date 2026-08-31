# Reproducing the 0.723415 prequential teacher

The repository now separates three claims that are easy to accidentally mix:

1. **Exact artifact replay.** The 37 accepted promotions, their reports, causal
   feature caches, source scores, and 20 frozen static rankers are hashed and
   published as one GitHub Release asset. Replaying those files reproduces
   `GAUC=0.821571529`, `nDCG@5=0.625259161`, and mean `0.723415375` on the
   22--28 April public development period.
2. **Source retraining.** The logistic and CatBoost source generators have
   explicit parameters in `configs/prequential_teacher.lock.json`. Seeded SGD
   is deterministic. Seeded CatBoost is checked numerically because exact ZIP
   bytes can still vary with CPU, OS, and thread scheduling.
3. **Final chronological evaluation.** 29 April--8 May is reserved and has not
   been evaluated. The development score above is validation-selected and must
   not be described as an untouched-test estimate.

## Quick reproduction

Use CPython 3.13.6, install the exact package closure, and download the original
KuaiRand-Pure data into `external/KuaiRand-Pure/data`:

```bash
python3.13 -m venv .venv-prequential
source .venv-prequential/bin/activate
python -m pip install -r requirements-prequential.lock.txt
python scripts/reproduce_prequential_teacher.py reproduce --download --strict-environment
```

`reproduce` performs the following operations through one entry point:

- validates the 37-stage lock;
- verifies all six dataset files against SHA-256 without redistributing them;
- verifies all 169 release files against SHA-256;
- mutates future labels in a synthetic stream and confirms earlier predictions
  are byte-identical;
- confirms every stage's `champion` is the preceding stage's `selected` array;
- confirms each report's source, transform, gate, and scalar weight;
- recomputes all three final metrics from the original public-period labels.

The default extracted-artifact directory is
`runtime/prequential-teacher-release`. Override it with
`KUAI_PREQUENTIAL_ARTIFACT_ROOT` or `--artifact-root`.

## Inspect before downloading

```bash
python scripts/reproduce_prequential_teacher.py doctor
```

CI uses this structural mode because it intentionally has neither the licensed
dataset nor the large release asset. For a fail-closed local audit, use:

```bash
python scripts/reproduce_prequential_teacher.py doctor \
  --require-dataset --require-artifacts --strict-environment
```

## Retrain an individual source

After downloading the artifact bundle and building the eight causal feature
caches, a recorded source can be rerun by stage number:

```bash
python scripts/reproduce_prequential_teacher.py retrain-source --stage 14
```

Stages 1--29 point directly to their locked pairwise-logistic or CatBoost
generator. Stages 30--37 use small assembled source caches; both each assembled
cache and its underlying model output are published and hashed. Use `verify`
for exact replay of those assemblies.

## Causality guarantee

For an impression at time `t` with play duration `p`, its outcome availability
is:

```text
available_at = t + max(p, 0)
```

A block starting at `b` can train only on `available_at < b`. Equality is
excluded, so simultaneous feedback is invisible. The CatBoost and pairwise
trainers call the shared primitive in `scripts/prequential_causality.py`.

Run the guards independently with:

```bash
python scripts/reproduce_prequential_teacher.py test-causality
```

## Reserved final period

The lock reserves 29 April--8 May. The local helper can export only a strict
allowlist of non-outcome columns:

```bash
python scripts/reproduce_prequential_teacher.py reserve-holdout \
  --output runtime/holdout-inputs.csv
```

It does not read `long_view`, play time, engagement labels, or dwell outcomes,
and it does not calculate a score. Final evaluation should use either a frozen
static scorer or an externally sealed streaming evaluator that reveals feedback
only after its availability time. Retrospectively splitting 22--28 April would
not create a genuinely untouched test because that period was already used for
many model and routing choices.

## Locks and publication

- `configs/prequential_teacher.lock.json`: all 37 source/transform/gate/weight
  decisions, exact source generators, periods, causality, and expected metrics.
- `requirements-prequential.lock.txt`: exact Python package closure.
- `results/prequential-online-teacher/checksums.json`: dataset, static model,
  causal cache, source, report, and promotion-output SHA-256 values.
- GitHub Release `prequential-teacher-v1`: the artifact bundle. The original
  dataset is excluded for licensing and distribution hygiene.

Maintainers can rebuild the deterministic release asset with:

```bash
python scripts/reproduce_prequential_teacher.py bundle \
  --work-root /path/to/experiment/work \
  --runtime-root runtime \
  --output prequential-teacher-artifacts-v1.tar.gz
```
