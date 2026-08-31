# Final submission runbook

## Frozen validation result

| Model | GAUC | nDCG@5 | Primary | Δ vs reproduced baseline |
| --- | ---: | ---: | ---: | ---: |
| Official organizer FM baseline | 0.667400 | 0.535700 | 0.601550 | +0.000000 |
| Reproduced organizer FM baseline | 0.667133 | 0.535806 | 0.601470 | +0.000000 |
| KuaiLab validation-best slate ensemble | **0.682354** | **0.543362** | **0.612858** | **+0.011389** |

The hidden test has not been accessed. `results/final-model/validation-submission.csv` is a validation-alignment artifact in the organizer's required schema, not a hidden-test score claim.

## Reproduce validation evidence

Install `requirements-research.txt`, place the official starter kit and KuaiRand-Pure files under `external/`, and rebuild the prerequisite score artifacts documented in the README. Then run:

```bash
python3 scripts/verify_slate_consensus.py \
  --scores-output results/final-model/validation-scores.npz

python3 scripts/export_submission.py \
  --interactions external/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv \
  --scores results/final-model/validation-scores.npz \
  --split validation \
  --output results/final-model/validation-submission.csv \
  --manifest results/final-model/validation-submission.manifest.json
```

The exporter checks the exact 124,909-row count, finite scores, alignment length, strictly increasing zero-based `row_id`, and the exact CSV header.

## Single final hidden-test export

Do this only after the validation-best recipe is frozen and the team has decided this is the one final submission. First generate hidden-test scores from the frozen model without fitting or tuning on 29 April–8 May. Then run:

```bash
python3 scripts/export_submission.py \
  --interactions external/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv \
  --scores /absolute/path/to/frozen-hidden-scores.npz \
  --split hidden-test \
  --confirm-final-hidden-test \
  --output /absolute/path/to/final-submission.csv
```

The explicit flag is mandatory. The first hidden export writes `runtime/hidden-test-access.json`; later attempts fail closed. Do not delete that receipt to rerun the hidden test.

## Preflight checklist

- Freeze the Git commit and `results/final-model/manifest.json`.
- Verify there are 170,588 hidden rows and no missing or non-finite score.
- Confirm the model reads only training outcomes and label-free evaluation identities/metadata.
- Confirm the CSV header is `row_id,user_id,video_id,score`.
- Record the hidden-test access receipt, submission checksum, total resource usage, and final result once.
