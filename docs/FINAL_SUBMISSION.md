# KuaiRand-Pure Final Submission Summary

## Selected model

The submitted model is the validation-best checkpoint from iteration 4 of the original KuaiLab autonomous campaign:

- NumPy Factorization Machine
- fields: user ID, video ID, author ID, tab, and train-quantile duration bucket
- latent dimension: 16
- learning rate: 0.001
- L2: 0.000001
- batch size: 8,192
- seed: 0
- positive-example loss weight: 2.75
- frozen final epoch: 8 (the validation-best checkpoint in the recorded run)

It is trained only on KuaiRand-Pure dates 20220408–20220421. The randomized log and other KuaiRand variants are not used.

## Validation result

| Model | GAUC | nDCG@5 | Primary | Δ GAUC | Δ nDCG@5 | Δ primary |
|---|---:|---:|---:|---:|---:|---:|
| Reproduced official FM | 0.667133 | 0.535806 | 0.601470 | — | — | — |
| Original KuaiLab champion | **0.669987** | **0.537574** | **0.603781** | **+0.002854** | **+0.001768** | **+0.002311** |

These are validation-only values. No hidden-test metric is claimed in this repository.

## Convergence and autonomy

The run proposed five iterations. Four produced validation scores and one failed. With `epsilon=0.002` and `N=3`, the best score in the last three scored iterations (`0.603781`) exceeds the best score before the window (`0.603493`) by `0.000288`, so the cumulative organizer rule declares convergence. The failed iteration counted toward the 50-iteration and wall-clock limits but did not advance or reset the scored window.

- iterations: 5 / 50
- manual interventions: 1
- stop reason: convergence
- retained checkpoint: iteration 4
- LLM input tokens: 6,800
- LLM output tokens: 18,002 (including 12,238 reasoning tokens)
- total API tokens: 24,802
- agent wall-clock: 479.67 seconds
- GPU-hours: 0

## Evidence

`results/run-9ecfd2aa09/` contains the sanitized original campaign:

- `summary.json`: trajectory, recovery, resources, and final choice
- `baseline/`: official reproduced metrics and epoch history
- `iteration-001` through `iteration-005`: hypothesis/proposal, candidate source, unified diff, metrics where scored, and training history

Iteration 2 was invalidated after termination because the ensemble path would have ignored the champion's positive weight. It was never scored or promoted. The prior champion was retained and the runner contract was corrected before iteration 3.

## Reproduce and export

Install `requirements.txt`, then run:

```bash
python3 scripts/final_model.py \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  --starter-dir /absolute/path/to/kuairand-starter-kit \
  reproduce-validation
```

For the one-time blind output:

```bash
python3 scripts/final_model.py \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  make-submission --confirm-final

python3 scripts/final_model.py \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  check-submission
```

The blind loader selects only feature columns from test rows and never loads or evaluates their outcomes. The checker verifies the header, 0-based contiguous row IDs, row count, source alignment, and finite numeric scores.
