# KuaiLab — Original Autonomous Research Agent

KuaiLab is a local-first autonomous ML research agent for the KuaiRand-Pure `long_view` ranking task. This submission preserves the original five-iteration system and recorded campaign. It does **not** include the later DeepFM, BPR, feature-lab, teacher, or prequential additions.

The agent runs a visible six-stage loop—inspect, hypothesize, implement, train, evaluate, and reflect—using GPT-5.6 Sol to propose one typed experiment at a time. A sealed NumPy benchmark adapter executes the proposal, the official evaluator reports validation GAUC and nDCG@5, and the validation-best checkpoint is retained. Every proposal, code diff, metric, failure, recovery, resource counter, and promotion decision is logged.

## Original campaign result

The recorded KuaiRand-Pure campaign on 29 August 2026 used only the official train split (8–21 April) for fitting and validation (22–28 April) for model selection.

| Model | GAUC | nDCG@5 | Primary | Δ primary vs reproduced FM |
|---|---:|---:|---:|---:|
| Reproduced official FM | 0.667133 | 0.535806 | 0.601470 | — |
| Original KuaiLab champion | **0.669987** | **0.537574** | **0.603781** | **+0.002311** |

The agent proposed five iterations. Iteration 4—one seed-0 FM with positive-example weight `2.75`—was retained. The cumulative organizer rule (`epsilon=0.002`, `N=3`) is satisfied after the fifth proposal: the best of the last three scored iterations improves on the best score before that window by only `0.000288`. Iteration 2 failed and consumed budget, but correctly did not advance or reset the scored convergence window.

Recorded resources: **24,802 API tokens**, **479.67 seconds** agent wall-clock, **5/50 iterations**, **0 GPU-hours**, and **1 manual intervention**. The complete sanitized evidence is in [`results/run-9ecfd2aa09`](results/run-9ecfd2aa09).

## Safety and split policy

- Training uses only `log_standard_4_08_to_4_21_pure.csv`, dates 20220408–20220421.
- Validation labels are used only for dates 20220422–20220428.
- `log_random_4_22_to_5_08_pure.csv` is never used for training.
- KuaiRand-1k and KuaiRand-27k are not used to train the Pure model.
- The blind export reads only `user_id`, `video_id`, `date`, `duration_ms`, and `tab` from test rows. It does not load or evaluate test outcomes.
- Final output has exactly `row_id,user_id,video_id,score` and 170,588 aligned rows.

## Requirements

- Python 3.11+
- NumPy 2.4.2
- Node.js 22.13+ and npm (dashboard only)
- The supplied KuaiRand-Pure `data/` directory
- The supplied `kuairand-starter-kit/` directory for official validation evaluation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
npm install
```

No API key is required to reproduce the recorded champion. A new autonomous campaign requires `OPENAI_API_KEY` in the backend environment; the key is never sent to the browser or committed.

## Reproduce the validation champion

```bash
python3 scripts/final_model.py \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  --starter-dir /absolute/path/to/kuairand-starter-kit \
  reproduce-validation
```

This retrains the frozen iteration-4 configuration for eight epochs, evaluates only the official validation split, and writes the checkpoint, encoder, training record, and `validation-results.json` under `results/final-model/`. The expected primary score is `0.6037807465` (small last-digit differences may occur across NumPy/platform builds).

## Create and check the one-time blind submission

Run this only for the final challenge export. It does not score the test labels.

```bash
python3 scripts/final_model.py \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  make-submission --confirm-final

python3 scripts/final_model.py \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  check-submission
```

The generated artifacts are `results/final-model/submission.csv`, `original-fm-checkpoint.npz`, `encoding.json`, `training.json`, and `submission-manifest.json`.

## Run the autonomous control room

```bash
npm run local
```

Open the shown local URL (normally `http://localhost:3000`). The demo planner and synthetic benchmark provide a no-cost smoke test. With KuaiRand-Pure and the starter kit extracted under `external/`, the real sealed adapter is discovered automatically. A fresh campaign can use the GPT provider after `OPENAI_API_KEY` is exported.

Useful checks:

```bash
npm run test:backend
npm run lint
npm run build
```

## Evidence layout

```text
backend/kuailab/                 autonomous engine, provider, state, benchmark adapter
scripts/kuairand_runner.py       trusted train/validation executor
scripts/final_model.py           frozen champion reproduction and blind export
results/run-9ecfd2aa09/          original five-iteration evidence
  summary.json
  baseline/
  iteration-001/ ... iteration-005/
docs/DEVPOST.md                  ready-to-paste written project description
docs/FINAL_SUBMISSION.md         results and resource summary
docs/SUBMISSION_CHECKLIST.md     final hand-in checklist
```

Each scored iteration contains its proposal/hypothesis, generated candidate, unified diff, metrics, and epoch history. The invalidated iteration retains its proposal and diff; the recovery is documented in `summary.json`.

## Limitations and future work

The original agent deliberately used a narrow trusted action space: a five-field NumPy Factorization Machine and typed configuration changes. This made execution auditable and inexpensive, but limited architectural and feature exploration. One ensemble attempt also required a manual runner correction, so the recorded campaign was not intervention-free. With more time, the executor could safely expose richer feature/model primitives, add subprocess isolation and checkpoint resume, and eliminate the remaining manual recovery path without changing the train/validation/test boundary.

## Contribution

**Aadithya Kumar** — system architecture, challenge integration, autonomous campaign execution, validation, dashboard, and submission packaging. GPT-5.6 Sol served as the autonomous experiment proposer; OpenAI Codex assisted implementation and documentation. Update this section if the Devpost team includes additional participants.

See [`docs/DEVPOST.md`](docs/DEVPOST.md) for the full project description and [`docs/FINAL_SUBMISSION.md`](docs/FINAL_SUBMISSION.md) for the compact results hand-in.
