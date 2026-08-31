# Submission Checklist

## Devpost

- [ ] Paste and personalize `docs/DEVPOST.md`.
- [ ] Add the public GitHub repository URL.
- [ ] Add screenshots and, optionally, a three-minute dashboard demo.
- [ ] Confirm the participant/contribution wording.

## Repository

- [x] Original autonomous system isolated from later modeling additions.
- [x] Setup and reproducibility instructions included.
- [x] Limitations and contribution statement included.
- [x] Original per-iteration proposals, diffs, metrics, histories, and recovery evidence included.
- [x] Cumulative convergence rule implemented and unit-tested.
- [ ] Publish this submission branch/repository publicly.

## Final artifacts

- [x] Run `reproduce-validation` and confirm primary is exactly `0.6037807465` in the verified environment.
- [x] Run `make-submission --confirm-final` once without inspecting test outcomes.
- [x] Run `check-submission` and obtain `PASS`.
- [x] Confirm `submission.csv` has 170,588 data rows and the exact header `row_id,user_id,video_id,score`.
- [ ] Upload `submission.csv`, `original-fm-checkpoint.npz`, `encoding.json`, `training.json`, and `submission-manifest.json`.
- [ ] Upload or link `docs/FINAL_SUBMISSION.md`.

## Safety review

- [x] Training uses dates 20220408–20220421 only.
- [x] Validation selection uses dates 20220422–20220428 only.
- [x] Randomized log is not a training input.
- [x] KuaiRand-1k/27k are not Pure training inputs.
- [x] Blind export does not load or score test outcomes.
- [ ] Do not run any separate evaluator on the final test file.
