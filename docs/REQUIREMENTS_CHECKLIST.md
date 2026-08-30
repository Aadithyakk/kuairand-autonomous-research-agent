# Project 2 completion checklist

| Official requirement | Implementation / evidence | Status |
| --- | --- | --- |
| Reproduce organizer FM baseline | `scripts/kuairand_runner.py`; `results/run-9ecfd2aa09/baseline` | Complete |
| Autonomous read → hypothesize → implement → train → evaluate → reflect loop | `backend/kuailab/engine.py`, dashboard live trace | Complete |
| Improve over baseline | 0.612858 vs reproduced 0.601470 | Complete |
| Minimal manual intervention | Original autonomous run records one intervention | Complete |
| Robust recovery/retry/routing | Planner retry, lower-resource worker retry, retained prior champion | Complete |
| Correct `long_view`, GAUC, nDCG@5 task | Runner hard-codes target and official evaluator | Complete |
| Fixed temporal split and row counts | Runner rejects any split-size mismatch | Complete |
| 50-iteration hard ceiling | Global budget survives campaign continuation | Complete |
| Six-hour wall-clock ceiling | Global cumulative wall budget survives continuation | Complete |
| ε=0.002, N=3 convergence | Forced for every real KuaiRand campaign | Complete |
| Validation-best checkpoint selection | Positive-gain promotion; 0.612858 evidence bootstrap | Complete |
| Continue training from validation-best model | `champion_residual_blend` retrains typed candidates against the checksum-verified frozen champion | Complete |
| Hidden test accessed only once | Excluded from development loader; guarded receipt on final export | Ready; intentionally not run |
| Required CSV schema | `scripts/export_submission.py`; checked validation artifact | Complete |
| Per-iteration hypothesis and code diff | `proposal.json`, `candidate.py`, `candidate.diff`, `iteration-log.json` | Complete |
| Metrics, errors, and recovery events | State, append-only JSONL, iteration log, stderr/stdout | Complete |
| Manual intervention count | Campaign state and resource summary | Complete |
| LLM token, wall, iteration, GPU accounting | `results/project-summary.json`, `docs/JUDGING_REPORT.md` | Complete |
| Public-repository README sections | Overview, setup, reproduction, limitations, contributions | Complete |
| Devpost written description | `docs/DEVPOST.md` | Complete |
| Final model output/checkpoint | Frozen manifest, score archive, validation schema export | Complete for validation |
| Results table and baseline delta | README, final runbook, judging report | Complete |
| Optional video | Detailed report supplied instead | Not required |
| Bonus KuaiRand-1k/27k | Future work | Optional |

The only deliberately unexecuted step is the organizer's one-time hidden-test run. Running it during development would violate the brief; `docs/FINAL_SUBMISSION.md` contains the final-only procedure.
