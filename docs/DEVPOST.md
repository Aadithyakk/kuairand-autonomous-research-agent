# KuaiLab: An Auditable Autonomous ML Research Agent

## Inspiration

Recommendation research is repetitive: inspect evidence, form a hypothesis, modify code, train, evaluate, and decide what to try next. KuaiLab turns that loop into a controlled autonomous system while keeping every decision inspectable. The goal was not simply to tune one model, but to show an agent completing a real recommender-system research campaign under fixed data, metric, iteration, and time constraints.

## What it does

KuaiLab is a local-first research control room for the required KuaiRand-Pure `long_view` benchmark. GPT-5.6 Sol receives the current champion, prior iterations, failures, remaining budget, and a strict executor contract. It returns one falsifiable, machine-readable experiment. A trusted adapter—not generated code—applies the typed change to the official NumPy Factorization Machine, trains on 8–21 April, and reports the official GAUC and nDCG@5 on validation from 22–28 April.

The system then validates the result, promotes it only if it beats the retained champion, records the full evidence, reflects, and continues. It enforces the 50-iteration and six-hour caps and the organizer's cumulative convergence window. Failed runs are recorded and recovered from without losing the champion; they consume budget but do not alter the convergence window.

The browser dashboard shows the live six-stage loop, champion trajectory, current hypothesis, proposal diff, token usage, wall-clock usage, events, errors, and operator controls. The final export retrains the frozen validation-best configuration and writes the required `row_id,user_id,video_id,score` file without loading test outcomes.

## Original autonomous run

The five proposals were:

1. Increase the positive-example loss weight from 1.0 to 2.0: accepted, primary `0.603493`.
2. Average three independently seeded models: runner terminated; inspection found the ensemble path would ignore the proposed class weight, so the result was invalidated and the runner contract was corrected.
3. Increase the weight from 2.0 to 2.5: accepted, primary `0.603684`.
4. Refine the weight from 2.5 to 2.75: accepted and retained, primary `0.603781`.
5. Increase the weight to 3.0: rejected, primary `0.603643`; the campaign converged.

| Result | GAUC | nDCG@5 | Primary | Δ primary |
|---|---:|---:|---:|---:|
| Reproduced FM baseline | 0.667133 | 0.535806 | 0.601470 | — |
| KuaiLab champion | **0.669987** | **0.537574** | **0.603781** | **+0.002311** |

The campaign used 24,802 model tokens, 479.67 seconds of agent wall-clock, five of the allowed 50 iterations, zero GPU-hours, and one manual intervention. All proposals, generated diffs, metrics, epoch histories, and the error/recovery record are included in the repository.

## How it addresses the challenge

KuaiLab automates the full iterative research decision loop rather than running a fixed hyperparameter sweep. Each next experiment depends on the observed campaign evidence. The model provider cannot read labels or execute arbitrary host code; it sees aggregate validation metrics and acts through a small typed contract. This separation gives the agent useful autonomy while keeping data access and execution auditable.

The original agent identified class imbalance as a cheap, direct training target. It then reacted to both successful results and a failed ensemble attempt, routing toward a lower-cost local refinement instead of stalling. The resulting model is modest, but the campaign demonstrates hypothesis generation, controlled implementation, official evaluation, champion management, recovery, convergence, and reproducible finalization as one coherent system.

## Built with

- OpenAI Responses API with GPT-5.6 Sol for experiment proposals and reflection
- Python 3.11 and NumPy for the trusted Factorization Machine executor
- The organizer-supplied KuaiRand starter kit and official `evaluate.py`
- React, TypeScript, Vinext, Vite, and Tailwind CSS for the local dashboard
- VS Code/Codex and the command line for implementation and verification

## Data and assets

Only the required KuaiRand-Pure dataset is used for model fitting. Training is restricted to the standard log from 8–21 April. Validation uses 22–28 April. The randomized log and the KuaiRand-1k/27k variants do not enter Pure training. Test rows from 29 April–8 May are used only once to generate aligned scores, and the export path never loads their outcomes.

## Challenges and lessons

The hardest design problem was balancing autonomy with safe execution. Directly running arbitrary generated code would be flexible but difficult to audit. KuaiLab instead stores generated code as evidence and applies a validated typed proposal through a trusted executor. This made the action space narrower, but failures became predictable and recoverable.

The ensemble failure was also valuable. The process terminated, and review showed a semantic mismatch: that executor path would not preserve the current positive-example weight. KuaiLab discarded the untrustworthy result, retained the champion, documented the event, and continued. That is the behavior needed for long autonomous experiments: a failed iteration must not corrupt the research state.

## Limitations and next steps

The original executor exposes only a five-field Factorization Machine and a small set of typed changes, so the agent cannot explore richer temporal histories, multi-task learning, or deep architectures. The system also needed one manual correction after the ensemble failure. Future work would expand the trusted action library, add isolated subprocess/container execution, checkpoint resume, automated semantic contract tests, and fully autonomous recovery. Those additions are intentionally outside this submission, which preserves the original system and campaign.

## Contribution

Aadithya Kumar designed and implemented the system, integrated the benchmark, ran and reviewed the campaign, built the dashboard, and prepared the submission. GPT-5.6 Sol acted as the autonomous experiment researcher. OpenAI Codex assisted engineering and documentation.
