# Paper-guided research plan

KuaiLab now combines four complementary agent patterns instead of asking an LLM
to guess another hyperparameter configuration:

- [AIDE](https://arxiv.org/abs/2502.13138): every result is a search-tree node
  with a parent, and each proposal applies one atomic `draft`, `improve`,
  `ablate`, `ensemble`, or `debug` operation.
- [MLE-STAR](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a9619dd0f0d54a5cf7734add1dc38cd1-Abstract-Conference.html): proposals identify the one
  component being refined—features, model, loss, training, ensemble, or reward—
  and use prior ablations to avoid changing several things at once.
- [Self-Evolving Recommendation System](https://arxiv.org/html/2602.10226):
  April 8–14 training plus April 15–21 validation is the cheap offline screen;
  April 22–28 is the slower confirmation tier. Only screened candidates read the
  confirmation labels.
- [DS-Agent](https://proceedings.mlr.press/v235/guo24b.html): a validated local
  method-card library records the method, fit, smallest useful ablation, risk,
  and whether an idea is unattempted, partial, research-only, or exhausted.

The planner must generate exactly three alternatives—one exploit, one explore,
and one innovate—then select one alternative and attach it to an existing tree
node. The organizer evaluator remains the only authority on scores.

## Current research frontier

| Priority | Method | Status | Smallest useful experiment |
| --- | --- | --- | --- |
| 1 | [Relative Advantage Debiasing](https://arxiv.org/abs/2508.11086) | Attempted—rejected | The auxiliary, α=0 control, pure RAD head, small head mixtures, and four held-out residual folds were measured. The control was stronger and no residual generalized. |
| 2 | [DVR Watch-Time-Gain](https://arxiv.org/abs/2208.05190) | Attempted—rejected | A paper-faithful duration-conditioned WTG auxiliary plus duration-adversarial learning passed the screen, then lost all three confirmation metrics and failed two frozen-champion residual folds. |
| 3 | [UMRE monotonic ensemble](https://arxiv.org/abs/2508.07613) | Attempted—rejected | A train-only monotonic fusion lost primary and GAUC to its matched linear consensus; no personalized gate was opened. |
| 4 | [SetRank](https://arxiv.org/abs/1912.05891) | Attempted—rejected | Low-rank self-attention lost primary and nDCG@5 to its matched mean-pool control on the locked screen. |
| 5 | [MaskNet](https://arxiv.org/abs/2102.07619) | Attempted—rejected | Its +0.000449 train-only gain reversed at confirmation and three of four champion-residual folds regressed. |
| 6 | [FinalMLP](https://arxiv.org/abs/2304.00902) | Attempted—rejected | Its +0.000756 train-only gain reversed at confirmation and every held-out residual fold lost primary. |
| 7 | [NeuralNDCG](https://arxiv.org/abs/1906.04262) | Attempted—rejected | Its isolated objective differential replicated at confirmation, but a fixed champion blend lost all three metrics and regressed in two folds. |
| 8 | [Contextual Distillation Model](https://arxiv.org/abs/2406.09021) | Attempted—rejected | Static context produced a tempting post-hoc maximum, but leave-one-fold-out selection and the learned context gate retained the unchanged champion. |
| 9 | [Conservative Doubly Robust learning](https://jiawei-chen.github.io/paper/CIKM23-CDR.pdf) | Research-only | Reconsider only if a random-exposure screen is rebuilt; its published setting does not match this challenge's standard-log confirmation distribution. |

LambdaMART/YetiRank, generic MMoE/PLE, counterfactual watch-time likelihoods,
pairwise/listwise fine-tuning, and the tested mean-pooled slate model are marked
exhausted or partial in `backend/kuailab/method_cards.json`; the planner is told
not to repeat their exact configurations.

## Promotion protocol

1. Screen a single atomic change on April 15–21 after training only on April
   8–14.
2. Reject weak candidates without reading April 22–28. An ensemble gets a small
   `0.002` screen allowance because diversity can be useful despite a weaker
   standalone score.
3. Confirm survivors on April 22–28 and compare them with the frozen `0.612858`
   champion.
4. For gains below `0.0001`, require both GAUC and nDCG@5 to be non-decreasing.
5. Store the selected parent, alternatives, typed configuration, diff, metrics,
   artifacts, recovery events, CPU/GPU time, wall time, RAM, and token use.

The first measured fast/slow audit is in
`results/final-model/fast-slow-slate-smoke.json`. It demonstrates why both tiers
are required: the slate candidate gained `+0.004450` internally but was neutral
after confirmation and held-out residual auditing.

The next paper-guided run tested RAD-UV. It passed the fast gate with a
`0.615556` long-view-head score, but its fixed 5% champion residual scored
`0.612856567`, below the frozen champion. An otherwise identical α=0 control
was stronger standalone (`0.604018` versus `0.603648`), pure RAD-head ranking
fell to `0.583969` internally, and all four held-out residual folds regressed.
See `results/final-model/rad-auxiliary-audit.json`.

The duration-debiasing follow-up implemented the central mechanism from
[DVR](https://arxiv.org/abs/2208.05190) rather than copying the supplied sketch
literally. The organizer's `long_view` target remained unchanged; a secondary
Watch-Time-Gain target standardized truncated play time using distributions fit
only on training rows, and a gradient-reversal head discouraged duration
reconstruction. This avoids the sketch's arbitrary 80% label, its
duration-reintroducing `watch_ratio * log(duration)` reward, and its cross-user
batch pairs. The selected configuration gained `+0.000148` primary on April
15–21, but lost `-0.000063` primary, `-0.000063` GAUC, and `-0.000063` nDCG@5
against its matched control on April 22–28. Its locked 5% frozen-champion
residual scored `0.612857640`, below `0.612858057`, and failed folds 2 and 3.
The screen and confirmation consumed `35.670` wall-seconds and `0.015121`
CPU-hours in total, used no GPU, and peaked at `942.406` MB RAM. Evidence is in
`results/duration-debiasing/summary.json`.

The parallel interaction-and-ranking sweep in `results/parallel-methods`
then tested UMRE-lite, SetRank attention, MaskNet, FinalMLP, outcome-free MMR,
near-tie hard-negative mining, and a bounded NeuralNDCG residual. MaskNet and
FinalMLP demonstrated why the slow tier is mandatory: both had positive matched
screens and then reversed at confirmation. MMR produced a global confirmation
micro-gain of `+0.0000126`, but failed the predeclared all-metric gate in two
actual-user-ID folds. No candidate was promoted; the exact `0.612858057`
champion remains frozen.

## Calibrated-ranking and interaction wave

The next parallel wave tested ten additional paper-guided changes with exact
matched controls: RCR, personalized direct-GAUC optimization, SBCR-lite,
position-aware distillation, conditional watch-time quantiles, JRC,
confidence-aware ranking, AFN, behavior-bias projection, and EulerNet. Model
selection used outcomes no later than April 14; April 15–21 was locked for
screening. Only RCR passed that screen and was therefore allowed to read April
22–28. No run parsed April 29+ outcomes.

RCR at alpha `0.0005` was the only repeatable standalone improvement. It gained
`+0.000186` primary on the screen and `+0.000117` on confirmation, with GAUC
and nDCG@5 positive both times. Its predeclared 5% residual into the actual
champion nevertheless lost `0.000002384` primary and regressed in three of four
actual-user-ID folds, so it was not promoted. PDAOM traded `+0.000144` nDCG@5
for `-0.000388` GAUC; JRC's locked primary gain shrank to `+0.000012` while
GAUC fell. SBCR, position KD, AFN, and bias projection selected their zero or
control arms. CQE, confidence ranking, and EulerNet all reversed or regressed
on the locked screen.

Tracked trainers used `346.197` aggregate wall-seconds and `501.143`
CPU-seconds (`0.139206` CPU-hours), no GPU, and at most `1490.812` MB RAM in
any one process. Because runs overlapped, aggregate trainer wall-time is not
end-to-end elapsed time and process telemetry does not measure combined RAM
across concurrent workers. The complete decision and per-method telemetry are
in `results/calibrated-ranking/summary.json`. The exact `0.612858057` champion
remains frozen.

## Contextual-distillation follow-up

[CDM](https://arxiv.org/abs/2406.09021), a Kuaishou contextual-distillation
method, motivated a focused attempt to stabilize the earlier MMR micro-gain.
A predeclared 360-configuration audit varied only static author, music, video
type, and tag context, penalty, and top-pool size. The largest global result was
`0.612908304`, but nDCG@5 fell by `0.000024080` and fold 0 regressed. Twenty-four
configurations preserved every metric in every fold, but they moved very few
rows: the strongest was author-only top-five reranking at `0.612860441`, a
`+0.000002384` gain. Three of four leave-one-fold-out selectors chose the
unchanged champion; the remaining selection lost all metrics on its held-out
fold.

The fixed author-only rule was then checked on two earlier out-of-time windows.
It was neutral on April 14 and gained `+0.000013351` on April 15–21, so it failed
the predeclared requirement for a positive result in both windows. Finally, a
small gradient-boosted context gate used only score margins, slate structure,
and static redundancy. Its best all-metric OOF arm gained `+0.000008821`, below
the locked `0.000010` threshold; zero was selected and confirmation stayed
closed for that branch. Successful evidence runs consumed `171.763` aggregate
wall-seconds and `0.052853` CPU-hours, used no GPU, and peaked at `497.344` MB
RAM. Two report-serialization recovery runs repeated the grid before the final
artifact, adding roughly `281.6` observed wall-seconds. See
`results/context-distillation/summary.json`.
