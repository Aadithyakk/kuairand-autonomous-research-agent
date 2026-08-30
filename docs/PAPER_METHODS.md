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
| 1 | [Relative Advantage Debiasing](https://arxiv.org/abs/2508.11086) | Unattempted | Construct train-only, smoothed user/video watch-time quantiles and use their fused relative-preference value as an auxiliary DeepFM target. Rank only with the `long_view` head. |
| 2 | [UMRE monotonic ensemble](https://arxiv.org/abs/2508.07613) | Unattempted | Cross-fit monotonic transforms over three diverse frozen score streams, followed by a strongly regularized context gate. |
| 3 | [SetRank](https://arxiv.org/abs/1912.05891) | Partial | Replace the rejected mean-pooled DeepSets context with one small permutation-invariant self-attention block, keeping the temporal protocol and BCE fixed. |
| 4 | [Conservative Doubly Robust learning](https://jiawei-chen.github.io/paper/CIKM23-CDR.pdf) | Research-only | Reconsider only if a random-exposure screen is rebuilt; its published setting does not match this challenge's standard-log confirmation distribution. |

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
