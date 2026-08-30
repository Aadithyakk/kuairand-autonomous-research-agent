# KuaiLab research replay & judge demo

This is the deterministic presentation path for TikTok TechJam 2026. It is a view over checked-in validation evidence, not a synthetic campaign and not a hidden-test result.

## Reproduce the demo

Requirements: Node.js 22.13+ and Python 3.11+.

```bash
npm install
npm run verify:demo
npm run local
```

Open `http://localhost:3000`. The default **Agent Research Replay** uses a deterministic 2:30 story reconstructed from checked-in reports. Press **Play story**; use **Audit** to pause and inspect report extracts, source fingerprints, and the clickable research tree. **Go live** opens the existing control room without starting a campaign. The previous five-part walkthrough remains under **Project overview**.

Replay needs no API key, backend, dataset download, or active training. Run `npm run dashboard -- --host 127.0.0.1 --port 3000` for the UI alone. On Windows, use `npm.cmd` if PowerShell blocks npm, and `runtime\train-env\Scripts\python.exe scripts/verify_judge_demo.py` for the original evidence verifier.

Before recording, verify the KPI strip says **Recorded champion · primary**. Never present the live synthetic smoke-test scores as model validation.

## Research replay · 2:30

| Time | Canvas | Key message |
| --- | --- | --- |
| 0:00 | Inspect | Explain the temporal windows and recorded validation counts. Missing profiling diagnostics are explicitly unavailable. |
| 0:18 | Research | Open the recorded RCR paper citation; no fake search activity. |
| 0:35 | Hypothesize | Show the predeclared four-weight experiment and matched control. |
| 0:52 | Implement | Show the objective change and locked training configuration. |
| 1:10 | Train | Show actual six-epoch training BCE, not a simulated training job. |
| 1:26 | Evaluate: screen | The candidate improves its matched base on 15–21 April. |
| 1:44 | Evaluate: confirm | The gain repeats on 22–28 April; integration still needs testing. |
| 2:02 | Reflect | The 5% champion residual regresses and only one user fold passes all metrics. Reject; retain the champion. |

The headline `0.601470 → 0.612858` improvement predates this RCR experiment. This wave made **zero promotions**. It would be misleading to animate that historical lift as the result of RCR. Decision cards summarize recorded observations and editorial explanations, not private chain-of-thought or a timestamped agent transcript. Playback time is presentation pacing; resource counters report saved measurements.

Every stage links to its evidence. Audit exposes seven allowlisted report extracts and original-file SHA-256 fingerprints; **Download evidence bundle** downloads the same JSON used by the UI. The bundle excludes raw outcome rows and model binaries. Source hashes identify content, not independent experimental validity. Confidence intervals and unrecorded search/profiling events are not fabricated.

After changing source reports, run `npm run replay:sync`, then `npm run test:replay`. Review narrative claims before accepting a changed experiment outcome. `npm run test:replay` checks freshness, metric relationships, replay transitions, each stage's server rendering, and the research tree. Run `npm run build` before presenting. The optional social preview uses localhost by default; set a trusted `NEXT_PUBLIC_SITE_URL` if this UI is later hosted.

## Optional Project overview · 3:00 narration

### 0:00–0:25 — The challenge

> Recommendation research is still a manual loop: form an idea, edit code, wait for training, compare metrics, and try again. KuaiLab turns that into a bounded autonomous workflow for KuaiRand-Pure. The target is long view, and the score balances per-user GAUC with nDCG at five.

Show step 1. Point to the train/validation windows, then the reproduced baseline and verified champion.

### 0:25–1:00 — The autonomous loop

> The LLM chooses what to investigate, but deterministic tools decide whether it worked. Each iteration inspects evidence, states a falsifiable hypothesis, produces an auditable configuration and diff, trains through a sealed evaluator, measures the official metrics and compute, then reflects. The campaign is capped at fifty experiments and six hours, and failures receive only one bounded recovery retry.

Show step 2. Pause on the real arm64 worker telemetry.

### 1:00–1:35 — Innovation and problem insight

> This is not blind hyperparameter search. KuaiLab keeps an experiment tree, retrieves reusable method cards, and proposes an exploit, explore, and innovate branch. On KuaiRand, we learned that a handful of unstable top-five swaps can erase globally good calibration. The retained recipe therefore combines calibrated pointwise models, training-only preferences, label-free slate structure, and conservative user-regime routing.

Show step 3. Point to the ten-method research wave and the zero unsafe promotions.

### 1:35–2:15 — Evidence over optimism

> Here is the most important behavior. Regression-Compatible Ranking improved its matched base on both a train-only screen and locked confirmation. But a predeclared residual into the much stronger champion lost 0.0000024 and regressed across user folds. KuaiLab rejected it automatically, kept the 0.612858 champion, and saved the failure as reusable memory. A useful research agent must know when not to deploy.

Show step 4. Pause on **RETAIN** and the resource telemetry.

### 2:15–2:45 — Reproducibility and honesty

> Every headline number resolves to checked-in JSON evidence. This consistency command verifies the champion metrics, metric formula, experiment-wave totals, worker telemetry, artifact presence, hidden-test claim, and frozen-score checksum. The reported lift is 1.89 percent over our reproduced baseline on 124,909 validation impressions. It is not a hidden-test claim.

Show step 5. Point to `npm run verify:demo`, the artifacts, and the honest limitations.

### 2:45–3:00 — Live product close

Select **Open live control room**.

> Underneath is the live campaign: champion metrics, current hypothesis, six-stage progress, append-only trace, official budget, compute, and every accepted or rejected iteration. KuaiLab makes autonomous model research faster, safer, and inspectable.

End on the iteration table with **Validation-best checkpoint retained · hidden test untouched** visible.

## Judge questions

**Is 0.6250 your score?** No. That is only the deterministic synthetic smoke-test ceiling. The verified KuaiRand validation champion is `0.612858057`.

**Did the LLM directly read labels or execute arbitrary code?** No. The planner produces typed hypotheses and patches; the trusted benchmark adapter owns data access, training, evaluation, and resource limits.

**Why is a rejected experiment featured?** It demonstrates the promotion policy, contamination boundary, and value of experiment memory. A system that only displays wins is not evidence of safe autonomy.

**Can this run without a GPU?** Yes. The verified worker smoke and the displayed experiment wave used zero GPU-hours on Apple arm64.

**What remains before submission?** Record this flow, upload an unlisted public three-minute YouTube video, place its URL in Devpost, and run the hidden test only through the guarded final procedure when authorized.

## Recording checklist

- Run `npm run verify:demo` immediately before recording.
- Use the five walkthrough steps in order; do not start a synthetic campaign.
- Keep the video at or below three minutes.
- Show the URL bar once so `localhost:3000` is visible.
- Do not expose an API key, local dataset path, or hidden-test file.
- Upload to YouTube with public or unlisted visibility and verify the link in a signed-out window.
