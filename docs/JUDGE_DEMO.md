# KuaiLab three-minute judge demo

This is the deterministic presentation path for TikTok TechJam 2026. It is a view over checked-in validation evidence, not a synthetic campaign and not a hidden-test result.

## Reproduce the demo

Requirements: Node.js 22.13+ and Python 3.11+.

```bash
npm install
npm run verify:demo
npm run local
```

Open `http://localhost:3000`, then select **3-minute walkthrough**. The walkthrough itself needs no OpenAI key or dataset download. The backend enables the live control-room portion; checked-in showcase evidence remains deterministic.

Before recording, use a 1440 × 900 browser window, collapse unrelated browser chrome, and verify that the dashboard says **Champion primary · verified** rather than **Demo primary · simulated**.

## Exact 3:00 narration

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
