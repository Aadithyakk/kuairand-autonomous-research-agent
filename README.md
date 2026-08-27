# KuaiRand Autonomous Research Agent

A local-first autonomous ML research loop for the TikTok TechJam recommender-systems challenge, with a human research cockpit for metrics, generated experiments, evidence, failures, memory, and steering.

The system treats KuaiRand-Pure as a controlled benchmark for an agent that can repeatedly:

1. load and verify the organizer baseline;
2. diagnose the current solution;
3. retrieve relevant literature and past lessons;
4. propose and rank feature, loss, model, data, validation, and ensemble experiments;
5. execute one bounded experiment;
6. evaluate it using the immutable metric contract;
7. reflect, remember, recover, branch, or converge;
8. keep a complete audit trail for human review.

## What is included

- Verified Notebook 01 baseline reproduction.
- Kaggle-ready model lab containing FM, leakage-safe history features, LightGBM, LambdaRank, CatBoost, BPR, DIN-lite, diagnostics, and rank blending.
- A generic planner → code generator → safety gate → external evaluator loop. The planner creates hypotheses instead of selecting from a fixed model sequence.
- Searchable literature and organizer-lesson cards.
- An OpenAI-compatible research model for proposing, implementing, and reviewing experiments.
- A deterministic validation model used only to test the orchestration—not as the competition policy.
- Persistent experiment state, convergence rules, failure recovery, and intervention accounting.
- A legacy command-worker bridge for the existing LambdaRank, BPR, and DIN notebook implementations. These are reusable tools, not the autonomous agent's menu.
- Local JSON API for live state and control.
- Responsive human dashboard with overview, experiments, audit events, literature, run controls, and steering.

## Architecture

```text
dataset + baseline + immutable evaluator
                  |
          diagnose current state
                  |
     local evidence + past experiments
             + research LLM
                  |
       propose several new hypotheses
                  |
 generic cost/risk acquisition function
                  |
      generate code → static safety gate
                  |
       isolated experiment process
                  |
       external metrics + artifacts
                  |
       retrospective experiment memory
                  |
          dashboard + human steering
```

The model owns research choices and experiment implementations. Deterministic controller code owns only the safety boundary, data contract, budget, evaluator, audit trail, champion promotion, and stopping rules. This is the key separation: the controller does not know that an experiment must be LambdaRank, BPR, DIN, or any other predefined family.

## What has actually been validated

There are two separate validations, because combining them would overstate the result:

1. **Autonomous mechanism validation** uses a small controlled ranking benchmark. The agent must create a hypothesis, generate code, survive a deliberately unsafe first implementation being rejected, use the failure as memory, recover with a valid implementation, obtain externally computed metrics, and promote a better champion with zero human interventions.
2. **KuaiRand readiness validation** checks the real archive schema, official split/label/metric contract, and the five-seed baseline artifact. It does not claim that a generated challenger has already beaten the official FM on KuaiRand.

Run both:

```bash
python3 -m research_agent.cli validate-agent
python3 -m research_agent.cli validate-kuairand
```

The machine-readable reports are written under `runtime/` (which is intentionally not committed).

## Quick start

Requirements: Python 3.10+, Node 22+, and the Iteration 0 baseline artifact.

```bash
cp .env.example .env
mkdir -p artifacts
# Place iteration_000_baseline_artifacts.zip in artifacts/

python3 -m research_agent.cli init
python3 -m research_agent.cli serve
```

In another terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard connects to the agent at `http://127.0.0.1:8765` by default.

## Existing notebook worker

The original dashboard controller can still be exercised in deterministic simulation mode:

```bash
python3 -m research_agent.cli run --reset
```

Simulation metrics are clearly labelled and are never competition results. This path and `knowledge/actions.json` are retained for dashboard demos and as a library of executable prior approaches; they are not evidence of open-ended autonomy.

For real notebook execution, copy `configs/default.json`, set `executor_mode` to `command`, and ensure the KuaiRand-Pure dataset is discoverable by `outputs/kuairand_autonomous_research_lab.ipynb`. The command worker currently supports these dynamically selected branches:

- leakage-safe history features with LightGBM LambdaRank;
- BPR with actually exposed negatives;
- causal DIN-lite sequence modelling.

Other catalog entries remain notebook research candidates until a worker adapter is registered.

## Research model

Run an OpenAI-compatible endpoint such as vLLM or llama.cpp, then change the LLM section in the configuration:

```json
{
  "mode": "openai_compatible",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct"
}
```

`OpenAICompatibleResearchModel` uses this endpoint for three distinct roles: research planning, complete experiment-program generation, and evidence-based reflection. Invalid planner/code responses fail closed. `ScriptedValidationModel` is only a test double proving that the controller is not secretly choosing the model family.

## API

Read-only:

- `GET /api/health`
- `GET /api/state`
- `GET /api/literature`
- `GET /api/actions`

Controls:

- `POST /api/run/start`
- `POST /api/run/pause`
- `POST /api/run/resume`
- `POST /api/run/stop`
- `POST /api/steer` with `{ "message": "..." }`

Every human intervention and controller decision is written to the audit log.

## Validation

```bash
python3 -m unittest discover -s tests -v
python3 -m research_agent.cli validate-agent
python3 -m research_agent.cli validate-kuairand
cd dashboard && npm run build
```

## Data and leakage policy

- Development uses the official training and validation dates only.
- Generated experiments receive only an explicit public workspace; evaluator labels remain outside it.
- The KuaiRand research cutoff is 2022-04-28. Later rows are never scored during development.
- Training-row history features must be causal.
- Validation outcomes never enter feature construction.
- Same-row post-exposure feedback is not used as an inference feature.
- The organizer evaluator remains outside LLM-authored code.

The starter kit and KuaiRand dataset are not redistributed in this repository. Obtain them from the organizers and respect their licence and competition rules.

## Current status

The generic autonomous outer loop is implemented and passes its controlled end-to-end validation, including safety rejection, recovery, external evaluation, memory, and champion promotion. The real KuaiRand archive and reproduced baseline pass readiness checks. The official FM validation mean remains **0.601572** primary (GAUC **0.667400**, nDCG@5 **0.535744**).

What is not yet claimed: no LLM-generated challenger has been run end-to-end against the full KuaiRand validation set, and the dashboard API still uses the legacy controller. Connecting the generic controller to the sanitized KuaiRand worker/Kaggle compute and running a budgeted research campaign are the next milestones.
