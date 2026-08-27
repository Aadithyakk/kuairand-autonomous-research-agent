# KuaiRand Autonomous Research Agent

A local-first autonomous ML research loop for the TikTok TechJam recommender-systems challenge, with a human research cockpit for live metrics, experiment decisions, evidence, failures, and steering.

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
- Cost-aware adaptive research policy; there is no fixed model sequence.
- Searchable literature and organizer-lesson cards.
- Optional OpenAI-compatible local LLM planner with deterministic fallback.
- Persistent experiment state, convergence rules, failure recovery, and intervention accounting.
- Command-worker bridge that executes agent-selected LambdaRank, BPR, or DIN branches through the Kaggle notebook.
- Local JSON API for live state and control.
- Responsive human dashboard with overview, experiments, audit events, literature, run controls, and steering.

## Architecture

```text
dataset + baseline + immutable evaluator
                  |
          diagnose current state
                  |
     local evidence + past experiments
          + optional LLM planner
                  |
      cost/risk/evidence search policy
                  |
        isolated experiment worker
                  |
       metrics + slices + artifacts
                  |
       retrospective experiment memory
                  |
          dashboard + human steering
```

The LLM advises and explains. Deterministic code enforces the action catalog, data boundary, compute budget, evaluator, and stopping rules.

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

## Running the autonomous loop

The default configuration uses deterministic simulation so the complete orchestration, recovery, convergence, API, and dashboard can be tested without the 1.1M-row dataset:

```bash
python3 -m research_agent.cli run --reset
```

Simulation metrics are clearly labelled and are never competition results.

For real notebook execution, copy `configs/default.json`, set `executor_mode` to `command`, and ensure the KuaiRand-Pure dataset is discoverable by `outputs/kuairand_autonomous_research_lab.ipynb`. The command worker currently supports these dynamically selected branches:

- leakage-safe history features with LightGBM LambdaRank;
- BPR with actually exposed negatives;
- causal DIN-lite sequence modelling.

Other catalog entries remain research candidates until a worker adapter is registered. In command mode, the policy automatically excludes actions without an executable adapter.

## Optional local LLM

Run an OpenAI-compatible endpoint such as vLLM or llama.cpp, then change the LLM section in the configuration:

```json
{
  "mode": "openai_compatible",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct"
}
```

The system continues with a deterministic policy if the model endpoint is unavailable or returns invalid JSON.

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
cd dashboard && npm run build
```

## Data and leakage policy

- Development uses the official training and validation dates only.
- The hidden test set is never accessed during research.
- Training-row history features must be causal.
- Validation outcomes never enter feature construction.
- Same-row post-exposure feedback is not used as an inference feature.
- The organizer evaluator remains outside LLM-authored code.

The starter kit and KuaiRand dataset are not redistributed in this repository. Obtain them from the organizers and respect their licence and competition rules.

## Current status

This repository is an executable research-agent foundation and live cockpit. Baseline reproduction and orchestration are verified. Real model execution is wired for the three branches above; live web retrieval, additional multi-task/watch-time workers, and remote Kaggle job synchronization are the next implementation milestones.
