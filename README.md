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
- Searchable literature and organizer-lesson cards used as lightweight retrieval-augmented generation (RAG).
- An OpenAI Responses research brain using GPT-5.6 Sol for planning, coding, repair, review, and reflection.
- Repeated planner → coder → methodological reviewer roles; revised code must be reviewed again and explicitly approved.
- A trusted Kaggle worker and packager that receive generated code but never receive the OpenAI API key.
- A deterministic validation model used only to test the orchestration—not as the competition policy.
- Persistent experiment state, convergence rules, failure recovery, and intervention accounting.
- A legacy command-worker bridge for the existing LambdaRank, BPR, and DIN notebook implementations. These are reusable tools, not the autonomous agent's menu.
- Local JSON API for live state and control.
- Responsive human dashboard with overview, experiments, audit events, literature, steering, time budgets, compute detection, and selectable execution cards.

## Architecture

```text
dataset + baseline + immutable evaluator
                  |
          diagnose current state
                  |
 local literature RAG + past experiments
         + Sol research planner
                  |
       propose several new hypotheses
                  |
 generic cost/risk acquisition function
                  |
        Sol coder → semantic reviewer
             |
        bounded repair loop
             |
       explicit approval + static gate
                  |
 training-only temporal FM tournament
                  |
 sanitized Kaggle worker / isolated process
                  |
       external metrics + artifacts
                  |
       retrospective experiment memory
                  |
          dashboard + human steering
```

The model owns research choices and experiment implementations. Deterministic controller code owns only retrieval boundaries, safety, data contracts, budgets, evaluator access, audit trails, champion promotion, and stopping rules. The controller does not prescribe LambdaRank, BPR, DIN, or another fixed model sequence.

## What has actually been validated

There are two separate validations, because combining them would overstate the result:

1. **Autonomous mechanism validation** uses a small controlled ranking benchmark. The agent must create a hypothesis, generate code, survive a deliberately unsafe first implementation being rejected, use the failure as memory, recover with a valid implementation, obtain externally computed metrics, and promote a better champion with zero human interventions.
2. **KuaiRand readiness validation** checks the real archive schema, official split/label/metric contract, and the five-seed baseline artifact.
3. **Real generated-program pilots** validate code generation, repair, Kaggle dispatch, trusted external metrics, and negative-result memory. The current controller additionally rejects candidates on training-only chronological FM-anchor screens before allowing external confirmation.

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
# Add your own OPENAI_API_KEY and KAGGLE_API_TOKEN to .env when needed.
# Or export them directly in your shell; no key is stored in repository files.
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

## Research model and credentials

The default production configuration uses the OpenAI Responses API:

```json
{
  "mode": "openai_responses",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-5.6-sol",
  "api_key_env": "OPENAI_API_KEY",
  "reasoning_effort": "medium"
}
```

Set keys only at runtime:

```bash
export OPENAI_API_KEY="your-key"
export KAGGLE_API_TOKEN="your-token"
```

`OpenAICompatibleResearchModel` uses the endpoint for planning, complete program generation, methodological review, repair, and evidence-based reflection. Invalid, unsafe, or unapproved programs fail closed. Code-valid candidates must then survive two training-only temporal tournaments against a fixed FM anchor before private external validation is used. A local OpenAI-compatible endpoint remains supported with `mode: openai_compatible`.

The supplied keys are never embedded in generated programs or Kaggle kernels. `.env`, runtime workspaces, Kaggle downloads, and virtual environments are ignored by Git.

## Literature retrieval

`knowledge/literature.json` is currently a lightweight RAG index of paper/organizer cards: title, year, URL, tags, claim, and cautions. Each iteration constructs a query from the benchmark diagnosis and past experiment lessons, retrieves the most relevant cards, and records their IDs in the decision audit.

This is not yet full-paper RAG. Live web discovery, PDF ingestion, passage chunking, embeddings, citation verification, and source refresh are intentionally listed as future work. Generated training code has no internet access; any future web or paper retrieval belongs in a separate controller-side research-librarian service.

## Real OpenAI → Kaggle pilot

Generate one bounded candidate from the real KuaiRand contract:

```bash
python3 -m research_agent.real_pilot \
  --output runtime/openai-pilot/iteration-001
```

After explicit reviewer approval, package it into the trusted worker:

```bash
python3 -m research_agent.kaggle_packager \
  --candidate runtime/openai-pilot/iteration-001/candidate.py \
  --proposal runtime/openai-pilot/iteration-001/proposal.json \
  --destination runtime/kaggle-worker/kuairand_candidate_worker.py
```

The trusted worker creates label-free validation Parquet data, executes only safety-approved code, keeps labels inside the evaluator process, and exports metrics plus an audit report. Kaggle upload/polling is deliberately environment-specific; users supply their own token and private kernel metadata.

## API

Read-only:

- `GET /api/health`
- `GET /api/state`
- `GET /api/literature`
- `GET /api/actions`
- `GET /api/compute`

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
- Generated experiments receive only an explicit public workspace; evaluator labels remain outside it and are never serialized into the experiment directory.
- The KuaiRand research cutoff is 2022-04-28. Later rows are never scored during development.
- Training-row history features must be causal.
- Validation outcomes never enter feature construction.
- Same-row post-exposure feedback is not used as an inference feature.
- The organizer evaluator remains outside LLM-authored code.

The starter kit and KuaiRand dataset are not redistributed in this repository. Obtain them from the organizers and respect their licence and competition rules.

## Current status

The autonomous outer loop passes its controlled validation, including safety rejection, repair, external evaluation, memory, and champion promotion. The real archive and five-seed FM reproduction pass readiness checks. The official FM mean remains **0.601572** primary (GAUC **0.667400**, nDCG@5 **0.535744**).

Multiple generated challengers have run end to end on real KuaiRand; none has beaten the official FM, and all negative results remain in research memory. The architecture now begins with the reproduced FM as its anchor, penalizes repetition of severely negative lineages, repairs implementation failures without spending scientific iterations, and screens model quality on chronological training holdouts before external evaluation.

Not yet claimed: the agent has not beaten the FM; live/full-text literature RAG is not implemented; and the hosted static dashboard cannot directly reach a private local controller.
