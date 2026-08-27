'use client';

import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';

type Metrics = { GAUC: number; 'nDCG@5': number; primary: number };
type Experiment = {
  experiment_id: string; action_id?: string; id?: string; title: string; family: string; status: string;
  hypothesis?: string; reason?: string; metrics?: Metrics; delta_vs_champion?: number | null; improved?: boolean;
  estimated_minutes?: number; elapsed_seconds?: number; error?: string; evidence?: LiteratureCard[];
};
type LiteratureCard = { id: string; title: string; year: number; url: string; tags: string[]; claim: string; cautions: string[] };
type EventItem = { id: string; timestamp: string; kind: string; message: string; metrics?: Metrics; improved?: boolean };
type AgentState = {
  run: { id: string; status: string; benchmark: string; label: string; budget_seconds: number; elapsed_seconds: number; max_experiments: number; manual_interventions: number; executor_mode: string; llm_mode: string };
  baseline: { metrics: Metrics; status: string };
  best: { experiment_id: string; title: string; metrics: Metrics };
  current_experiment: Experiment | null;
  candidate_queue: Experiment[];
  experiments: Experiment[];
  events: EventItem[];
  literature_hits: LiteratureCard[];
  steering: { id: string; timestamp: string; message: string; status: string }[];
  updated_at?: string;
};

const baseline: Metrics = { GAUC: 0.667400, 'nDCG@5': 0.535744, primary: 0.601572 };
const demoState: AgentState = {
  run: { id: 'preview', status: 'ready', benchmark: 'KuaiRand-Pure', label: 'long_view', budget_seconds: 10800, elapsed_seconds: 0, max_experiments: 10, manual_interventions: 0, executor_mode: 'simulation', llm_mode: 'fallback' },
  baseline: { metrics: baseline, status: 'passed' },
  best: { experiment_id: 'iteration-000', title: 'Official FM baseline', metrics: baseline },
  current_experiment: null,
  candidate_queue: [], experiments: [], steering: [], literature_hits: [],
  events: [{ id: 'baseline', timestamp: new Date().toISOString(), kind: 'baseline', message: 'Official FM baseline reproduced across five seeds.' }],
};

const nav = [
  { id: 'overview', symbol: '⌁', label: 'Overview' },
  { id: 'experiments', symbol: '◫', label: 'Experiments' },
  { id: 'literature', symbol: '≡', label: 'Literature' },
  { id: 'controls', symbol: '⚙', label: 'Controls' },
];

function formatScore(value?: number) { return typeof value === 'number' ? value.toFixed(4) : '—'; }
function formatDuration(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safe / 3600); const minutes = Math.floor((safe % 3600) / 60); const secs = safe % 60;
  return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${secs}s` : `${secs}s`;
}
function humanStatus(status: string) { return status.replaceAll('_', ' ').replace(/^./, char => char.toUpperCase()); }

function Trajectory({ state }: { state: AgentState }) {
  const points = useMemo(() => [
    { id: 'iteration-000', value: state.baseline.metrics.primary, improved: true },
    ...state.experiments.filter(item => item.metrics).map(item => ({ id: item.experiment_id, value: item.metrics!.primary, improved: !!item.improved })),
  ], [state]);
  const values = points.map(point => point.value);
  const low = Math.min(...values, state.baseline.metrics.primary - 0.008);
  const high = Math.max(...values, state.baseline.metrics.primary + 0.018);
  const positions = points.map((point, index) => ({
    ...point,
    x: points.length === 1 ? 7 : 7 + (index / Math.max(1, points.length - 1)) * 82,
    y: 82 - ((point.value - low) / Math.max(.0001, high - low)) * 62,
  }));
  const baselineY = 82 - ((state.baseline.metrics.primary - low) / Math.max(.0001, high - low)) * 62;
  return (
    <div className="chart" aria-label="Primary score trajectory">
      <div className="target-line" style={{ top: `${baselineY}%` }}><span>baseline {formatScore(state.baseline.metrics.primary)}</span></div>
      {positions.slice(1).map((point, index) => {
        const previous = positions[index]; const dx = point.x - previous.x; const dy = point.y - previous.y;
        const length = Math.sqrt(dx * dx + dy * dy); const angle = Math.atan2(dy, dx) * 180 / Math.PI;
        return <div key={`line-${point.id}`} className="dynamic-line" style={{ left: `${previous.x}%`, top: `${previous.y}%`, width: `${length}%`, transform: `rotate(${angle}deg)` }} />;
      })}
      {positions.map((point, index) => <div key={point.id} className={`dynamic-point ${point.improved ? 'winner' : ''}`} style={{ left: `${point.x}%`, top: `${point.y}%` }}><span>{index === positions.length - 1 ? formatScore(point.value) : `#${index}`}</span></div>)}
      {state.current_experiment && <div className="chart-running"><b /> evaluating #{String(state.experiments.length + 1).padStart(3, '0')}</div>}
      <div className="axis-labels"><span>Baseline</span><span>Research iterations</span><span>Convergence</span></div>
    </div>
  );
}

function MetricCard({ label, value, note, hero = false }: { label: string; value: string; note: string; hero?: boolean }) {
  return <article className={`metric-card ${hero ? 'hero-metric' : ''}`}><p>{label}</p><strong>{value}</strong><span>{note}</span></article>;
}

export default function Home() {
  const [tab, setTab] = useState('overview');
  const [state, setState] = useState<AgentState>(demoState);
  const [connected, setConnected] = useState(false);
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:8765');
  const [draftUrl, setDraftUrl] = useState('http://127.0.0.1:8765');
  const [message, setMessage] = useState('');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    const saved = window.localStorage.getItem('kuairand-agent-api');
    if (saved) { setApiUrl(saved); setDraftUrl(saved); }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/state`, { cache: 'no-store' });
      if (!response.ok) throw new Error('Agent API unavailable');
      setState(await response.json()); setConnected(true);
    } catch { setConnected(false); }
  }, [apiUrl]);

  useEffect(() => { refresh(); const timer = window.setInterval(refresh, 1400); return () => window.clearInterval(timer); }, [refresh]);

  async function post(path: string, payload?: object) {
    try {
      const response = await fetch(`${apiUrl}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload ?? {}) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? 'Request failed');
      setState(body); setConnected(true); setNotice('Command accepted'); window.setTimeout(() => setNotice(''), 2200);
    } catch (error) { setNotice(error instanceof Error ? error.message : 'Could not reach agent'); }
  }

  async function steer(event: FormEvent) {
    event.preventDefault(); if (!message.trim()) return;
    await post('/api/steer', { message }); setMessage('');
  }

  function saveApi(event: FormEvent) {
    event.preventDefault(); const clean = draftUrl.trim().replace(/\/$/, '');
    window.localStorage.setItem('kuairand-agent-api', clean); setApiUrl(clean); setNotice('Connection address saved');
  }

  const remaining = Math.max(0, state.run.budget_seconds - state.run.elapsed_seconds);
  const latestDecision = [...state.events].reverse().find(event => event.kind === 'decision');
  const active = state.current_experiment;
  const displayedExperiments: Experiment[] = [
    { experiment_id: 'iteration-000', title: 'Official FM baseline', family: 'factorization_machine', status: 'completed', metrics: state.baseline.metrics, improved: true },
    ...state.experiments,
    ...(active ? [active] : []),
  ];

  return (
    <main className="shell">
      <aside className="rail">
        <button className="brand-mark" onClick={() => setTab('overview')} aria-label="KuaiRand Research Cockpit home">KR</button>
        <nav aria-label="Primary navigation">{nav.map(item => <button key={item.id} className={`rail-button ${tab === item.id ? 'active' : ''}`} onClick={() => setTab(item.id)} aria-label={item.label} title={item.label}>{item.symbol}</button>)}</nav>
        <div className="rail-spacer" /><div className={`status-dot ${connected ? 'online' : ''}`} title={connected ? 'Agent connected' : 'Preview mode'} />
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">{state.run.benchmark} · autonomous research</p><h1>{nav.find(item => item.id === tab)?.label ?? 'Research cockpit'}</h1></div>
          <div className="header-actions">
            <span className={`connection ${connected ? 'connected' : ''}`}>{connected ? 'Live agent' : 'Preview data'}</span>
            <span className={`run-state status-${state.run.status}`}><i /> {humanStatus(state.run.status)}</span>
          </div>
        </header>

        {tab === 'overview' && <>
          <div className="metric-grid">
            <MetricCard label="Best primary" value={formatScore(state.best.metrics.primary)} note={state.best.title} hero />
            <MetricCard label="GAUC" value={formatScore(state.best.metrics.GAUC)} note={`${(state.best.metrics.GAUC - state.baseline.metrics.GAUC >= 0 ? '+' : '')}${(state.best.metrics.GAUC - state.baseline.metrics.GAUC).toFixed(4)} vs baseline`} />
            <MetricCard label="nDCG@5" value={formatScore(state.best.metrics['nDCG@5'])} note={`${(state.best.metrics['nDCG@5'] - state.baseline.metrics['nDCG@5'] >= 0 ? '+' : '')}${(state.best.metrics['nDCG@5'] - state.baseline.metrics['nDCG@5']).toFixed(4)} vs baseline`} />
            <MetricCard label="Budget left" value={formatDuration(remaining)} note={`${Math.max(0, state.run.max_experiments - state.experiments.length)} experiment slots remain`} />
          </div>

          <div className="main-grid">
            <section className="panel chart-panel">
              <div className="panel-heading"><div><p className="eyebrow">Progress</p><h2>Validation trajectory</h2></div><span className="legend"><i /> Primary score</span></div>
              <Trajectory state={state} />
            </section>
            <aside className="panel agent-panel">
              <div className="panel-heading"><div><p className="eyebrow">Agent rationale</p><h2>{active ? 'Current experiment' : 'Next research move'}</h2></div><span className="brain">A</span></div>
              <div className="thought"><p className="thought-label">{active ? 'CURRENT HYPOTHESIS' : 'LATEST DECISION'}</p><p>{active?.hypothesis ?? latestDecision?.message ?? 'Ready to inspect the baseline and select the first evidence-backed experiment.'}</p></div>
              <div className="evidence"><span>Why this</span><p>{active?.reason ?? active?.evidence?.[0]?.claim ?? 'Selection balances expected gain, literature support, novelty, risk, and remaining compute.'}</p></div>
              <button className="secondary-button" onClick={() => setTab('experiments')}>Inspect decision trace →</button>
            </aside>
          </div>

          <section className="lower-grid">
            <div className="panel experiments-panel">
              <div className="panel-heading"><div><p className="eyebrow">Run history</p><h2>Experiments</h2></div><button className="quiet-button" onClick={() => setTab('experiments')}>View all</button></div>
              <div className="experiment-list">{displayedExperiments.slice(-4).map(experiment => <ExperimentRow key={experiment.experiment_id} experiment={experiment} />)}</div>
            </div>
            <SteeringPanel message={message} setMessage={setMessage} onSubmit={steer} disabled={!connected} />
          </section>
        </>}

        {tab === 'experiments' && <section className="content-grid">
          <div className="panel wide-panel">
            <div className="panel-heading"><div><p className="eyebrow">Evidence ledger</p><h2>Every attempt, including failures</h2></div><span className="small-stat">{displayedExperiments.length} recorded</span></div>
            <div className="experiment-table">
              <div className="table-head"><span>ID</span><span>Experiment</span><span>Family</span><span>Primary</span><span>Δ champion</span><span>Status</span></div>
              {displayedExperiments.map(item => <div className="table-row" key={item.experiment_id}><span className="mono">{item.experiment_id.replace('iteration-', '#')}</span><span><strong>{item.title}</strong><small>{item.error ?? item.hypothesis ?? 'Organizer-provided reference pipeline'}</small></span><span>{item.family.replaceAll('_', ' ')}</span><span className="mono">{formatScore(item.metrics?.primary)}</span><span className={`mono ${item.delta_vs_champion && item.delta_vs_champion > 0 ? 'positive' : ''}`}>{item.delta_vs_champion == null ? '—' : `${item.delta_vs_champion > 0 ? '+' : ''}${item.delta_vs_champion.toFixed(4)}`}</span><span><StatusPill status={item.status} /></span></div>)}
            </div>
          </div>
          <aside className="panel event-panel"><p className="eyebrow">Live audit trail</p><h2>Agent events</h2><div className="event-list">{[...state.events].reverse().slice(0, 18).map(event => <div className="event-item" key={event.id}><span className={`event-icon event-${event.kind}`} /> <div><strong>{event.message}</strong><small>{new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })} · {event.kind}</small></div></div>)}</div></aside>
        </section>}

        {tab === 'literature' && <section className="literature-layout">
          <div className="literature-intro"><p className="eyebrow">Research memory</p><h2>Evidence the agent can retrieve</h2><p>These cards are priors, not a prescribed model order. The agent retrieves them according to the observed failure mode, then records exactly what influenced each experiment.</p></div>
          <div className="literature-grid">{(state.literature_hits.length ? state.literature_hits : demoLiterature).map(card => <a className="literature-card" key={card.id} href={card.url} target="_blank" rel="noreferrer"><div><span>{card.year}</span><span>{card.tags.slice(0, 2).join(' · ')}</span></div><h3>{card.title}</h3><p>{card.claim}</p>{card.cautions?.[0] && <small>Watch: {card.cautions[0]}</small>}</a>)}</div>
        </section>}

        {tab === 'controls' && <section className="control-layout">
          <div className="panel control-card"><p className="eyebrow">Run controls</p><h2>Autonomy with explicit human authority</h2><p>Start, pause, resume, or stop the loop. Every action is added to the audit trail.</p><div className="button-cluster"><button className="primary-button" onClick={() => post('/api/run/start')} disabled={!connected || state.run.status === 'running'}>Start run</button><button onClick={() => post(state.run.status === 'paused' ? '/api/run/resume' : '/api/run/pause')} disabled={!connected || !['running', 'paused'].includes(state.run.status)}>{state.run.status === 'paused' ? 'Resume' : 'Pause'}</button><button className="danger-button" onClick={() => post('/api/run/stop')} disabled={!connected}>Stop</button></div><dl><div><dt>Executor</dt><dd>{state.run.executor_mode}</dd></div><div><dt>LLM planner</dt><dd>{state.run.llm_mode}</dd></div><div><dt>Manual interventions</dt><dd>{state.run.manual_interventions}</dd></div><div><dt>Label contract</dt><dd>{state.run.label}</dd></div></dl></div>
          <SteeringPanel message={message} setMessage={setMessage} onSubmit={steer} disabled={!connected} expanded />
          <form className="panel connection-card" onSubmit={saveApi}><p className="eyebrow">Connection</p><h2>Local agent address</h2><p>The dashboard remains useful as a read-only preview when the controller is offline.</p><label htmlFor="api-url">Agent API URL</label><div><input id="api-url" value={draftUrl} onChange={event => setDraftUrl(event.target.value)} /><button>Connect</button></div><span className={connected ? 'positive' : ''}>{connected ? `Connected · updated ${state.updated_at ? new Date(state.updated_at).toLocaleTimeString() : 'now'}` : 'Offline · showing preview state'}</span></form>
        </section>}

        {notice && <div className="toast" role="status">{notice}</div>}
      </section>
    </main>
  );
}

function ExperimentRow({ experiment }: { experiment: Experiment }) {
  const score = experiment.metrics?.primary;
  return <div className="experiment-row"><span className={`experiment-state ${experiment.status}`} /><span className="experiment-id">{experiment.experiment_id.replace('iteration-', '#')}</span><strong>{experiment.title}</strong><span className="experiment-score">{score == null ? humanStatus(experiment.status) : formatScore(score)}</span><StatusPill status={experiment.status} /></div>;
}

function StatusPill({ status }: { status: string }) { return <span className={`experiment-delta status-${status}`}>{humanStatus(status)}</span>; }

function SteeringPanel({ message, setMessage, onSubmit, disabled, expanded = false }: { message: string; setMessage: (value: string) => void; onSubmit: (event: FormEvent) => void; disabled: boolean; expanded?: boolean }) {
  return <div className={`panel steering-panel ${expanded ? 'expanded' : ''}`}><div><p className="eyebrow">Human steering</p><h2>Guide without taking over</h2></div><p className="steering-copy">Add a constraint, point to evidence, or change priorities. The intervention is recorded and considered at the next decision boundary.</p><form className="steer-form" onSubmit={onSubmit}><label htmlFor={expanded ? 'steer-expanded' : 'steer'}>Message the research agent</label><textarea id={expanded ? 'steer-expanded' : 'steer'} value={message} onChange={event => setMessage(event.target.value)} placeholder="e.g. Prioritize experiments under 15 minutes; do not use validation labels in features." disabled={disabled} rows={expanded ? 7 : 2} /><div><span>{message.length}/1000</span><button type="submit" disabled={disabled || !message.trim()}>Send guidance</button></div></form></div>;
}

const demoLiterature: LiteratureCard[] = [
  { id: 'aide', title: 'AIDE: AI-Driven Exploration in the Space of Code', year: 2025, url: 'https://arxiv.org/abs/2502.13138', tags: ['agent', 'tree search'], claim: 'Organizes ML engineering as search over code solutions with improvement and debugging branches.', cautions: ['Keep evaluation outside model-authored code.'] },
  { id: 'bpr', title: 'BPR: Bayesian Personalized Ranking', year: 2009, url: 'https://arxiv.org/abs/1205.2618', tags: ['pairwise', 'ranking'], claim: 'Directly optimizes preference for observed positives over sampled alternatives.', cautions: ['Negative sampling changes the objective.'] },
  { id: 'din', title: 'Deep Interest Network', year: 2018, url: 'https://arxiv.org/abs/1706.06978', tags: ['sequence', 'attention'], claim: 'Uses candidate-aware attention to activate relevant behavior history.', cautions: ['History must be causal.'] },
  { id: 'mmoe', title: 'Multi-gate Mixture-of-Experts', year: 2018, url: 'https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/', tags: ['multi-task', 'experts'], claim: 'Task gates can share useful structure while limiting negative transfer.', cautions: ['Auxiliary labels require ablation.'] },
];
