'use client';

import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';

const API = process.env.NEXT_PUBLIC_KUAILAB_API ?? 'http://127.0.0.1:8787';
const stageLabels: Record<string, string> = {
  inspect: 'Inspect', hypothesize: 'Hypothesize', implement: 'Implement', train: 'Train', evaluate: 'Evaluate', reflect: 'Reflect',
};

type Metrics = { primary: number; gauc: number; ndcg5: number };
type Stage = { name: string; status: 'done' | 'active' | 'waiting' };
type ResourceUsage = {
  wall_seconds: number; train_seconds: number; cpu_seconds: number; cpu_hours: number; cpu_utilization_percent: number;
  peak_rss_mb: number; gpu_count: number; gpu_seconds: number; gpu_hours: number; peak_gpu_memory_mb: number; device: string;
};
type Iteration = {
  number: number; title: string; status: string; stage: string; metrics: Metrics | null; delta: number | null;
  gain?: number; duration_seconds: number; error?: string; evidence?: string; artifact?: string; accepted: boolean; resource_usage?: ResourceUsage | null;
};
type EventItem = { id: number; time: string; kind: string; title: string; detail: string; iteration?: number; stage?: string };
type RunLimits = { max_iterations: number; max_hours: number; convergence_epsilon: number; convergence_patience: number; bootstrap_verified: boolean };
type State = {
  campaign: { id: string | null; status: string; mode: string; provider: string; started_at: string | null; stop_reason: string | null; steering: string | null; continuations: number; session_start_iteration: number; session_start_wall_seconds: number; limits: RunLimits };
  config: { model: string; reasoning_effort: string; max_iterations: number; max_hours: number; convergence_epsilon: number; convergence_patience: number; api_key_available: boolean; dataset_available: boolean; adapter_available: boolean };
  current: null | { number: number; title: string; hypothesis: string; stage: string; status: string; activity: string; stages: Stage[]; acceptance: string; abort_condition: string; expected_gain: number | null; error?: string };
  metrics: { baseline: Metrics; champion: Metrics; delta: number };
  usage: { input_tokens: number; output_tokens: number; reasoning_tokens: number; total_tokens: number; wall_seconds: number; train_seconds: number; cpu_seconds: number; cpu_hours: number; gpu_hours: number; peak_rss_mb: number; peak_gpu_memory_mb: number; experiments_measured: number };
  iterations: Iteration[];
  events: EventItem[];
};

function score(value?: number | null) { return typeof value === 'number' ? value.toFixed(4) : '—'; }
function elapsed(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 3600)).padStart(2, '0')}:${String(Math.floor((whole % 3600) / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
}
function computeHours(value: number) { return value < 0.01 ? `${(value * 60).toFixed(1)}m` : `${value.toFixed(2)}h`; }
function memory(value?: number) { return value ? `${Math.round(value).toLocaleString()} MB` : '—'; }
function timeLabel(value: string) { return new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
function statusTone(status: string) {
  if (status === 'accepted' || status === 'complete') return 'good';
  if (status === 'failed') return 'warn';
  if (status === 'running') return 'active';
  return 'base';
}

export default function Home() {
  const [state, setState] = useState<State | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState('');
  const [showSetup, setShowSetup] = useState(false);
  const [showSteer, setShowSteer] = useState(false);
  const [setupMode, setSetupMode] = useState<'new' | 'continue'>('new');
  const [provider, setProvider] = useState<'demo' | 'gpt'>('demo');
  const [mode, setMode] = useState<'demo' | 'kuairand'>('demo');
  const [maxIterations, setMaxIterations] = useState(20);
  const [maxHours, setMaxHours] = useState(6);
  const [convergenceEpsilon, setConvergenceEpsilon] = useState(0.0001);
  const [convergencePatience, setConvergencePatience] = useState(8);
  const [bootstrapVerified, setBootstrapVerified] = useState(true);
  const [instruction, setInstruction] = useState('');

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/state`, { cache: 'no-store' });
      if (!response.ok) throw new Error('Backend returned an error');
      setState(await response.json());
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    const kickoff = window.setTimeout(refresh, 0);
    const timer = window.setInterval(refresh, 900);
    return () => { window.clearTimeout(kickoff); window.clearInterval(timer); };
  }, [refresh]);

  async function action(path: string, body: Record<string, unknown> = {}) {
    setError('');
    try {
      const response = await fetch(`${API}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error ?? 'Request failed');
      setState(payload.state);
      return true;
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Request failed');
      return false;
    }
  }

  async function startRun(event: FormEvent) {
    event.preventDefault();
    const limits = {
      max_iterations: maxIterations,
      max_hours: maxHours,
      convergence_epsilon: convergenceEpsilon,
      convergence_patience: convergencePatience,
    };
    const path = setupMode === 'continue' ? '/api/run/continue' : '/api/run/start';
    const body = setupMode === 'continue' ? { limits } : { provider, mode, limits, bootstrap_verified: bootstrapVerified };
    if (await action(path, body)) setShowSetup(false);
  }

  function openSetup(kind: 'new' | 'continue') {
    setSetupMode(kind);
    setMaxIterations(kind === 'continue' ? 20 : (state?.config.max_iterations ?? 50));
    setMaxHours(state?.config.max_hours ?? 6);
    setConvergenceEpsilon(state?.config.convergence_epsilon ?? 0.0001);
    setConvergencePatience(state?.config.convergence_patience ?? 8);
    if (kind === 'new') {
      setProvider(state?.config.api_key_available ? 'gpt' : 'demo');
      setMode(state?.config.dataset_available && state?.config.adapter_available ? 'kuairand' : 'demo');
    }
    setShowSetup(true);
  }

  async function steer(event: FormEvent) {
    event.preventDefault();
    if (await action('/api/steer', { instruction })) { setInstruction(''); setShowSteer(false); }
  }

  const status = state?.campaign.status ?? 'offline';
  const active = ['running', 'paused', 'stopping'].includes(status);
  const synthetic = state?.campaign.mode === 'demo';
  const canContinue = Boolean(!active && state?.campaign.id && status !== 'idle');
  const champion = state?.metrics.champion;
  const current = state?.current;
  const completedCount = Math.max(0, (state?.iterations.length ?? 1) - 1);
  const runLimits = state?.campaign.limits ?? { max_iterations: state?.config.max_iterations ?? 50, max_hours: state?.config.max_hours ?? 6, convergence_epsilon: state?.config.convergence_epsilon ?? 0.0001, convergence_patience: state?.config.convergence_patience ?? 8, bootstrap_verified: true };
  const sessionStart = state?.campaign.session_start_iteration ?? 1;
  const sessionCompleted = (state?.iterations ?? []).filter((item) => item.number >= sessionStart).length;
  const sessionWall = Math.max(0, (state?.usage.wall_seconds ?? 0) - (state?.campaign.session_start_wall_seconds ?? 0));
  const remainingSeconds = Math.max(0, runLimits.max_hours * 3600 - sessionWall);
  const events = useMemo(() => [...(state?.events ?? [])].reverse().slice(0, 7), [state]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">KL</div>
        <nav aria-label="Dashboard sections">
          <a className="nav-item active" href="#overview" aria-label="Run overview">⌁</a>
          <a className="nav-item" href="#iterations" aria-label="Iterations">◇</a>
          <a className="nav-item" href="#trace" aria-label="Evidence trace">≡</a>
        </nav>
        <div className={`sidebar-status ${connected ? '' : 'offline'}`} title={connected ? 'Local engine connected' : 'Local engine offline'}><span />{connected ? 'API' : 'OFF'}</div>
      </aside>

      <section className="workspace" id="overview">
        <header className="topbar">
          <div>
            <p className="eyebrow">KuaiRand-Pure · autonomous campaign</p>
            <h1>Research control room</h1>
          </div>
          <div className="top-actions">
            <span className={`live-pill ${status}`}><i /> {connected ? status : 'backend offline'}</span>
            {status === 'running' && <button className="button ghost" onClick={() => action('/api/run/pause')}>Pause</button>}
            {status === 'paused' && <button className="button ghost" onClick={() => action('/api/run/resume')}>Resume</button>}
            {active && <button className="button danger" onClick={() => action('/api/run/stop')}>Stop</button>}
            {active && <button className="button primary" onClick={() => setShowSteer(true)}>Steer agent</button>}
            {canContinue && <button className="button ghost" onClick={() => openSetup('continue')}>Continue champion</button>}
            {!active && <button className="button primary" onClick={() => openSetup('new')}>New campaign</button>}
          </div>
        </header>

        {!connected && <div className="banner warn-banner"><b>Backend is offline.</b> Start the local engine to enable real controls and live iteration updates.</div>}
        {error && <div className="banner error-banner" role="alert"><b>Couldn’t complete that action.</b> {error}<button onClick={() => setError('')} aria-label="Dismiss">×</button></div>}
        {connected && synthetic && <div className="banner warn-banner"><b>Synthetic smoke-test evidence.</b> These scores—including the 0.6250 demo ceiling—are simulated to test the workflow, not trained KuaiRand validation results.</div>}

        <div className="metrics-grid">
          <article className="metric-card featured"><span>{synthetic ? 'Demo primary · simulated' : 'Champion primary · verified'}</span><strong>{score(champion?.primary)}</strong><small>{state ? `${state.metrics.delta >= 0 ? '+' : ''}${state.metrics.delta.toFixed(4)} over ${synthetic ? 'demo' : 'reproduced'} baseline` : 'Waiting for local engine'}</small></article>
          <article className="metric-card"><span>GAUC</span><strong>{score(champion?.gauc)}</strong><small>{synthetic ? 'simulated smoke test' : 'validation-best'}</small></article>
          <article className="metric-card"><span>nDCG@5</span><strong>{score(champion?.ndcg5)}</strong><small>{synthetic ? 'simulated smoke test' : 'validation-best'}</small></article>
          <article className="metric-card"><span>Compute used</span><strong>{computeHours(state?.usage.cpu_hours ?? 0)} CPU</strong><small>{computeHours(state?.usage.gpu_hours ?? 0)} GPU · {memory(state?.usage.peak_rss_mb)} peak RAM</small></article>
          <article className="metric-card"><span>Session budget</span><strong>{String(sessionCompleted).padStart(2, '0')} / {runLimits.max_iterations}</strong><small>{elapsed(sessionWall)} this session · {elapsed(remainingSeconds)} left · {completedCount} total</small></article>
        </div>

        <div className="content-grid">
          <section className="panel active-run">
            <div className="panel-heading">
              <div><p className="eyebrow">{current ? `Iteration ${String(current.number).padStart(3, '0')}` : 'Campaign state'}</p><h2>{current?.title ?? (status === 'idle' ? 'Ready for the first experiment' : 'No active iteration')}</h2></div>
              <span className="agent-badge">{state?.campaign.provider === 'gpt' ? state.config.model : 'Demo planner'} · {synthetic ? 'synthetic benchmark' : state?.config.reasoning_effort ?? 'high'}</span>
            </div>
            <p className="hypothesis">{current?.hypothesis ?? 'Choose a demo campaign to verify the full loop, or connect GPT-5.6 Sol and the organizer runner for real experiments.'}</p>
            <div className="stage-track" aria-label="Iteration stages">
              {(current?.stages ?? Object.keys(stageLabels).map((name) => ({ name, status: 'waiting' as const }))).map((stage, index) => (
                <div className={`stage ${stage.status}`} key={stage.name}>
                  <span>{stage.status === 'done' ? '✓' : index + 1}</span><b>{stageLabels[stage.name]}</b>
                </div>
              ))}
            </div>
            <div className={`activity-card ${current?.status === 'failed' ? 'failed' : ''}`}>
              <div className="activity-icon">{current?.status === 'failed' ? '!' : '⌘'}</div>
              <div><strong>{current ? `${stageLabels[current.stage] ?? current.stage} ${current.status === 'failed' ? 'failed' : 'stage'}` : 'Engine standing by'}</strong><p>{current?.error ?? current?.activity ?? 'Stage transitions and evidence will appear here in real time.'}</p></div>
              {status === 'running' && <span className="pulse-dots">•••</span>}
            </div>
            <div className="run-meta">
              <div><span>Acceptance</span><b>{current?.acceptance ?? `Any positive gain · stop sensitivity ${runLimits.convergence_epsilon.toFixed(6)}`}</b></div>
              <div><span>Abort condition</span><b>{current?.abort_condition ?? 'Invalid metrics, timeout, or runner failure'}</b></div>
              <div><span>Resources</span><b>{(state?.usage.total_tokens ?? 0).toLocaleString()} tokens · {computeHours(state?.usage.cpu_hours ?? 0)} CPU · {computeHours(state?.usage.gpu_hours ?? 0)} GPU</b></div>
            </div>
          </section>

          <aside className="panel event-panel" id="trace">
            <div className="panel-heading"><div><p className="eyebrow">Append-only evidence</p><h2>Live trace</h2></div><span className="evidence-dot" title="Persisted locally" /></div>
            <ol className="timeline">
              {events.map((item, index) => <li className={index === 0 ? 'current' : ''} key={item.id}><time>{timeLabel(item.time)}</time><div><b>{item.title}</b><p>{item.detail}</p></div></li>)}
            </ol>
          </aside>
        </div>

        <section className="panel table-panel" id="iterations">
          <div className="panel-heading"><div><p className="eyebrow">Campaign history</p><h2>Iterations</h2></div><span className="subtle">{synthetic ? 'Synthetic workflow evidence · no model training' : 'Validation-best checkpoint retained · hidden test untouched'}</span></div>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Iteration</th><th>Experiment</th><th>Status</th><th>Primary</th><th>Δ baseline</th><th>Train</th><th>Compute</th><th>Peak RAM</th></tr></thead>
              <tbody>{[...(state?.iterations ?? [])].reverse().map((item) => <tr key={item.number}><td className="mono">#{String(item.number).padStart(3, '0')}</td><td><b>{item.title}</b>{item.error && <small className="row-note">{item.error}</small>}</td><td><span className={`status ${statusTone(item.status)}`}>{item.status}</span></td><td className="mono">{score(item.metrics?.primary)}</td><td className={`mono ${(item.delta ?? 0) > 0 ? 'positive' : ''}`}>{item.delta == null ? '—' : `${item.delta >= 0 ? '+' : ''}${item.delta.toFixed(4)}`}</td><td className="mono">{item.resource_usage ? elapsed(item.resource_usage.train_seconds) : elapsed(item.duration_seconds)}</td><td className="mono resource-cell">{item.resource_usage ? `${computeHours(item.resource_usage.cpu_hours)} CPU · ${computeHours(item.resource_usage.gpu_hours)} GPU` : '—'}</td><td className="mono">{memory(item.resource_usage?.peak_rss_mb)}</td></tr>)}</tbody>
            </table>
          </div>
        </section>

        <section className="readiness-strip" aria-label="Connection readiness">
          <div><span className={state?.config.api_key_available ? 'ready' : 'missing'} />GPT key <b>{state?.config.api_key_available ? 'ready' : 'not set'}</b></div>
          <div><span className={state?.config.dataset_available ? 'ready' : 'missing'} />KuaiRand data <b>{state?.config.dataset_available ? 'ready' : 'not connected'}</b></div>
          <div><span className={state?.config.adapter_available ? 'ready' : 'missing'} />Runner adapter <b>{state?.config.adapter_available ? 'ready' : 'not connected'}</b></div>
          <div className="resource-total">Train {elapsed(state?.usage.train_seconds ?? 0)} · CPU {computeHours(state?.usage.cpu_hours ?? 0)} · GPU {computeHours(state?.usage.gpu_hours ?? 0)} · Tokens {(state?.usage.total_tokens ?? 0).toLocaleString()}</div>
        </section>
      </section>

      {showSetup && <div className="modal-backdrop" onMouseDown={() => setShowSetup(false)}><form className="modal campaign-modal" onSubmit={startRun} onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-heading"><div><p className="eyebrow">{setupMode === 'continue' ? 'Retained research' : 'New campaign'}</p><h2>{setupMode === 'continue' ? 'Continue from the champion' : 'Configure autonomous research'}</h2></div><button type="button" className="close-button" onClick={() => setShowSetup(false)}>×</button></div>
        {setupMode === 'new' ? <>
          <label>Researcher<select value={provider} onChange={(event) => setProvider(event.target.value as 'demo' | 'gpt')}><option value="demo">Demo planner — no API cost</option><option value="gpt" disabled={!state?.config.api_key_available}>GPT-5.6 Sol — high reasoning{!state?.config.api_key_available ? ' (key not set)' : ''}</option></select></label>
          <label>Benchmark<select value={mode} onChange={(event) => setMode(event.target.value as 'demo' | 'kuairand')}><option value="demo">Synthetic smoke test</option><option value="kuairand" disabled={!state?.config.dataset_available || !state?.config.adapter_available}>KuaiRand-Pure validation{!state?.config.dataset_available || !state?.config.adapter_available ? ' (setup required)' : ''}</option></select></label>
          {mode === 'kuairand' && <label className="check-label"><input type="checkbox" checked={bootstrapVerified} onChange={(event) => setBootstrapVerified(event.target.checked)} /><span><b>Start from verified 0.605885 champion</b><small>Imports the clean temporal FM + BPR + DeepFM recipe and searches above it.</small></span></label>}
        </> : <div className="resume-summary"><span>Retained primary</span><strong>{score(champion?.primary)}</strong><small>{state?.campaign.provider === 'gpt' ? state.config.model : 'Demo planner'} · {state?.campaign.mode === 'kuairand' ? 'KuaiRand-Pure validation' : 'Synthetic smoke test'} · {completedCount} recorded experiments</small></div>}
        <div className="form-grid">
          <label>Experiments this session<input type="number" min="1" max="100" step="1" value={maxIterations} onChange={(event) => setMaxIterations(Number(event.target.value))} required /></label>
          <label>Time budget (hours)<input type="number" min="0.1" max="24" step="0.1" value={maxHours} onChange={(event) => setMaxHours(Number(event.target.value))} required /></label>
          <label>Small-gain threshold<input type="number" min="0" max="0.01" step="0.00001" value={convergenceEpsilon} onChange={(event) => setConvergenceEpsilon(Number(event.target.value))} required /></label>
          <label>Stop after small gains<input type="number" min="0" max="50" step="1" value={convergencePatience} onChange={(event) => setConvergencePatience(Number(event.target.value))} required /></label>
        </div>
        <p className="modal-note">Every positive validation gain is retained. Set “stop after small gains” to 0 to use the full experiment/time budget. Real autonomous research needs the dataset, runner, and your API key available to the local engine.</p>
        <button className="button primary wide" type="submit">{setupMode === 'continue' ? 'Continue autonomous research' : 'Start autonomous campaign'}</button>
      </form></div>}

      {showSteer && <div className="modal-backdrop" onMouseDown={() => setShowSteer(false)}><form className="modal" onSubmit={steer} onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-heading"><div><p className="eyebrow">Operator intervention</p><h2>Guide the next iteration</h2></div><button type="button" className="close-button" onClick={() => setShowSteer(false)}>×</button></div>
        <label>Instruction<textarea value={instruction} onChange={(event) => setInstruction(event.target.value)} placeholder="Example: Focus next on exposure bias; keep runtime under 10 minutes." required maxLength={1000} /></label>
        <p className="modal-note">This is logged as an intervention and applied to the next hypothesis.</p>
        <button className="button primary wide" type="submit">Queue guidance</button>
      </form></div>}
    </main>
  );
}
