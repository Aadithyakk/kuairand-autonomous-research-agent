'use client';

import { useEffect, useReducer, useState, type CSSProperties } from 'react';
import evidence from '../public/research-replay.json';
import { clock, initialReplayState, replayDuration, replayReducer, replaySteps, signed, stepAt, stepStart } from './replay-model';

const wave = evidence.wave;
const rcr = wave.experiments.find(item => item.id === 'rcr')!;
const selectedRun = evidence.screen.runs.find(item => item.alpha === evidence.screen.selected_alpha)!;
const controlRun = evidence.screen.runs.find(item => item.alpha === 0)!;
const metricKeys = ['primary', 'GAUC', 'nDCG@5'] as const;
type Metrics = { primary: number; GAUC: number; 'nDCG@5': number };
type SourceId = typeof evidence.sources[number]['id'];

function MetricComparison({ base, candidate, delta, baselineLabel }: { base: Metrics; candidate: Metrics; delta: Metrics; baselineLabel: string }) {
  return <div className="replay-comparison">
    <div className="replay-comparison-labels"><span>{baselineLabel}</span><span>Candidate</span></div>
    {metricKeys.map(key => <div className="replay-metric-pair" key={key}>
      <div><b>{key === 'primary' ? 'Primary' : key}</b><strong className={delta[key] >= 0 ? 'positive' : 'replay-negative'}>{signed(delta[key])}</strong></div>
      <div className="replay-metric-bars"><span style={{ width: `${base[key] * 100}%` }} /><span style={{ width: `${candidate[key] * 100}%` }} /></div>
      <div className="replay-metric-values"><span>{base[key].toFixed(9)}</span><span>{candidate[key].toFixed(9)}</span></div>
    </div>)}
    <small>Bars share a 0–1 scale; exact deltas show changes too small to see.</small>
  </div>;
}

function TrainingChart() {
  const points = (rows: typeof selectedRun.training_history) => rows.map(row => `${42 + (row.epoch - 1) * 62},${20 + (0.65 - row.BCE) / 0.2 * 150}`).join(' ');
  return <figure className="replay-training-chart">
    <svg viewBox="0 0 380 205" role="img" aria-labelledby="training-title training-desc">
      <title id="training-title">Recorded screen training BCE over six epochs</title>
      <desc id="training-desc">Control and selected RCR candidate training BCE decline from about 0.623 to 0.480. These are training losses, not validation ranking scores.</desc>
      {[0.65, 0.55, 0.45].map(value => <g key={value}><line x1="42" x2="360" y1={20 + (0.65 - value) / 0.2 * 150} y2={20 + (0.65 - value) / 0.2 * 150} stroke="#dce4df" /><text x="5" y={24 + (0.65 - value) / 0.2 * 150}>{value.toFixed(2)}</text></g>)}
      <polyline points={points(controlRun.training_history)} fill="none" stroke="#879a90" strokeWidth="4" strokeDasharray="5 4" />
      <polyline points={points(selectedRun.training_history)} fill="none" stroke="#1f6b48" strokeWidth="2" />
      {selectedRun.training_history.map(row => <g key={row.epoch}><circle cx={42 + (row.epoch - 1) * 62} cy={20 + (0.65 - row.BCE) / 0.2 * 150} r="3" fill="#1f6b48" /><text x={39 + (row.epoch - 1) * 62} y="197">{row.epoch}</text></g>)}
    </svg>
    <figcaption><span>— RCR</span><span>┄ Control</span><span>Epoch · training BCE</span></figcaption>
  </figure>;
}

export function StageCanvas({ id, inspect, showTree }: { id: string; inspect: (id: SourceId) => void; showTree: () => void }) {
  switch (id) {
    case 'inspect': return <>
      <div className="replay-boundary"><b>KuaiRand-Pure · target: long_view</b><div><span>Model selection<br /><strong>08–14 Apr</strong></span><span>Locked screen<br /><strong>15–21 Apr</strong></span><span>Confirmation<br /><strong>22–28 Apr</strong></span></div><small>2022 outcomes · hidden test from 29 Apr excluded in the recorded protocol.</small></div>
      <div className="replay-facts"><div><strong>{evidence.champion.champion.rows.toLocaleString()}</strong><span>Confirmation impressions</span></div><div><strong>{evidence.champion.champion.users.toLocaleString()}</strong><span>Confirmation users</span></div></div>
      <p className="replay-note">Model input fields recorded in the config</p><div className="replay-chips">{evidence.screen.matched_configuration.fields.map(field => <code key={field}>{field}</code>)}</div>
      <p className="replay-unavailable">Missingness, target balance, and correlation diagnostics were not saved in this evidence bundle. No profile charts are invented.</p>
      <button className="replay-text-button" onClick={() => inspect('wave')}>Inspect the temporal protocol →</button>
    </>;
    case 'research': return <>
      <article className="replay-paper"><div className="replay-paper-meta"><span>PRIMARY SOURCE · RECORDED CITATION</span><span>2022 / revised 2023</span></div><h4>Regression Compatible Listwise Objectives for Calibrated Ranking with Binary Relevance</h4><p>Bai et al. · arXiv:2211.01494</p><p>The paper studies ranking objectives compatible with regression. Here, the recorded local adaptation mixes BCE with a sigmoid-based listwise term.</p><a href={evidence.screen.paper} target="_blank" rel="noopener noreferrer">Read the paper ↗</a></article>
      <div className="replay-callout"><b>Why it belongs in this experiment</b><p>It addresses the ranking–calibration trade-off while allowing a small, matched loss change. Published results do not establish a gain on this split.</p></div>
      <p className="replay-unavailable">Search queries and browsing timestamps were not recorded. This is a cited source, not a simulated live search.</p>
    </>;
    case 'hypothesize': return <>
      <div className="replay-callout"><b>Hypothesis · editorial reconstruction</b><p>A very small ranking term might improve ordering without sacrificing the calibrated pointwise signal.</p></div>
      <div className="replay-alpha-grid">{evidence.screen.alpha_grid_predeclared.map(alpha => <div key={alpha}><span>{alpha === 0 ? 'Matched control' : 'Candidate weight'}</span><strong>α = {alpha}</strong></div>)}</div>
      <p className="replay-note">Predeclared grid, not a confidence ranking. Expected gain and pre-run compute estimates were not recorded.</p>
      <button className="replay-text-button" onClick={showTree}>Explore all {wave.experiments.length} tested branches →</button>
    </>;
    case 'implement': return <>
      <div className="replay-config"><div><span>Control objective</span><code>BCE</code></div><div className="replay-config-added"><span>Candidate objective</span><code>(1 − α) × BCE<br />+ α × ListCE(sigmoid)</code></div></div>
      <dl className="replay-config-meta"><div><dt>Backbone unchanged</dt><dd>{evidence.screen.matched_configuration.backbone}</dd></div><div><dt>Seed / batch size</dt><dd>{evidence.screen.matched_configuration.seed} / {evidence.screen.matched_configuration.batch_size.toLocaleString()}</dd></div><div><dt>Locked training schedule</dt><dd>{evidence.screen.locked_epoch_count} epochs · same initialization and group order</dd></div><div><dt>Screen-selected α</dt><dd>{evidence.screen.selected_alpha} · locked for confirmation</dd></div></dl>
      <button className="replay-text-button" onClick={() => inspect('screen')}>Inspect full recorded configuration →</button>
    </>;
    case 'train': return <>
      <TrainingChart />
      <div className="replay-facts"><div><strong>{rcr.resource_usage.wall_seconds.toFixed(1)}s</strong><span>RCR aggregate wall time</span></div><div><strong>{rcr.resource_usage.cpu_hours.toFixed(4)}h</strong><span>RCR CPU · 0 GPU hours</span></div></div>
      <p className="replay-note">Resource totals cover the RCR experiment, not just this plotted screen fit. Recorded peak RAM: {Math.round(rcr.resource_usage.peak_rss_mb)} MB.</p>
      <button className="replay-text-button" onClick={() => inspect('screen')}>Inspect exact per-epoch losses →</button>
    </>;
    case 'screen': return <>
      <div className="replay-gate pass">✓ Screen passed <span>249,694 impressions · 15–21 Apr</span></div>
      <MetricComparison base={controlRun.metrics} candidate={selectedRun.metrics} delta={rcr.screen_delta} baselineLabel="Matched screen control" />
      <p className="replay-note">These screen scores use a different window from the champion. Do not compare their absolute values.</p>
      <button className="replay-text-button" onClick={() => inspect('screen')}>Inspect the four tested weights →</button>
    </>;
    case 'confirm': return <>
      <div className="replay-gate pass">✓ Standalone gain confirmed <span>124,909 impressions · 22–28 Apr</span></div>
      <MetricComparison base={evidence.confirmation.control.metrics} candidate={evidence.confirmation.candidate.metrics} delta={evidence.confirmation.deltas_vs_control} baselineLabel="Matched confirmation control" />
      <div className="replay-callout"><b>Next question: does it help the champion?</b><p>The candidate’s primary is {evidence.confirmation.candidate.metrics.primary.toFixed(6)}, not the champion’s {evidence.champion.champion.primary.toFixed(6)}. Integration still has to pass.</p></div>
    </>;
    default: return <>
      <div className="replay-gate reject">Not promoted <span>Frozen champion retained</span></div>
      <MetricComparison base={evidence.audit.baseline_metrics} candidate={evidence.audit.fixed_residual_metrics} delta={evidence.audit.fixed_residual_gains} baselineLabel="Frozen champion" />
      <div className="replay-folds" aria-label="Four user-fold gate outcomes">{evidence.audit.folds.map(fold => <div key={fold.fold} className={fold.all_metrics_nonnegative ? 'pass' : 'reject'}><span>Fold {fold.fold + 1}</span><b>{fold.all_metrics_nonnegative ? 'Pass' : 'Regressed'}</b><small>Δ {signed(fold.gains.primary)}</small></div>)}</div>
      <p className="replay-note">Lesson: standalone lift ≠ ensemble lift. These are empirical folds, not a confidence interval or proof of statistical significance.</p>
      <button className="replay-text-button" onClick={() => inspect('audit')}>Inspect every fold and rejection rule →</button>
    </>;
  }
}

export function ResearchTree({ selected, select }: { selected: string; select: (id: string) => void }) {
  const maxWall = Math.max(...wave.experiments.map(item => item.resource_usage.wall_seconds));
  const branch = wave.experiments.find(item => item.id === selected) ?? rcr;
  return <div className="replay-tree">
    <div className="replay-tree-root"><span>FROZEN CHAMPION</span><strong>{evidence.champion.champion.primary.toFixed(6)}</strong><small>Retained · 0 promotions in this wave</small></div>
    <p className="replay-note">Branches are alternatives tested in the same wave, not a timestamped search order. Line thickness scales linearly with recorded wall time.</p>
    <div className="replay-tree-branches">{wave.experiments.map(item => <button className={item.id === selected ? 'selected' : ''} key={item.id} onClick={() => select(item.id)} aria-pressed={item.id === selected} style={{ '--branch-width': `${1 + 5 * item.resource_usage.wall_seconds / maxWall}px` } as CSSProperties}><span className="replay-tree-connector" /><span><b>{item.method}</b><small>{item.id === 'rcr' ? 'Confirmed base gain; integration rejected' : item.status.includes('zero_selected') ? 'Rejected · zero weight retained' : item.status === 'architecture_selection_rejected' ? 'Rejected at architecture selection' : 'Rejected at screen'}</small></span><time>{item.resource_usage.wall_seconds.toFixed(1)}s</time></button>)}</div>
    <div className="replay-branch-detail"><p className="eyebrow">Selected branch · recorded results</p><h4>{branch.method}</h4><p>Screen Δ primary <b className={branch.screen_delta.primary >= 0 ? 'positive' : 'replay-negative'}>{signed(branch.screen_delta.primary)}</b></p><p>CPU {branch.resource_usage.cpu_hours.toFixed(4)}h · Peak RAM {Math.round(branch.resource_usage.peak_rss_mb)} MB</p><code>{branch.status}</code><p className="replay-note">Gate: {wave.protocol.promotion_gate}</p><details><summary>Inspect branch record & artifact references</summary><pre>{JSON.stringify(branch, null, 2)}</pre></details></div>
    <p className="replay-note">Grey = rejected branch. Green = retained champion. This wave records no promoted or safety-failed branch.</p>
  </div>;
}

export default function ResearchReplay({ suspended = false }: { suspended?: boolean }) {
  const [playback, dispatch] = useReducer(replayReducer, initialReplayState);
  const { time, playing, speed, mode } = playback;
  const [tree, setTree] = useState(false);
  const [branch, setBranch] = useState('rcr');
  const [sourceId, setSourceId] = useState<string | null>(null);
  const index = stepAt(time);
  const step = replaySteps[index];
  const selectedSource = evidence.sources.find(item => item.id === (sourceId ?? step.sources[0]))!;
  const sourceChoices = mode === 'audit' ? evidence.sources : evidence.sources.filter(item => (step.sources as readonly string[]).includes(item.id));
  useEffect(() => {
    if (!playing || suspended) return;
    let last = performance.now();
    const timer = window.setInterval(() => { const now = performance.now(); dispatch({ type: 'tick', value: (now - last) / 1000 }); last = now; }, 250);
    return () => window.clearInterval(timer);
  }, [playing, speed, suspended]);
  useEffect(() => {
    const pauseWhenHidden = () => { if (document.hidden) dispatch({ type: 'pause' }); };
    document.addEventListener('visibilitychange', pauseWhenHidden);
    return () => document.removeEventListener('visibilitychange', pauseWhenHidden);
  }, []);
  function jump(next: number) { dispatch({ type: 'seek', value: stepStart(next) }); setSourceId(null); setTree(false); }
  function inspect(id: SourceId) { dispatch({ type: 'mode', value: 'audit' }); setSourceId(id); }
  function showTree() { dispatch({ type: 'mode', value: 'audit' }); setTree(true); setSourceId('wave'); }
  const result = evidence.showcase.result;

  return <section className={`research-replay ${mode === 'audit' ? 'replay-audit' : ''}`} aria-label="Agent Research Replay">
    <div className="metrics-grid replay-metrics">
      <article className="metric-card featured"><span>Recorded champion · primary</span><strong>{result.champion_primary.toFixed(6)}</strong><small>Public validation · not hidden test</small></article>
      <article className="metric-card"><span>GAUC</span><strong>{result.gauc.toFixed(6)}</strong><small>User-level discrimination</small></article>
      <article className="metric-card"><span>nDCG@5</span><strong>{result.ndcg5.toFixed(6)}</strong><small>Top-five ranking quality</small></article>
      <article className="metric-card"><span>Historical lift</span><strong>+{result.relative_gain_percent.toFixed(2)}%</strong><small>vs. {result.baseline_primary.toFixed(6)} baseline</small></article>
      <article className="metric-card"><span>Recorded research wave</span><strong>{wave.experiments.length} methods</strong><small>{wave.totals.screen_survivors} screen survivor · {wave.totals.champion_promotions} promotions</small></article>
    </div>
    <div className="replay-intro"><div><p className="eyebrow">Evidence, not a monologue</p><h2>Agent Research Replay</h2><p>A promising idea. A controlled test. A decision you can inspect.</p></div><div className="replay-mode" aria-label="Presentation mode"><button aria-pressed={mode === 'story'} onClick={() => { dispatch({ type: 'mode', value: 'story' }); setTree(false); setSourceId(null); }}>Story</button><button aria-pressed={mode === 'audit'} onClick={() => dispatch({ type: 'mode', value: 'audit' })}>Audit</button></div></div>
    <p className="replay-provenance">Recorded reports · curated {clock(replayDuration)} sequence, not an agent transcript. The champion predates this experiment; replay does not train or change it.</p>
    <section className="replay-player panel" aria-label="Replay playback">
      <div className="replay-controls"><button className="button primary" onClick={() => { if (playing) dispatch({ type: 'pause' }); else { dispatch({ type: 'play' }); setTree(false); setSourceId(null); } }}>{playing ? 'Pause' : time >= replayDuration ? 'Replay story' : 'Play story'}</button><button className="replay-skip" aria-label="Previous stage" disabled={index === 0} onClick={() => jump(index - 1)}>←</button><button className="replay-skip" aria-label="Next stage" disabled={index === replaySteps.length - 1} onClick={() => jump(index + 1)}>→</button><span className="mono">{clock(time)} / {clock(replayDuration)}</span><label>Speed <select value={speed} onChange={event => dispatch({ type: 'speed', value: Number(event.target.value) })}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></label><small>{mode === 'audit' ? 'Audit pauses playback. Play story to resume.' : 'Presentation time, not training time'}</small></div>
      <input aria-label="Replay position" aria-valuetext={`${clock(time)}, ${step.stage}: ${step.title}`} type="range" min={0} max={replayDuration} step={0.25} value={time} onChange={event => { dispatch({ type: 'seek', value: Number(event.target.value) }); setSourceId(null); setTree(false); }} />
      <nav className="replay-stages" aria-label="Jump to research stage">{replaySteps.map((item, i) => <button key={item.id} aria-current={i === index ? 'step' : undefined} onClick={() => jump(i)}><span>{String(i + 1).padStart(2, '0')}</span>{item.stage}{item.id === 'screen' ? ' · screen' : item.id === 'confirm' ? ' · confirm' : ''}</button>)}</nav>
    </section>
    <div className="replay-grid">
      <section className="panel replay-canvas"><div className="replay-canvas-heading"><p className="eyebrow">{tree ? 'The research landscape' : `${String(index + 1).padStart(2, '0')} / 08 · ${step.stage}`}</p><button className="replay-text-button" aria-pressed={tree} onClick={() => { if (tree) setTree(false); else showTree(); }}>{tree ? '← Stage view' : 'Research tree ↗'}</button></div><h3>{tree ? 'One champion. Ten alternatives.' : step.title}</h3><p>{tree ? 'Click a branch to inspect its outcome and measured cost.' : step.subtitle}</p>{tree ? <ResearchTree selected={branch} select={setBranch} /> : <StageCanvas id={step.id} inspect={inspect} showTree={showTree} />}</section>
      <section className="panel replay-decision" aria-live="polite" aria-atomic="true"><p className="eyebrow">Decision card · editorial summary</p><h3>{tree ? 'The same gate for every idea.' : 'Why this step?'}</h3>{tree ? <dl><dt>What was tested</dt><dd>{wave.experiments.find(item => item.id === branch)!.method}</dd><dt>Why this comparison</dt><dd>Every alternative must improve its matched control before it can challenge the frozen champion.</dd><dt>Evidence</dt><dd>The selected branch record and wave protocol are available in the evidence trail.</dd><dt>What would prove it wrong</dt><dd>{wave.protocol.promotion_gate}</dd></dl> : <dl><dt>What I noticed</dt><dd>{step.observation}</dd><dt>Why this test</dt><dd>{step.rationale}</dd><dt>What I’m testing</dt><dd>{step.testing}</dd><dt>What would prove me wrong</dt><dd>{step.falsifier}</dd></dl>}<div className="replay-decision-footer">Observable evidence + editorial explanation. Not private chain-of-thought.</div></section>
      <aside className="panel replay-evidence" id="replay-evidence"><p className="eyebrow">Evidence trail · {mode === 'audit' ? 'all artifacts' : 'this step'}</p><h3>Follow the receipts</h3><div className="replay-source-list">{sourceChoices.map(source => <button key={source.id} className={source.id === selectedSource.id ? 'selected' : ''} onClick={() => inspect(source.id)} aria-pressed={source.id === selectedSource.id}><span>↳</span><b>{source.title}</b><small>JSON</small></button>)}</div><div className="replay-source-detail" key={selectedSource.id}><b>{selectedSource.title}</b><p>{selectedSource.path}</p><details open={mode === 'audit'}><summary>Inspect report extract</summary><pre tabIndex={0} aria-label={`${selectedSource.title} JSON extract`}>{JSON.stringify(selectedSource.excerpt, null, 2)}</pre></details><details><summary>Source fingerprint · SHA-256</summary><code>{selectedSource.sha256}</code><p>Hash of the original report file. This view contains selected fields, not the full file. A matching hash identifies content; it does not independently prove the experiment.</p></details></div><a className="button ghost" href="/research-replay.json" download="kuailab-research-evidence.json" onClick={() => dispatch({ type: 'pause' })}>Download evidence bundle ↓</a><p className="replay-note">7 report extracts + source fingerprints. No raw outcomes or model binaries.</p></aside>
    </div>
    {mode === 'audit' && <section className="panel replay-audit-notes"><div><p className="eyebrow">Audit boundaries</p><h3>What this evidence does—and doesn’t—say</h3></div><ul><li>Primary = 0.5 × GAUC + 0.5 × nDCG@5. All headline scores are public-validation results, not hidden-test or live-demo scores.</li><li>The baseline-to-champion lift is historical context. This RCR wave made no champion promotion.</li><li>Four user-ID folds are reported; confidence intervals, pre-run hypothesis rankings, and profiling diagnostics were not recorded here.</li><li>Paper relevance and decision explanations are editorial summaries. No private reasoning, invented tool calls, or search timestamps are shown.</li><li>The bundle contains selected report fields. Reproduction still needs the original data, code, and model artifacts.</li></ul></section>}
  </section>;
}
