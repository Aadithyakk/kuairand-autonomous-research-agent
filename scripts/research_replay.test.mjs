import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import ts from 'typescript';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import * as replayModel from '../app/replay-model.ts';
import { replaySteps, replayDuration, clampTime, stepStart, stepAt, advanceTime, clock, initialReplayState, replayReducer } from '../app/replay-model.ts';

const bundle = JSON.parse(readFileSync(new URL('../public/research-replay.json', import.meta.url), 'utf8'));
const close = (actual, expected, tolerance = 5e-8) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test('eight stages form a bounded 2:30 story with no gaps', () => {
  assert.equal(replayDuration, 150);
  assert.equal(replaySteps.length, 8);
  assert.equal(new Set(replaySteps.map(step => step.id)).size, 8);
  replaySteps.forEach((step, index) => {
    assert.equal(stepAt(stepStart(index)), index);
    assert.equal(stepAt(stepStart(index) + step.duration - 0.01), index);
  });
  assert.equal(stepAt(150), 7);
  assert.equal(stepAt(1000), 7);
  assert.equal(stepAt(-1), 0);
});
test('playback clamps time and supports only declared speeds', () => {
  assert.equal(clampTime(NaN), 0);
  assert.equal(clampTime(-10), 0);
  assert.equal(clampTime(200), 150);
  assert.equal(advanceTime(10, 2, 2), 14);
  assert.equal(advanceTime(10, 2, 0.5), 11);
  assert.equal(advanceTime(10, 2, 42), 12);
  assert.equal(advanceTime(10, -4, 1), 10);
  assert.equal(advanceTime(10, NaN, 1), 10);
  assert.equal(clock(150), '2:30');
});
test('play, pause, resume and end-of-story reset are deterministic', () => {
  let state = replayReducer(initialReplayState, { type: 'play' });
  state = replayReducer(state, { type: 'tick', value: 30 });
  assert.equal(state.time, 30);
  state = replayReducer(state, { type: 'pause' });
  assert.equal(replayReducer(state, { type: 'tick', value: 15 }).time, 30);
  state = replayReducer(state, { type: 'play' });
  state = replayReducer(state, { type: 'tick', value: 500 });
  assert.equal(state.time, 150);
  assert.equal(state.playing, false);
  state = replayReducer(state, { type: 'play' });
  assert.equal(state.time, 0);
  assert.equal(state.playing, true);
});
test('scrubbing and entering audit pause; play resumes story at the same position', () => {
  let state = replayReducer(initialReplayState, { type: 'play' });
  state = replayReducer(state, { type: 'seek', value: 42 });
  assert.equal(state.playing, false);
  state = replayReducer(state, { type: 'play' });
  state = replayReducer(state, { type: 'mode', value: 'audit' });
  assert.equal(state.playing, false);
  assert.equal(replayReducer(state, { type: 'tick', value: 20 }).time, 42);
  state = replayReducer(state, { type: 'play' });
  assert.equal(state.mode, 'story');
  assert.equal(state.time, 42);
  state = replayReducer(state, { type: 'speed', value: 2 });
  assert.equal(replayReducer(state, { type: 'tick', value: 3 }).time, 48);
  assert.equal(replayReducer(state, { type: 'speed', value: 99 }).speed, 2);
});
test('every decision has observation, rationale, test, falsifier and resolvable evidence', () => {
  const ids = new Set(bundle.sources.map(source => source.id));
  for (const step of replaySteps) {
    for (const key of ['observation', 'rationale', 'testing', 'falsifier']) assert.ok(step[key].length > 20);
    for (const id of step.sources) assert.ok(ids.has(id), `Missing source ${id}`);
  }
});
test('downloaded extracts exactly match their allowlisted source fields and hashes', () => {
  assert.equal(bundle.sources.length, 7);
  for (const source of bundle.sources) {
    assert.match(source.path, /^(public|results)\/[a-z0-9/-]+\.json$/);
    const raw = readFileSync(new URL(`../${source.path}`, import.meta.url));
    assert.equal(createHash('sha256').update(raw).digest('hex'), source.sha256);
    const original = JSON.parse(raw);
    for (const [field, value] of Object.entries(source.excerpt)) assert.deepEqual(value, original[field]);
    assert.deepEqual(bundle[source.id], source.excerpt);
  }
});
test('headline is the frozen champion, not the RCR candidate or synthetic live state', () => {
  const { primary, gauc, ndcg5 } = bundle.champion.champion;
  close(primary, (gauc + ndcg5) / 2);
  close(primary, bundle.showcase.result.champion_primary);
  close(primary, bundle.audit.baseline_metrics.primary);
  close(primary, bundle.manifest.validation_metrics.primary);
  assert.notEqual(primary, bundle.confirmation.candidate.metrics.primary);
  close(bundle.showcase.result.absolute_gain, primary - bundle.champion.baseline_primary);
  assert.equal(bundle.wave.totals.champion_promotions, 0);
});
test('screen and confirmation pass but champion residual fails on all global metrics', () => {
  const rcr = bundle.wave.experiments.find(item => item.id === 'rcr');
  for (const key of ['primary', 'GAUC', 'nDCG@5']) {
    assert.ok(rcr.screen_delta[key] > 0);
    assert.ok(rcr.confirmation_delta[key] > 0);
    assert.ok(bundle.audit.fixed_residual_gains[key] < 0);
    close(bundle.audit.fixed_residual_metrics[key] - bundle.audit.baseline_metrics[key], bundle.audit.fixed_residual_gains[key]);
  }
  assert.equal(bundle.audit.merits_promotion, false);
  assert.equal(bundle.audit.folds.length, 4);
  assert.equal(bundle.audit.folds.filter(fold => fold.all_metrics_nonnegative).length, 1);
  for (const fold of bundle.audit.folds) assert.equal(fold.all_metrics_nonnegative, Object.values(fold.gains).every(gain => gain >= 0));
});
test('hidden-test boundary, locked alpha, train curves and branch costs are recorded', () => {
  assert.equal(bundle.champion.split.hidden_test_accessed, false);
  assert.equal(bundle.screen.protocol.hidden_test_outcomes_parsed, false);
  assert.equal(bundle.confirmation.protocol.hidden_test_outcomes_parsed, false);
  assert.equal(bundle.audit.hidden_test_outcomes_parsed, false);
  assert.equal(bundle.confirmation.protocol.retuning, false);
  assert.equal(bundle.confirmation.selected_alpha_locked, bundle.screen.selected_alpha);
  for (const run of bundle.screen.runs) assert.equal(run.training_history.length, bundle.screen.locked_epoch_count);
  assert.equal(bundle.wave.experiments.length, 10);
  for (const experiment of bundle.wave.experiments) assert.ok(experiment.resource_usage.wall_seconds > 0);
});
test('replay component cannot invoke training controls or claim a live research transcript', () => {
  const ui = readFileSync(new URL('../app/research-replay.tsx', import.meta.url), 'utf8');
  assert.doesNotMatch(ui, /fetch\(|\/api\/run\/|XMLHttpRequest|WebSocket/);
  assert.match(ui, /not an agent transcript/);
  assert.match(ui, /replay does not train or change it/);
  assert.match(ui, /visibilitychange/);
  assert.match(ui, /clearInterval/);
});

// Exercise every canvas with server rendering, without a browser or any network calls.
const componentUrl = new URL('../app/research-replay.tsx', import.meta.url);
const compiled = ts.transpileModule(readFileSync(componentUrl, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
}).outputText;
const localRequire = createRequire(componentUrl);
const componentModule = { exports: {} };
new Function('require', 'module', 'exports', compiled)(
  id => id === './replay-model' ? replayModel : localRequire(id), componentModule, componentModule.exports,
);
test('all eight stage canvases render their real evidence without runtime errors', () => {
  const checks = {
    inspect: ['124,909', '22,377', 'Missingness'],
    research: ['https://arxiv.org/abs/2211.01494', 'not a simulated live search'],
    hypothesize: ['0.0005', 'Predeclared grid'],
    implement: ['ListCE(sigmoid)', '260830'],
    train: ['training BCE', '73.3s'],
    screen: ['0.613469958', '0.613283992'],
    confirm: ['0.600879371', '0.600762129'],
    reflect: ['Not promoted', 'Fold 4', '0.612855673'],
  };
  for (const step of replaySteps) {
    const html = renderToStaticMarkup(createElement(componentModule.exports.StageCanvas, { id: step.id, inspect() {}, showTree() {} }));
    for (const text of checks[step.id]) assert.ok(html.includes(text), `${step.id}: missing ${text}`);
  }
});
test('every branch is selectable and renders its recorded status and cost', () => {
  for (const experiment of bundle.wave.experiments) {
    const html = renderToStaticMarkup(createElement(componentModule.exports.ResearchTree, { selected: experiment.id, select() {} }));
    assert.equal((html.match(/aria-pressed="true"/g) ?? []).length, 1);
    assert.equal((html.match(/aria-pressed=/g) ?? []).length, 10);
    assert.ok(html.includes(experiment.status));
    assert.ok(html.includes(experiment.resource_usage.wall_seconds.toFixed(1)));
  }
});
test('default replay is deterministic with controls ahead of the canvas and no live metrics', () => {
  const render = () => renderToStaticMarkup(createElement(componentModule.exports.default));
  const html = render();
  assert.equal(html, render());
  assert.ok(html.indexOf('Replay playback') < html.indexOf('replay-grid'));
  assert.ok(html.includes('0.612858'));
  assert.ok(html.includes('Play story'));
  assert.ok(html.includes('download="kuailab-research-evidence.json"'));
  assert.ok(!html.includes('0.6250'));
});
