/** Build a public, outcome-free evidence extract. Never reads raw datasets or starts training. */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const specs = [
  ['showcase', 'Benchmark & historical result', 'public/judge-showcase.json', ['benchmark', 'result', 'limitations']],
  ['champion', 'Frozen champion report', 'results/verified-slate-consensus/summary.json', ['champion', 'baseline_primary', 'gain_over_baseline', 'split', 'leakage_note']],
  ['wave', 'Ten-method research wave', 'results/calibrated-ranking/summary.json', ['campaign', 'date', 'protocol', 'totals', 'experiments', 'recommendation']],
  ['screen', 'RCR configuration & screen', 'results/calibrated-ranking/rcr-screen.json', ['paper', 'protocol', 'matched_configuration', 'alpha_grid_predeclared', 'locked_epoch_count', 'selected_alpha', 'zero_preferring_selection', 'rows', 'runs', 'merits_confirmation']],
  ['confirmation', 'Locked confirmation', 'results/calibrated-ranking/rcr-confirmation.json', ['protocol', 'control', 'candidate', 'deltas_vs_control', 'selected_alpha_locked', 'matched_candidate_improves_all_metrics']],
  ['audit', 'Champion residual & user folds', 'results/calibrated-ranking/rcr-residual-audit.json', ['baseline_metrics', 'fixed_residual_metrics', 'fixed_residual_gains', 'fixed_weight', 'folds', 'merits_promotion', 'all_four_folds_all_metrics_nonnegative', 'hidden_test_outcomes_parsed', 'integrity_incidents']],
  ['manifest', 'Champion artifact manifest', 'results/final-model/manifest.json', ['artifact', 'validation_metrics', 'validation_scores', 'validation_scores_sha256', 'hidden_test_accessed', 'notes']],
];
const sources = specs.map(([id, title, path, fields]) => {
  const raw = readFileSync(resolve(root, path));
  const parsed = JSON.parse(raw.toString('utf8'));
  const excerpt = Object.fromEntries(fields.map(field => {
    if (!(field in parsed)) throw new Error(`Missing evidence field: ${path} / ${field}`);
    return [field, parsed[field]];
  }));
  return { id, title, path, sha256: createHash('sha256').update(raw).digest('hex'), excerpt };
});
const data = Object.fromEntries(sources.map(source => [source.id, source.excerpt]));
const close = (a, b) => {
  if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a - b) > 5e-8) throw new Error(`Evidence mismatch: ${a} vs ${b}`);
};
const champion = data.champion.champion;
close(champion.primary, (champion.gauc + champion.ndcg5) / 2);
close(champion.primary, data.showcase.result.champion_primary);
close(champion.primary, data.manifest.validation_metrics.primary);
close(champion.primary, data.audit.baseline_metrics.primary);
for (const [key, reportKey] of [['gauc', 'GAUC'], ['ndcg5', 'nDCG@5']]) {
  close(champion[key], data.showcase.result[key]);
  close(champion[key], data.manifest.validation_metrics[key]);
  close(champion[key], data.audit.baseline_metrics[reportKey]);
}
// The editorial story names these historical counts and outcomes. Fail closed if they change.
if (champion.rows !== 124909 || champion.users !== 22377 || data.wave.experiments.length !== 10 || data.wave.totals.screen_survivors !== 1 || data.wave.totals.champion_promotions !== 0 || data.audit.folds.length !== 4 || data.audit.folds.filter(fold => fold.all_metrics_nonnegative).length !== 1) throw new Error('Historical narrative changed; review the replay before regenerating.');
if (data.champion.split.hidden_test_accessed !== false || data.manifest.hidden_test_accessed !== false || data.screen.protocol.hidden_test_outcomes_parsed !== false || data.confirmation.protocol.hidden_test_outcomes_parsed !== false || data.confirmation.protocol.retuning !== false) throw new Error('Temporal evidence boundary changed.');
if (data.screen.paper !== 'https://arxiv.org/abs/2211.01494') throw new Error('Paper citation changed; review the source card.');
if (data.screen.selected_alpha !== data.confirmation.selected_alpha_locked) throw new Error('Confirmation did not retain the screen-selected alpha.');
const rcr = data.wave.experiments.find(item => item.id === 'rcr');
if (!rcr || data.audit.merits_promotion !== false || data.audit.hidden_test_outcomes_parsed !== false) throw new Error('Unexpected replay decision; review the story before regenerating.');
for (const key of ['primary', 'GAUC', 'nDCG@5']) {
  close(rcr.champion_residual_delta[key], data.audit.fixed_residual_gains[key]);
  close(rcr.confirmation_delta[key], data.confirmation.deltas_vs_control[key]);
  close(rcr.screen_delta[key], data.screen.runs.find(run => run.alpha === data.screen.selected_alpha).deltas_vs_control[key]);
}
const bundle = {
  schema_version: 1,
  narrative_type: 'Editorial reconstruction from checked-in reports; not a timestamped agent transcript.',
  time_basis: 'Playback seconds are presentation pacing, not experiment runtime.',
  source_policy: 'Allowlisted aggregate report extracts only. Hashes identify complete source files, not these excerpts. Raw outcomes and model binaries are excluded.',
  sources,
  ...data,
};
const target = resolve(root, 'public/research-replay.json');
const output = `${JSON.stringify(bundle, null, 2)}\n`;
if (process.argv.includes('--check')) {
  if (readFileSync(target, 'utf8') !== output) throw new Error('Replay bundle is stale. Run node scripts/build_research_replay.mjs.');
  console.log(`PASS replay bundle matches ${sources.length} source reports; metrics and rejection agree.`);
} else {
  writeFileSync(target, output);
  console.log(`Built public/research-replay.json from ${sources.length} aggregate reports.`);
}
