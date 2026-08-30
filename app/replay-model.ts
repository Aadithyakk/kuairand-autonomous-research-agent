export const replaySteps = [
  { id: 'inspect', stage: 'Inspect', duration: 18, title: 'Start with the boundary, not the score.', subtitle: 'A temporal split keeps future outcomes out of model selection.', observation: 'The frozen champion was evaluated on 124,909 impressions from 22,377 users.', rationale: 'Keep model selection, the locked screen, and final confirmation separate.', testing: 'Whether a new ranking objective can improve recommendations under the same temporal protocol.', falsifier: 'Any use of hidden-test outcomes invalidates the comparison.', sources: ['showcase', 'champion', 'wave'] },
  { id: 'research', stage: 'Research', duration: 17, title: 'Better ranking. Still calibrated.', subtitle: 'A recorded paper reference leads to a controlled loss-function experiment.', observation: 'The RCR screen cites a regression-compatible ranking objective.', rationale: 'Test a small ranking term while keeping the pointwise model and its control matched.', testing: 'A ranking objective compatible with probability calibration, inspired by Bai et al.', falsifier: 'A paper result alone is not evidence of improvement on this KuaiRand split.', sources: ['screen', 'wave'] },
  { id: 'hypothesize', stage: 'Hypothesize', duration: 17, title: 'Change one thing. Keep a real control.', subtitle: 'Ten methods were tested; this replay follows the only screen survivor.', observation: 'The recorded wave includes loss, distillation, watch-time, and architecture experiments.', rationale: 'Use a matched control so a gain can be attributed to the tested change.', testing: 'Add a small ListCE(sigmoid) term to BCE; test four predeclared weights, including zero.', falsifier: 'Keep alpha at zero unless primary improves by at least 0.0001 and both component metrics improve.', sources: ['screen', 'wave'] },
  { id: 'implement', stage: 'Implement', duration: 18, title: 'A small, inspectable change.', subtitle: 'The experiment changes the loss, not the entire recommender.', observation: 'Initialization and group order are matched. Six epochs were locked before screening.', rationale: 'Control architecture, seed, and training schedule to isolate the objective.', testing: 'BCE becomes (1 − alpha) × BCE + alpha × ListCE(sigmoid).', falsifier: 'A gain dependent on an unmatched setup would not isolate the ranking objective.', sources: ['screen'] },
  { id: 'train', stage: 'Train', duration: 16, title: 'The training run, already recorded.', subtitle: 'No model is training during this replay.', observation: 'The selected screen candidate and control each have six recorded training epochs.', rationale: 'Inspect the measured run rather than animate invented progress.', testing: 'Compare training BCE at the locked schedule; reserve ranking judgment for held-out data.', falsifier: 'Lower training loss is not enough: held-out ranking must also improve.', sources: ['screen', 'wave'] },
  { id: 'screen', stage: 'Evaluate', duration: 18, title: 'First gate: a promising signal.', subtitle: '15–21 Apr · matched-base screen, not champion performance.', observation: 'RCR improves primary, GAUC, and nDCG@5 against its screen control.', rationale: 'Only the screen survivor is allowed to proceed to confirmation.', testing: 'The locked alpha of 0.0005 on a later, untouched confirmation window.', falsifier: 'If either ranking metric regresses on confirmation, reject the candidate.', sources: ['screen', 'wave'] },
  { id: 'confirm', stage: 'Evaluate', duration: 18, title: 'The gain repeats. Is it useful yet?', subtitle: '22–28 Apr · refit through 21 Apr, with no retuning.', observation: 'The candidate improves all three metrics against its matched confirmation control.', rationale: 'A stronger standalone model does not necessarily add value to an ensemble.', testing: 'The predeclared 5% residual against the frozen champion, then four user-ID folds.', falsifier: 'Reject if the champion comparison or any user fold loses a required metric.', sources: ['confirmation', 'audit', 'wave'] },
  { id: 'reflect', stage: 'Reflect', duration: 28, title: 'The right decision was not to deploy.', subtitle: 'A positive standalone result can still be a negative ensemble result.', observation: 'The champion residual loses all three global metrics. Only one of four folds is non-decreasing on all metrics.', rationale: 'Enforce the acceptance rule even when the standalone result is encouraging.', testing: 'Retain the frozen champion and preserve the failed integration as evidence.', falsifier: 'Promotion would require a new, predeclared experiment that passes every gate.', sources: ['audit', 'wave', 'manifest'] },
] as const;

export const replayDuration = replaySteps.reduce((sum, step) => sum + step.duration, 0);
export function clampTime(seconds: number) { return Math.max(0, Math.min(replayDuration, Number.isFinite(seconds) ? seconds : 0)); }
export function stepStart(index: number) { return replaySteps.slice(0, Math.max(0, Math.min(replaySteps.length - 1, index))).reduce((sum, step) => sum + step.duration, 0); }
export function stepAt(seconds: number) {
  const time = clampTime(seconds);
  let end = 0;
  return replaySteps.findIndex((step, index) => { end += step.duration; return time < end || index === replaySteps.length - 1; });
}
export function advanceTime(seconds: number, delta: number, speed: number) { return clampTime(seconds + (Number.isFinite(delta) ? Math.max(0, delta) : 0) * ([0.5, 1, 2].includes(speed) ? speed : 1)); }
export function clock(seconds: number) { const s = Math.floor(clampTime(seconds)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
export function signed(value: number) { return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(9)}`; }

export type ReplayState = { time: number; playing: boolean; speed: number; mode: 'story' | 'audit' };
export const initialReplayState: ReplayState = { time: 0, playing: false, speed: 1, mode: 'story' };
export type ReplayAction = { type: 'play' | 'pause' } | { type: 'tick' | 'seek' | 'speed'; value: number } | { type: 'mode'; value: 'story' | 'audit' };
export function replayReducer(state: ReplayState, action: ReplayAction): ReplayState {
  switch (action.type) {
    case 'play': return { ...state, mode: 'story', playing: true, time: state.time >= replayDuration ? 0 : state.time };
    case 'pause': return { ...state, playing: false };
    case 'seek': return { ...state, time: clampTime(action.value), playing: false };
    case 'speed': return { ...state, speed: [0.5, 1, 2].includes(action.value) ? action.value : state.speed };
    case 'mode': return { ...state, mode: action.value, playing: false };
    case 'tick': {
      if (!state.playing || state.mode !== 'story') return state;
      const time = advanceTime(state.time, action.value, state.speed);
      return { ...state, time, playing: time < replayDuration };
    }
  }
}
