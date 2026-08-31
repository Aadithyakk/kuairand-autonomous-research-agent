# Submission Provenance

This package is based on Git commit `fcfc555f517d425f856c44efc1540f040ca770c1` (`Rebuild autonomous KuaiRand research agent`, 29 August 2026).

That commit is the original KuaiLab autonomous system and its five-iteration run. It predates the later BPR, DeepFM, feature-lab, teacher, prequential, and manual research-wave additions. None of those additions are included here.

Submission packaging makes only these compliance and reproducibility changes:

1. Implements the organizer's 31 August clarification of cumulative scored-iteration convergence. Re-evaluating the recorded trajectory still stops after proposal 5 and retains iteration 4.
2. Adds a frozen champion reproduction and blind CSV exporter. It does not expand the agent's research action space or change the recorded model.
3. Adds setup, Devpost, results, limitations, contribution, and checklist documentation.
4. Adds generated final checkpoint/encoding/output artifacts from the frozen original champion.

The original evidence in `results/run-9ecfd2aa09/` is preserved unchanged.
