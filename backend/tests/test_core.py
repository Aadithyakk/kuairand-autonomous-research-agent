from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from backend.kuailab.benchmark import SyntheticBenchmark, validate_metrics
from backend.kuailab.config import Settings
from backend.kuailab.engine import CampaignEngine, convergence_window
from backend.kuailab.provider import DemoProvider, OpenAIProvider
from backend.kuailab.state import StateStore


class CoreTests(unittest.TestCase):
    def test_cumulative_convergence_matches_original_campaign(self):
        baseline = 0.6014695167541504
        scored = [0.6034932136535645, 0.6036843061447144, 0.6037807464599609]
        self.assertEqual(convergence_window(scored, baseline, 0.002, 3), (False, scored[-1] - baseline))
        scored.append(0.6036427021026611)
        converged, improvement = convergence_window(scored, baseline, 0.002, 3)
        self.assertTrue(converged)
        self.assertAlmostEqual(improvement, 0.6037807464599609 - 0.6034932136535645)

    def test_failure_does_not_advance_convergence_window(self):
        # A failed iteration is not appended, so two scored results cannot fill N=3.
        self.assertEqual(convergence_window([0.61, 0.611], 0.60, 0.002, 3), (False, None))

    def test_demo_proposal_is_auditable(self):
        proposal = DemoProvider().propose({"iteration": 1, "epsilon": 0.002, "steering": None})
        self.assertIn("def configure_experiment", proposal.code)
        self.assertTrue(proposal.hypothesis)
        self.assertEqual(proposal.usage["total_tokens"], 0)

    def test_responses_output_text_fallback(self):
        response = {"output": [{"content": [{"type": "output_text", "text": "{\"ok\":true}"}]}]}
        self.assertEqual(OpenAIProvider._output_text(response), '{"ok":true}')

    def test_metric_validation_rejects_impossible_values(self):
        with self.assertRaises(ValueError):
            validate_metrics({"primary": 1.2, "gauc": 0.6, "ndcg5": 0.5})

    def test_synthetic_failure_is_visible(self):
        proposal = DemoProvider().propose({"iteration": 4, "epsilon": 0.002, "steering": None})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "out-of-memory"):
                SyntheticBenchmark().evaluate(proposal, 4, Path(directory))

    def test_campaign_persists_iterations_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(max_iterations=2, max_hours=1, stage_delay_seconds=0, state_dir=Path(directory))
            store = StateStore(Path(directory), settings.public_dict())
            engine = CampaignEngine(settings, store)
            engine.start("demo", "demo")
            deadline = time.time() + 5
            while engine.running and time.time() < deadline:
                time.sleep(0.02)
            snapshot = store.snapshot()
            self.assertEqual(snapshot["campaign"]["status"], "complete")
            self.assertEqual(len(snapshot["iterations"]), 3)
            self.assertGreater(snapshot["metrics"]["champion"]["primary"], snapshot["metrics"]["baseline"]["primary"])
            artifact = Path(snapshot["iterations"][-1]["artifact"])
            self.assertTrue((artifact / "proposal.json").exists())
            self.assertTrue((artifact / "candidate.diff").exists())
            self.assertEqual(json.loads((Path(directory) / "state.json").read_text())["version"], 2)


if __name__ == "__main__":
    unittest.main()
