from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from backend.kuailab.benchmark import SyntheticBenchmark, validate_metrics
from backend.kuailab.config import Settings
from backend.kuailab.engine import CampaignEngine
from backend.kuailab.provider import DemoProvider, OpenAIProvider
from backend.kuailab.resources import normalize_resource_usage
from backend.kuailab.state import StateStore


class CoreTests(unittest.TestCase):
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

    def test_resource_usage_derives_hours_and_utilization(self):
        usage = normalize_resource_usage({"wall_seconds": 10, "train_seconds": 8, "cpu_seconds": 20, "gpu_seconds": 5, "gpu_count": 1})
        self.assertEqual(usage["cpu_hours"], round(20 / 3600, 6))
        self.assertEqual(usage["gpu_hours"], round(5 / 3600, 6))
        self.assertEqual(usage["cpu_utilization_percent"], 200.0)

    def test_synthetic_failure_is_visible(self):
        proposal = DemoProvider().propose({"iteration": 4, "epsilon": 0.002, "steering": None})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "out-of-memory") as caught:
                SyntheticBenchmark().evaluate(proposal, 4, Path(directory))
            self.assertGreater(caught.exception.resource_usage["cpu_seconds"], 0)
            self.assertTrue((Path(directory) / "resource-usage.json").exists())

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
            self.assertGreater(snapshot["usage"]["cpu_hours"], 0)
            self.assertEqual(snapshot["usage"]["experiments_measured"], 2)
            self.assertTrue((Path(directory) / "campaigns" / snapshot["campaign"]["id"] / "resource-summary.json").exists())
            self.assertEqual(json.loads((Path(directory) / "state.json").read_text())["version"], 3)


if __name__ == "__main__":
    unittest.main()
