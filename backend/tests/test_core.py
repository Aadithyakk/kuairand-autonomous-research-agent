from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from backend.kuailab.benchmark import SyntheticBenchmark, validate_metrics
from backend.kuailab.config import Settings
from backend.kuailab.engine import CampaignEngine
from backend.kuailab.provider import DemoProvider, OpenAIProvider
from backend.kuailab.pairwise import sample_pair_indices
from backend.kuailab.resources import normalize_resource_usage
from backend.kuailab.state import StateStore


class CoreTests(unittest.TestCase):
    @staticmethod
    def wait_for(engine: CampaignEngine) -> None:
        deadline = time.time() + 5
        while engine.running and time.time() < deadline:
            time.sleep(0.02)

    def test_demo_proposal_is_auditable(self):
        proposal = DemoProvider().propose({"iteration": 1, "epsilon": 0.002, "steering": None})
        self.assertIn("def configure_experiment", proposal.code)
        self.assertTrue(proposal.hypothesis)
        self.assertEqual(proposal.usage["total_tokens"], 0)

    def test_responses_output_text_fallback(self):
        response = {"output": [{"content": [{"type": "output_text", "text": "{\"ok\":true}"}]}]}
        self.assertEqual(OpenAIProvider._output_text(response), '{"ok":true}')

    def test_pairwise_sampler_uses_same_user_logged_negatives(self):
        users = ["u1", "u1", "u1", "u2", "u2", "u3"]
        labels = np.asarray([1, 0, 1, 1, 1, 0], dtype=np.float32)
        positives, negatives = sample_pair_indices(users, labels, np.random.default_rng(7))
        self.assertEqual(len(positives), 2)
        for positive, negative in zip(positives, negatives):
            self.assertEqual(users[positive], users[negative])
            self.assertEqual(labels[positive], 1)
            self.assertEqual(labels[negative], 0)

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
            self.wait_for(engine)
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
            self.assertEqual(snapshot["iterations"][-1]["experiment_type"], "fm_config")
            self.assertTrue(snapshot["iterations"][-1]["parameters"])
            self.assertEqual(json.loads((Path(directory) / "state.json").read_text())["version"], 4)

    def test_campaign_continuation_preserves_champion_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(max_iterations=1, max_hours=1, stage_delay_seconds=0, state_dir=Path(directory))
            store = StateStore(Path(directory), settings.public_dict())
            engine = CampaignEngine(settings, store)
            limits = {"max_iterations": 1, "max_hours": 1, "convergence_epsilon": 0.0001, "convergence_patience": 0}
            engine.start("demo", "demo", limits)
            self.wait_for(engine)
            before = store.snapshot()
            previous_champion = before["metrics"]["champion"]["primary"]
            previous_wall = before["usage"]["wall_seconds"]
            previous_iterations = list(before["iterations"])

            engine.continue_campaign(limits)
            self.wait_for(engine)
            after = store.snapshot()

            self.assertEqual(after["iterations"][:2], previous_iterations)
            self.assertEqual(len(after["iterations"]), 3)
            self.assertGreaterEqual(after["metrics"]["champion"]["primary"], previous_champion)
            self.assertGreaterEqual(after["usage"]["wall_seconds"], previous_wall)
            self.assertEqual(after["campaign"]["continuations"], 1)
            self.assertEqual(after["campaign"]["session_start_iteration"], 2)
            self.assertEqual(after["campaign"]["limits"]["max_iterations"], 1)

    def test_run_limits_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(stage_delay_seconds=0, state_dir=Path(directory))
            engine = CampaignEngine(settings, StateStore(Path(directory), settings.public_dict()))
            with self.assertRaisesRegex(ValueError, "max_iterations"):
                engine.start("demo", "demo", {"max_iterations": 0})

    def test_verified_real_champion_can_seed_future_search(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(stage_delay_seconds=0, state_dir=Path(directory))
            engine = CampaignEngine(settings, StateStore(Path(directory), settings.public_dict()))
            retained = engine._verified_champion({"primary": 0.6014695167541504, "gauc": 0.6671333909034729, "ndcg5": 0.5358057022094727})
            self.assertIsNotNone(retained)
            self.assertEqual(retained["experiment_type"], "fm_temporal_deep_blend")
            self.assertAlmostEqual(retained["metrics"]["primary"], 0.6058847904205322)
            self.assertEqual(retained["parameters"]["deep_blend_weight"], 0.23)
            self.assertEqual(retained["parameters"]["temporal_blend_weight"], 0.024)


if __name__ == "__main__":
    unittest.main()
