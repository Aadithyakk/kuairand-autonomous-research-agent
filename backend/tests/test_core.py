from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from backend.kuailab.benchmark import SyntheticBenchmark, validate_metrics
from backend.kuailab.champion import blend_with_champion, load_champion_scores, within_user_rank
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

    def test_frozen_champion_is_checksum_verified(self):
        scores, manifest = load_champion_scores(expected_rows=124_909)
        self.assertEqual(len(scores), 124_909)
        self.assertAlmostEqual(manifest["validation_metrics"]["primary"], 0.6128580570220947)
        self.assertFalse(manifest["hidden_test_accessed"])

    def test_champion_residual_blend_preserves_base_at_zero_weight(self):
        users = ["a", "a", "a", "b", "b"]
        champion = np.asarray([0.1, 0.9, 0.4, 0.8, 0.2])
        candidate = np.asarray([0.9, 0.1, 0.4, 0.3, 0.7])
        blended = blend_with_champion(users, champion, candidate, 0.0)
        np.testing.assert_allclose(blended, within_user_rank(users, champion))
        with self.assertRaisesRegex(ValueError, "between -0.25 and 0.25"):
            blend_with_champion(users, champion, candidate, 0.3)

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
            self.assertEqual(json.loads((Path(directory) / "state.json").read_text())["version"], 5)

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
            self.assertEqual(retained["experiment_type"], "offline_slate_reranking")
            self.assertAlmostEqual(retained["metrics"]["primary"], 0.6128580570220947)
            self.assertFalse(retained["budget_counted"])

    def test_failed_worker_is_retried_and_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(max_iterations=4, max_hours=1, convergence_patience=0, stage_delay_seconds=0, state_dir=Path(directory))
            store = StateStore(Path(directory), settings.public_dict())
            engine = CampaignEngine(settings, store)
            engine.start("demo", "demo")
            self.wait_for(engine)
            snapshot = store.snapshot()
            failed = snapshot["iterations"][-1]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(snapshot["campaign"]["recovery_count"], 1)
            self.assertEqual(snapshot["campaign"]["failure_count"], 2)
            self.assertEqual(failed["recovery_events"][0]["outcome"], "failed")
            artifact = Path(failed["artifact"])
            self.assertTrue((artifact / "retry-1" / "proposal.json").exists())
            self.assertTrue((artifact / "iteration-log.json").exists())

    def test_continuation_cannot_exceed_global_iteration_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(stage_delay_seconds=0, state_dir=Path(directory))
            store = StateStore(Path(directory), settings.public_dict())
            def exhaust(state):
                state["campaign"].update(id="run-exhausted", mode="demo", provider="demo", status="complete")
                state["iterations"].extend(
                    {"number": number, "status": "rejected", "budget_counted": True}
                    for number in range(1, 51)
                )
            store.update(exhaust)
            engine = CampaignEngine(settings, store)
            with self.assertRaisesRegex(RuntimeError, "50-iteration"):
                engine.continue_campaign()


if __name__ == "__main__":
    unittest.main()
