from __future__ import annotations

import json
import ssl
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
from backend.kuailab.ordinal import build_ordinal_watch_labels
from backend.kuailab.pairwise import sample_pair_indices
from backend.kuailab.resources import normalize_resource_usage
from backend.kuailab.rad import build_rad_labels
from backend.kuailab.slate import build_slate_features
from backend.kuailab.research import load_method_cards, summarize_search_tree
from backend.kuailab.state import StateStore
from scripts.train_dvr_wtg import auxiliary_targets, final_scores, fit_wtg_reference


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
        self.assertEqual(
            {alternative["strategy"] for alternative in proposal.alternatives},
            {"exploit", "explore", "innovate"},
        )
        self.assertIn(proposal.strategy, {"exploit", "explore", "innovate"})

    def test_research_memory_has_prioritized_and_exhausted_cards(self):
        cards = load_method_cards()
        statuses = {card["status"] for card in cards}
        self.assertIn("unattempted-priority", statuses)
        self.assertIn("exhausted", statuses)
        tree = summarize_search_tree([{"number": 0, "title": "root", "status": "baseline", "metrics": {"primary": 0.6}}])
        self.assertEqual(tree[0]["node"], 0)

    def test_responses_output_text_fallback(self):
        response = {"output": [{"content": [{"type": "output_text", "text": "{\"ok\":true}"}]}]}
        self.assertEqual(OpenAIProvider._output_text(response), '{"ok":true}')

    def test_openai_provider_uses_verified_tls_context(self):
        context = OpenAIProvider._ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

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

    def test_slate_features_are_outcome_free_and_session_aware(self):
        base = [
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 0, 8, 1_000),
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 1, 8, 2_000),
            (20220409, "u1", "v2", "a1", "0", 20_000.0, 0, 9, 2_000_000),
        ]
        changed_labels = [tuple(list(row[:6]) + [1 - row[6]] + list(row[7:])) for row in base]
        features, names = build_slate_features(base)
        changed, _ = build_slate_features(changed_labels)
        np.testing.assert_allclose(features, changed)
        self.assertEqual(features.shape, (3, len(names)))
        session_fraction = names.index("session_fraction")
        self.assertEqual(features[0, session_fraction], features[1, session_fraction])
        self.assertGreater(features[2, session_fraction], features[1, session_fraction])

    def test_rad_labels_use_video_and_user_duration_reference_groups(self):
        rows = [
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 0, 8, 1_000, 1_000.0),
            (20220408, "u2", "v1", "a1", "0", 10_000.0, 1, 8, 2_000, 9_000.0),
            (20220408, "u1", "v2", "a2", "0", 30_000.0, 1, 8, 3_000, 15_000.0),
            (20220408, "u2", "v2", "a2", "0", 30_000.0, 0, 8, 4_000, 2_000.0),
        ]
        try:
            labels, stats = build_rad_labels(rows, duration_bins=2)
        except RuntimeError as error:
            if "SciPy" in str(error):
                self.skipTest("RAD deep optional dependency is not installed")
            raise
        self.assertEqual(labels.shape, (4,))
        self.assertTrue(np.all((labels > 0) & (labels < 1)))
        self.assertGreater(labels[1], labels[0])
        self.assertGreater(labels[2], labels[3])
        self.assertEqual(stats["duration_bins"], 2)

    def test_ordinal_watch_labels_are_nested_and_duration_capped(self):
        rows = [
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 0, 8, 1_000, 4_000.0),
            (20220408, "u1", "v2", "a1", "0", 30_000.0, 1, 8, 2_000, 14_000.0),
        ]
        labels, stats = build_ordinal_watch_labels(rows)
        np.testing.assert_array_equal(labels[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(labels[1], [1.0, 1.0, 1.0])
        self.assertEqual(labels.shape, (2, 3))
        self.assertEqual(stats["thresholds"], [0.25, 0.5, 0.75])

    def test_dvr_wtg_reference_is_finite_and_preserves_within_duration_order(self):
        rows = [
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 0, 1_000.0, 1_000),
            (20220408, "u2", "v2", "a2", "0", 10_000.0, 1, 9_000.0, 2_000),
            (20220408, "u3", "v3", "a3", "0", 20_000.0, 1, 15_000.0, 3_000),
        ]
        reference = fit_wtg_reference(rows)
        gain, duration_z = auxiliary_targets(rows, reference)
        self.assertTrue(np.all(np.isfinite(gain)))
        self.assertTrue(np.all(np.isfinite(duration_z)))
        self.assertLess(gain[0], gain[1])
        self.assertEqual(reference["observed_duration_buckets"], 2)

    def test_dvr_wtg_zero_rank_blend_preserves_long_view_scores(self):
        long_view = np.asarray([0.7, 0.2, 0.8], dtype=np.float32)
        wtg = np.asarray([0.1, 0.9, 0.3], dtype=np.float32)
        output = final_scores(["u1", "u1", "u2"], long_view, wtg, 0.0)
        np.testing.assert_array_equal(output, long_view)

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
