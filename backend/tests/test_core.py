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
from backend.kuailab.incubator import (
    EXECUTOR_FAMILY,
    REQUIRED_CONTRACT_TESTS,
    load_executor_registry,
    require_registered_program,
    review_and_register_executor,
)
from backend.kuailab.provider import DemoProvider, OpenAIProvider
from backend.kuailab.ordinal import build_ordinal_watch_labels
from backend.kuailab.pairwise import sample_pair_indices
from backend.kuailab.resources import normalize_resource_usage
from backend.kuailab.rad import build_rad_labels
from backend.kuailab.slate import build_slate_features
from backend.kuailab.research import load_method_cards, load_research_priors, summarize_search_tree
from backend.kuailab.state import StateStore
from backend.kuailab.live_predictor import predict_slate, score_candidate
from backend.kuailab.neighbors import build_multiview_neighbor_scores
from backend.kuailab.paper_executor import build_paper_signal_scores
from scripts.train_dvr_wtg import auxiliary_targets, final_scores, fit_wtg_reference
from scripts.audit_cdm_context_grid import prepare_context, rerank


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

    def test_live_predictor_scores_at_request_time_and_sorts_slate(self):
        artifact = {
            "model": {
                "name": "test logistic",
                "candidate_weights": {"1": 1.0, "2": -1.0},
                "intercept": 0.0,
                "numeric_means": [0.0],
                "numeric_scales": [1.0],
                "numeric_weights": [0.5],
            },
            "users": [{"user_id": "u1", "candidate_count": 2}],
            "target": {"date": "2022-04-29"},
            "evaluation": {"primary": 0.5},
            "integrity": {"target_outcomes_accessed": False},
            "candidates": [
                {"user_id": "u1", "video_id": "low", "author_id": "a", "video_type": "NORMAL", "tab": "0", "hour": 8, "duration_seconds": 10, "exposure_index": 0, "categorical_indices": [2], "numeric_values": [0.0]},
                {"user_id": "u1", "video_id": "high", "author_id": "b", "video_type": "NORMAL", "tab": "1", "hour": 9, "duration_seconds": 20, "exposure_index": 1, "categorical_indices": [1], "numeric_values": [1.0]},
            ],
        }
        self.assertGreater(score_candidate(artifact, artifact["candidates"][1]), 0.5)
        result = predict_slate(artifact, "u1", 2)
        self.assertEqual([row["video_id"] for row in result["ranking"]], ["high", "low"])
        self.assertNotIn("categorical_indices", result["ranking"][0])

    def test_exported_live_candidates_contain_no_outcomes_or_saved_scores(self):
        artifact_path = Path(__file__).resolve().parents[2] / "public" / "live-predictor.json"
        if not artifact_path.exists():
            self.skipTest("live predictor has not been trained")
        artifact = json.loads(artifact_path.read_text())
        forbidden = {"long_view", "play_time_ms", "is_click", "is_like", "score"}
        self.assertFalse({key for row in artifact["candidates"] for key in row}.intersection(forbidden))
        self.assertFalse(artifact["integrity"]["target_outcomes_accessed"])
        self.assertFalse(artifact["integrity"]["target_scores_precomputed"])

    def test_research_memory_has_prioritized_and_exhausted_cards(self):
        cards = load_method_cards()
        statuses = {card["status"] for card in cards}
        self.assertIn("unattempted-priority", statuses)
        self.assertIn("exhausted", statuses)
        tree = summarize_search_tree([{"number": 0, "title": "root", "status": "baseline", "metrics": {"primary": 0.6}}])
        self.assertEqual(tree[0]["node"], 0)

    def test_online_teacher_is_directional_submission_safe_prior(self):
        prior = load_research_priors()["online_teacher_distillation"]
        self.assertAlmostEqual(prior["teacher"]["metrics"]["primary"], 0.7234153747558594)
        self.assertIn("not a static submission", prior["teacher"]["evaluation_mode"])
        self.assertTrue(any("without hidden outcomes" in item for item in prior["submission_safe_translation"]))
        self.assertIn("Do not report 0.723415", prior["prohibited_claim"])

    def test_responses_output_text_fallback(self):
        response = {"output": [{"content": [{"type": "output_text", "text": "{\"ok\":true}"}]}]}
        self.assertEqual(OpenAIProvider._output_text(response), '{"ok":true}')

    def test_academic_search_trace_keeps_only_allowed_https_sources(self):
        response = {
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "type": "search",
                        "query": "user similarity recommender systems ranking paper",
                        "queries": ["multi-view collaborative filtering paper"],
                        "sources": [
                            {"title": "Allowed preprint", "url": "https://arxiv.org/abs/2302.02352"},
                            {"title": "Untrusted blog", "url": "https://example.com/recommenders"},
                            {"title": "Insecure URL", "url": "http://dl.acm.org/doi/10.1/test"},
                        ],
                    },
                },
                {
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "{}",
                        "annotations": [{
                            "type": "url_citation",
                            "title": "Allowed proceedings paper",
                            "url": "https://proceedings.mlr.press/v202/example.html",
                        }],
                    }],
                },
            ],
        }
        sources, queries = OpenAIProvider._research_trace(response)
        self.assertEqual(
            [source["domain"] for source in sources],
            ["arxiv.org", "proceedings.mlr.press"],
        )
        self.assertEqual(len(queries), 2)
        self.assertTrue(all(source["url"].startswith("https://") for source in sources))

    def test_academic_search_is_bounded_and_returns_source_metadata(self):
        options = OpenAIProvider._academic_search_options()
        self.assertEqual(options["max_tool_calls"], 2)
        self.assertEqual(options["tool_choice"], "auto")
        self.assertEqual(options["include"], ["web_search_call.action.sources"])
        self.assertEqual(options["tools"][0]["type"], "web_search")
        self.assertIn("arxiv.org", options["tools"][0]["filters"]["allowed_domains"])

    def test_openai_provider_uses_verified_tls_context(self):
        context = OpenAIProvider._ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_openai_strict_schema_requires_every_object_property(self):
        def assert_strict_objects(node, path="schema"):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    properties = node.get("properties", {})
                    self.assertEqual(
                        set(node.get("required", [])),
                        set(properties),
                        f"{path} must require every declared property",
                    )
                for key, value in node.items():
                    assert_strict_objects(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    assert_strict_objects(value, f"{path}[{index}]")

        assert_strict_objects(OpenAIProvider._proposal_schema())

    def test_multiview_neighbor_scores_ignore_validation_outcomes(self):
        train = [
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 1, 8, 1_000, 9_000.0),
            (20220408, "u1", "v2", "a2", "0", 10_000.0, 0, 8, 2_000, 1_000.0),
            (20220408, "u2", "v1", "a1", "0", 10_000.0, 1, 9, 3_000, 8_000.0),
            (20220408, "u2", "v3", "a3", "1", 20_000.0, 0, 9, 4_000, 2_000.0),
            (20220408, "u3", "v2", "a2", "1", 10_000.0, 0, 10, 5_000, 1_000.0),
            (20220408, "u3", "v3", "a3", "1", 20_000.0, 1, 10, 6_000, 18_000.0),
        ]
        valid = [
            (20220422, "u1", "v3", "a3", "1", 20_000.0, 0, 11, 7_000, 0.0),
            (20220422, "u2", "v2", "a2", "0", 10_000.0, 1, 11, 8_000, 0.0),
            (20220422, "u3", "v1", "a1", "0", 10_000.0, 1, 11, 9_000, 0.0),
        ]
        changed = [tuple([*row[:6], 1 - row[6], *row[7:]]) for row in valid]
        metadata = {
            "v1": {"author_id": "a1", "music_id": "m1", "tag": "dance", "video_type": "NORMAL"},
            "v2": {"author_id": "a2", "music_id": "m1", "tag": "comedy", "video_type": "NORMAL"},
            "v3": {"author_id": "a3", "music_id": "m2", "tag": "dance", "video_type": "AD"},
        }
        parameters = {
            "neighbor_views": ["item", "author", "music", "tag", "video_type"],
            "neighbor_view_weights": [0.4, 0.2, 0.15, 0.15, 0.1],
            "neighbor_profile_mode": "exposure",
            "neighbor_count": 10,
            "neighbor_similarity_power": 2.0,
            "neighbor_idf_power": 0.5,
            "neighbor_min_similarity": 0.0,
            "neighbor_smoothing": 2.0,
            "neighbor_item_smoothing": 4.0,
        }
        scores, diagnostics = build_multiview_neighbor_scores(
            train, valid, Path("."), parameters, video_metadata=metadata,
        )
        changed_scores, _ = build_multiview_neighbor_scores(
            train, changed, Path("."), parameters, video_metadata=metadata,
        )
        np.testing.assert_allclose(scores, changed_scores)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertFalse(diagnostics["validation_outcomes_accessed"])
        self.assertEqual(diagnostics["views"], parameters["neighbor_views"])

    def test_paper_signal_scores_ignore_validation_outcomes(self):
        train = [
            (20220408, "u1", "v1", "a1", "0", 10_000.0, 1, 8, 1_000, 9_000.0),
            (20220408, "u1", "v2", "a2", "0", 20_000.0, 0, 8, 2_000, 2_000.0),
            (20220409, "u2", "v1", "a1", "1", 10_000.0, 1, 9, 3_000, 8_000.0),
            (20220409, "u2", "v3", "a3", "1", 30_000.0, 0, 9, 4_000, 3_000.0),
        ]
        valid = [
            (20220422, "u1", "v1", "a1", "0", 10_000.0, 0, 11, 7_000, 0.0),
            (20220422, "u1", "v3", "a3", "1", 30_000.0, 1, 11, 8_000, 0.0),
            (20220422, "u2", "v2", "a2", "0", 20_000.0, 1, 11, 9_000, 0.0),
        ]
        changed = [tuple([*row[:6], 1 - row[6], *row[7:]]) for row in valid]
        metadata = {
            "v1": {"author_id": "a1", "music_id": "m1", "tag": "dance", "video_type": "NORMAL"},
            "v2": {"author_id": "a2", "music_id": "m1", "tag": "comedy", "video_type": "NORMAL"},
            "v3": {"author_id": "a3", "music_id": "m2", "tag": "dance", "video_type": "AD"},
        }
        parameters = {
            "paper_executor_slug": "paper_affinity_v1",
            "paper_signals": ["user_author_affinity", "tag_prior", "duration_match"],
            "paper_signal_weights": [0.6, 0.25, 0.15],
            "paper_smoothing": 4.0,
            "paper_item_smoothing": 8.0,
            "paper_blend_weight": 0.01,
        }
        scores, diagnostics = build_paper_signal_scores(
            train, valid, Path("."), parameters, video_metadata=metadata,
        )
        changed_scores, _ = build_paper_signal_scores(
            train, changed, Path("."), parameters, video_metadata=metadata,
        )
        np.testing.assert_allclose(scores, changed_scores)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertFalse(diagnostics["validation_outcomes_accessed"])
        self.assertFalse(diagnostics["hidden_test_accessed"])

    def test_executor_incubator_registers_only_exact_audited_program(self):
        paper_url = "https://arxiv.org/abs/2302.02352"
        extension = {
            "requested": True,
            "slug": "paper_affinity_v1",
            "paper_title": "A reviewed recommender systems method",
            "paper_url": paper_url,
            "family": EXECUTOR_FAMILY,
            "method_summary": "Combine smoothed author affinity with duration compatibility ranks.",
            "why_new_executor": "The current executors cannot express this paper-backed signed combination.",
            "signals": ["user_author_affinity", "duration_match"],
            "signal_weights": [0.7, 0.3],
            "smoothing": 8.0,
            "entity_smoothing": 20.0,
            "blend_weight": 0.01,
            "resource_class": "small",
            "required_tests": sorted(REQUIRED_CONTRACT_TESTS),
        }
        sources = [{"title": "Audited paper", "url": paper_url, "domain": "arxiv.org"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review = review_and_register_executor(
                root / "registry", root / "workspace", extension, sources,
            )
            self.assertEqual(review["status"], "approved")
            self.assertTrue(all(review["contract"]["tests"].values()))
            self.assertFalse(review["generated_code_executed"])
            self.assertTrue((root / "workspace" / "executor-incubator" / "paper_affinity_v1" / "manifest.json").exists())
            self.assertEqual(len(load_executor_registry(root / "registry")), 1)
            parameters = dict(review["program"])
            self.assertEqual(
                require_registered_program(root / "registry", parameters)["slug"],
                "paper_affinity_v1",
            )
            parameters["paper_signal_weights"] = [0.6, 0.4]
            with self.assertRaisesRegex(ValueError, "exact reviewed registry program"):
                require_registered_program(root / "registry", parameters)

            unaudited = {**extension, "slug": "unaudited_method", "paper_url": "https://example.com/paper"}
            rejected = review_and_register_executor(
                root / "registry", root / "rejected-workspace", unaudited, sources,
            )
            self.assertEqual(rejected["status"], "rejected")
            self.assertTrue(any("audited HTTPS" in error for error in rejected["errors"]))

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

    def test_cdm_context_rerank_is_outcome_free(self):
        rows = [
            (20220422, "u1", "v1", "a1", "m1", "NORMAL", frozenset({"x"}), 0),
            (20220422, "u1", "v2", "a2", "m2", "NORMAL", frozenset({"x"}), 1),
            (20220422, "u1", "v3", "a3", "m3", "NORMAL", frozenset({"y"}), 0),
        ]
        changed = [tuple(list(row[:-1]) + [1 - row[-1]]) for row in rows]
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
        base, context = prepare_context(rows, ["u1"] * 3, scores)
        changed_base, changed_context = prepare_context(changed, ["u1"] * 3, scores)
        candidate = rerank(base, context, (0.0, 0.0, 0.0, 1.0), 0.2, 3)
        changed_candidate = rerank(
            changed_base, changed_context, (0.0, 0.0, 0.0, 1.0), 0.2, 3
        )
        np.testing.assert_array_equal(candidate, changed_candidate)

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
