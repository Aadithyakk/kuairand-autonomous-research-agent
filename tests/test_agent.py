import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from research_agent.core import LiteratureIndex, ResearchController


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "knowledge").mkdir()
        (self.root / "runtime").mkdir()
        literature = [
            {"id": "ranking", "title": "Ranking paper", "year": 2024, "url": "https://example.test", "tags": ["ranking", "bpr"], "claim": "pairwise ranking", "cautions": []}
        ]
        actions = [
            {"id": "quick_ranker", "title": "Quick ranker", "family": "tree_ranker", "hypothesis": "Ranking alignment helps.", "when": ["ranking metric"], "evidence_tags": ["ranking"], "estimated_minutes": 1, "risk": 0.1, "prior_gain": 0.01, "resources": "cpu", "command": None},
            {"id": "pairwise", "title": "Pairwise model", "family": "pairwise_cf", "hypothesis": "Pairwise learning helps.", "when": ["ranking metric"], "evidence_tags": ["bpr"], "estimated_minutes": 1, "risk": 0.2, "prior_gain": 0.008, "resources": "cpu", "command": None},
        ]
        (self.root / "knowledge/literature.json").write_text(json.dumps(literature))
        (self.root / "knowledge/actions.json").write_text(json.dumps(actions))
        artifact = {"status": "passed", "validation_mean": {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.60155}}
        baseline_zip = self.root / "baseline.zip"
        with zipfile.ZipFile(baseline_zip, "w") as archive:
            archive.writestr("iteration_000_baseline.json", json.dumps(artifact))
        self.config = {
            "benchmark": "KuaiRand-Pure", "label": "long_view", "metrics": ["GAUC", "nDCG@5"], "primary_metric": "primary",
            "max_experiments": 2, "budget_seconds": 300, "convergence_epsilon": 0.002, "convergence_patience": 3,
            "executor_mode": "simulation", "random_seed": 9,
            "llm": {"mode": "fallback", "base_url": "http://127.0.0.1:1/v1", "model": "none", "api_key_env": "NONE"},
            "paths": {"state": "runtime/state.json", "events": "runtime/events.jsonl", "literature": "knowledge/literature.json", "actions": "knowledge/actions.json", "baseline_artifact": str(baseline_zip), "workspace": "runtime/experiments"},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_literature_search(self):
        index = LiteratureIndex.from_path(self.root / "knowledge/literature.json")
        self.assertEqual(index.search("BPR pairwise ranking")[0]["id"], "ranking")

    def test_end_to_end_simulated_loop(self):
        controller = ResearchController(self.root, self.config)
        state = controller.initialize()
        self.assertAlmostEqual(state["best"]["metrics"]["primary"], 0.60155)
        controller.start()
        deadline = time.time() + 5
        while time.time() < deadline:
            state = controller.store.load()
            if state["run"]["status"] in {"completed", "converged", "error"}:
                break
            time.sleep(0.05)
        self.assertEqual(state["run"]["status"], "completed")
        self.assertEqual(len(state["experiments"]), 2)
        self.assertTrue(state["decisions"])
        self.assertTrue(any(event["kind"] == "result" for event in state["events"]))

    def test_steering_is_audited(self):
        controller = ResearchController(self.root, self.config)
        controller.initialize()
        state = controller.steer("Prioritize ranking experiments under ten minutes")
        self.assertEqual(state["run"]["manual_interventions"], 1)
        self.assertEqual(state["steering"][0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
