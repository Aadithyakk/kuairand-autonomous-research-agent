import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from research_agent.core import LLMClient, LiteratureIndex, ResearchController
from research_agent.kaggle_packager import package


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


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

    def test_run_accepts_time_and_compute_limits(self):
        controller = ResearchController(self.root, self.config)
        controller.initialize()
        state = controller.start({"budget_minutes": 2, "compute_profile_id": "local-cpu"})
        self.assertEqual(state["run"]["budget_seconds"], 120)
        self.assertEqual(state["run"]["compute_profile_id"], "local-cpu")
        self.assertTrue(state["run"]["compute"]["available"])
        controller.stop()

    @patch.dict("os.environ", {"TEST_OPENAI_KEY": "transient"})
    @patch("urllib.request.urlopen")
    def test_responses_client_parses_json_without_persisting_key(self, mocked_open):
        mocked_open.return_value = FakeHTTPResponse({
            "output": [{"content": [{"type": "output_text", "text": "{\"status\": \"online\"}"}]}]
        })
        client = LLMClient({
            "mode": "openai_responses",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-5.6-luna",
            "api_key_env": "TEST_OPENAI_KEY",
            "reasoning_effort": "medium",
        })
        self.assertEqual(client.complete_json("Return JSON", "probe"), {"status": "online"})
        request = mocked_open.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["model"], "gpt-5.6-luna")
        self.assertNotIn("transient", request.data.decode("utf-8"))

    def test_kaggle_packager_resolves_all_markers_without_credentials(self):
        template = self.root / "worker.py"
        candidate = self.root / "candidate.py"
        proposal = self.root / "proposal.json"
        destination = self.root / "rendered.py"
        template.write_text(
            'import json\nSOURCE = "__CANDIDATE_SOURCE_REPR__"\nPROPOSAL = json.loads("__PROPOSAL_REPR__")\n',
            encoding="utf-8",
        )
        candidate.write_text("def main():\n    return 'safe'\n", encoding="utf-8")
        proposal.write_text(json.dumps({"id": "candidate-1"}), encoding="utf-8")
        package(template, candidate, proposal, destination)
        rendered = destination.read_text(encoding="utf-8")
        self.assertNotIn("__CANDIDATE_SOURCE_REPR__", rendered)
        self.assertNotIn("__PROPOSAL_REPR__", rendered)
        self.assertNotIn("OPENAI_API_KEY", rendered)
        compile(rendered, str(destination), "exec")


if __name__ == "__main__":
    unittest.main()
