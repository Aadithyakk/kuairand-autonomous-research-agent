import json
import tempfile
import unittest
from pathlib import Path

from research_agent.autonomous import GenericResearchAgent, ScriptedValidationModel
from research_agent.benchmark import ToyRankingBenchmark, evaluate_ranking
from research_agent.core import LiteratureIndex
from research_agent.safety import CodeSafetyGate


class AutonomousAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        cards = [
            {
                "id": "ranking-evidence", "title": "Personalized ranking evidence", "year": 2025,
                "url": "https://example.test", "tags": ["ranking", "user", "affinity"],
                "claim": "User-conditioned affinity can rank unseen items.", "cautions": ["Keep histories causal."],
            }
        ]
        literature_path = self.root / "literature.json"
        literature_path.write_text(json.dumps(cards), encoding="utf-8")
        self.literature = LiteratureIndex.from_path(literature_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_ranking_evaluator_handles_ties(self):
        rows = [
            {"row_id": 0, "user_id": "u", "label": 0},
            {"row_id": 1, "user_id": "u", "label": 1},
            {"row_id": 2, "user_id": "u", "label": 0},
            {"row_id": 3, "user_id": "u", "label": 1},
        ]
        metrics = evaluate_ranking(rows, [0.5, 0.5, 0.5, 0.5], k=5)
        self.assertEqual(metrics["GAUC"], 0.5)
        self.assertGreater(metrics["nDCG@5"], 0)

    def test_safety_gate_rejects_network_and_private_labels(self):
        source = """import urllib.request
def main():
    open('../private/validation_labels.json').read()
"""
        result = CodeSafetyGate().inspect(source)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Import is not allowed" in finding for finding in result["findings"]))
        self.assertTrue(any("validation_labels" in finding for finding in result["findings"]))

    def test_safety_gate_rejects_absolute_paths(self):
        source = """from pathlib import Path
def main():
    Path('/tmp/outside.json').write_text('x')
"""
        result = CodeSafetyGate().inspect(source)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Absolute" in finding for finding in result["findings"]))

    def test_agent_generates_recovers_evaluates_and_promotes(self):
        benchmark = ToyRankingBenchmark(self.root / "benchmark")
        agent = GenericResearchAgent(
            benchmark=benchmark,
            research_model=ScriptedValidationModel(),
            literature=self.literature,
            workspace=self.root / "agent",
            max_experiments=2,
            budget_seconds=30,
        )
        state = agent.run(force=True)
        self.assertEqual(state["run"]["status"], "completed")
        self.assertEqual(len(state["experiments"]), 2)
        self.assertEqual(state["experiments"][0]["stage"], "safety")
        self.assertEqual(state["experiments"][0]["status"], "failed")
        self.assertEqual(state["experiments"][1]["status"], "completed")
        self.assertGreater(state["best"]["metrics"]["primary"], state["baseline"]["metrics"]["primary"])
        self.assertEqual(state["best"]["experiment_id"], "iteration-002")
        self.assertEqual(len(state["memory"]), 2)
        self.assertEqual(state["run"]["manual_interventions"], 0)
        exposed_files = list((self.root / "agent/experiments").rglob("*"))
        self.assertFalse(any("validation_labels" in str(path) for path in exposed_files))


if __name__ == "__main__":
    unittest.main()
