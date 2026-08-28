import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from outputs.trusted_components import fit_predict_fm, paired_fm_predictions
except ModuleNotFoundError:  # lightweight controller environments need not install the Kaggle data stack
    np = pd = fit_predict_fm = paired_fm_predictions = None
from research_agent.autonomous import GenericResearchAgent, ScriptedValidationModel
from research_agent.benchmark import ToyRankingBenchmark, evaluate_ranking
from research_agent.core import LiteratureIndex
from research_agent.safety import CodeSafetyGate
from research_agent.real_pilot import (
    deterministic_findings, feasibility_findings, kuairand_context,
    normalize_requirements, trusted_api_findings,
)


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

    def test_safety_gate_allows_boolean_array_negation(self):
        source = """import numpy as np
def main():
    mask = np.array([True, False])
    inverse = ~mask
    print(inverse)
"""
        result = CodeSafetyGate({"numpy"}).inspect(source)
        self.assertTrue(result["passed"], result["findings"])

    def test_deterministic_preflight_rejects_syntax_and_missing_output_contract(self):
        findings = deterministic_findings("def main(:\n    pass", {"title": "Tree ranker", "hypothesis": "rank", "model_family": "tree"})
        self.assertTrue(any("Syntax" in item or "syntax" in item for item in findings))
        self.assertTrue(any("predictions.npy" in item for item in findings))

    def test_deterministic_preflight_accepts_trusted_paired_fm_output(self):
        source = """from trusted_components import paired_fm_predictions, save_predictions
def main():
    _, scores = paired_fm_predictions(train, validation, control, treatment)
    save_predictions(scores, validation)
if __name__ == '__main__':
    main()
"""
        proposal = {"title": "Paired FM feature ablation", "hypothesis": "FM feature improves ranking", "model_family": "factorization_machine"}
        self.assertEqual(deterministic_findings(source, proposal), [])

    def test_feasibility_gate_rejects_unavailable_fm_score_artifact(self):
        context = kuairand_context([])
        proposal = {
            "title": "FM residual reranker", "hypothesis": "Learn a residual over FM scores",
            "change_kind": "hybrid", "required_inputs": ["data/baseline_fm_validation.npy"],
        }
        findings = feasibility_findings(proposal, context)
        self.assertTrue(any("Unavailable required input" in item for item in findings))
        self.assertTrue(any("FM residual" in item for item in findings))

    def test_feasibility_gate_allows_residual_that_trains_its_own_fm(self):
        context = kuairand_context([])
        proposal = {
            "title": "FM residual reranker",
            "hypothesis": "Train an FM backbone and learn a bounded residual",
            "change_kind": "hybrid",
            "required_inputs": ["data/train.parquet", "data/validation.parquet"],
            "required_capabilities": ["trusted_components.fit_predict_fm", "lightgbm.LGBMRanker"],
        }
        self.assertEqual(feasibility_findings(proposal, context), [])

    def test_requirements_separate_files_from_trusted_capabilities(self):
        context = kuairand_context([])
        proposal = {
            "required_inputs": [
                "data/train.parquet", "data/validation.parquet",
                "trusted_components.fit_predict_fm(train, validation, columns)",
            ]
        }
        normalize_requirements(proposal, context)
        self.assertEqual(proposal["required_inputs"], ["data/train.parquet", "data/validation.parquet"])
        self.assertEqual(len(proposal["required_capabilities"]), 1)
        self.assertEqual(feasibility_findings(proposal, context), [])

    def test_trusted_api_gate_rejects_unknown_exports_and_constructor_keywords(self):
        source = """import trusted_components as tc
def main():
    tc.made_up_helper()
    tc.TrustedFM(dimension=10, n_factors=8)
"""
        findings = trusted_api_findings(source)
        self.assertTrue(any("Unknown trusted_components export" in item for item in findings))
        self.assertTrue(any("Unsupported TrustedFM constructor" in item for item in findings))

    @unittest.skipIf(fit_predict_fm is None, "Kaggle numerical stack is not installed locally")
    def test_high_level_fm_primitives_fit_and_return_aligned_scores(self):
        train = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u2"] * 16,
            "video_id": ["v1", "v2", "v1", "v3"] * 16,
            "hour": [1, 2, 1, 3] * 16,
            "long_view": [1, 0, 0, 1] * 16,
        })
        validation = pd.DataFrame({"row_id": np.arange(4), "user_id": ["u1", "u2", "u1", "u2"], "video_id": ["v2", "v3", "v1", "v1"], "hour": [2, 3, 1, 1]})
        scores = fit_predict_fm(train, validation, ["user_id", "video_id"], epochs=1, batch_size=128)
        control, treatment = paired_fm_predictions(
            train, validation, ["user_id", "video_id"], ["user_id", "video_id", "hour"], epochs=1, batch_size=128
        )
        self.assertEqual(len(scores), len(validation))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(len(control), len(treatment))

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
