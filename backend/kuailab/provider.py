from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Proposal:
    title: str
    experiment_type: str
    hypothesis: str
    rationale: str
    change_summary: str
    code: str
    parameters: dict[str, Any]
    acceptance: str
    abort_condition: str
    expected_gain: float
    usage: dict[str, int]
    response_id: str | None = None


class DemoProvider:
    ideas = [
        ("Calibrated pairwise ranker", "A pairwise objective should better align training with top-k ordering.", {"pairwise_weight": 0.35, "embedding_dim": 32}, 0.0018),
        ("Author-duration affinity", "Crossed user-author and watch-duration features should improve long-tail preference estimates.", {"author_cross": 1, "duration_bins": 8}, 0.0027),
        ("Temporal affinity ranker", "Recency-weighted affinities should adapt to short-term interest drift without erasing stable taste.", {"recency_half_life_days": 14, "history_window": 80}, 0.0031),
        ("Exposure-debiased objective", "Inverse-propensity weighting should reduce popularity bias in implicit feedback.", {"ips_clip": 8, "negative_ratio": 5}, 0.0014),
        ("Multi-task long-view head", "An auxiliary duration head should regularize the long-view target.", {"aux_weight": 0.18, "dropout": 0.12}, 0.0011),
    ]

    def propose(self, context: dict) -> Proposal:
        number = int(context["iteration"])
        title, hypothesis, params, gain = self.ideas[(number - 1) % len(self.ideas)]
        steering = context.get("steering")
        if steering:
            hypothesis = f"{hypothesis} Operator direction: {steering}"
        code = "def configure_experiment(config):\n" + "\n".join(
            f"    config[{key!r}] = {value!r}" for key, value in params.items()
        ) + "\n    return config\n"
        return Proposal(
            title=title,
            experiment_type="fm_config",
            hypothesis=hypothesis,
            rationale="Selected from prior metrics, failure evidence, and remaining compute budget.",
            change_summary=", ".join(f"{key}={value}" for key, value in params.items()),
            code=code,
            parameters=params,
            acceptance=f"Primary improves by at least {context['epsilon']:.4f} or both secondary metrics improve.",
            abort_condition="Invalid metrics, non-zero runner exit, or iteration timeout.",
            expected_gain=gain,
            usage={"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
        )


class OpenAIProvider:
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, model: str, reasoning_effort: str):
        self.model = model
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _output_text(response: dict) -> str:
        if response.get("output_text"):
            return response["output_text"]
        chunks: list[str] = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    chunks.append(content.get("text", ""))
        return "".join(chunks)

    def propose(self, context: dict) -> Proposal:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add a newly rotated key to your shell environment.")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "experiment_type", "hypothesis", "rationale", "change_summary", "code", "parameters", "acceptance", "abort_condition", "expected_gain"],
            "properties": {
                "title": {"type": "string"},
                "experiment_type": {"type": "string", "enum": [
                    "fm_config", "fm_positive_weight", "fm_ensemble", "fm_pairwise", "fm_pairwise_blend",
                    "fm_deep_blend", "fm_temporal_deep_blend", "champion_residual_blend"
                ]},
                "hypothesis": {"type": "string"}, "rationale": {"type": "string"},
                "change_summary": {"type": "string"}, "code": {"type": "string"},
                "parameters": {"type": "object", "additionalProperties": False,
                    "required": [
                        "k", "lr", "epochs", "batch_size", "patience", "seed", "ensemble_seeds", "positive_weight",
                        "pairwise_lr", "pairwise_epochs", "pairwise_patience", "pairwise_seed", "blend_weight",
                        "deep_lr", "deep_epochs", "deep_patience", "deep_seed", "deep_hidden", "deep_dropout",
                        "deep_threads", "deep_blend_weight", "temporal_blend_weight",
                        "champion_candidate_family", "champion_blend_weight"
                    ],
                    "properties": {
                        "k": {"type": "integer"},
                        "lr": {"type": "number"},
                        "epochs": {"type": "integer"},
                        "batch_size": {"type": "integer"},
                        "patience": {"type": "integer"},
                        "seed": {"type": "integer"},
                        "ensemble_seeds": {"type": "array", "items": {"type": "integer"}},
                        "positive_weight": {"type": "number"},
                        "pairwise_lr": {"type": "number"},
                        "pairwise_epochs": {"type": "integer"},
                        "pairwise_patience": {"type": "integer"},
                        "pairwise_seed": {"type": "integer"},
                        "blend_weight": {"type": "number"},
                        "deep_lr": {"type": "number"},
                        "deep_epochs": {"type": "integer"},
                        "deep_patience": {"type": "integer"},
                        "deep_seed": {"type": "integer"},
                        "deep_hidden": {"type": "integer"},
                        "deep_dropout": {"type": "number"},
                        "deep_threads": {"type": "integer"},
                        "deep_blend_weight": {"type": "number"},
                        "temporal_blend_weight": {"type": "number"},
                        "champion_candidate_family": {
                            "type": "string",
                            "enum": ["pointwise_fm", "pairwise_fm", "deepfm_blend", "temporal_deepfm_blend"]
                        },
                        "champion_blend_weight": {"type": "number", "minimum": -0.25, "maximum": 0.25}
                    }},
                "acceptance": {"type": "string"}, "abort_condition": {"type": "string"}, "expected_gain": {"type": "number"},
            },
        }
        body = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "instructions": "You are the experiment-design component of an autonomous recommender-systems researcher. Propose exactly one falsifiable, budget-aware KuaiRand-Pure long_view experiment using the executor contract in the input. Never claim a metric you have not observed and do not repeat an exact tested parameter configuration. Use only the trusted typed executors: NumPy for FM variants, the installed PyTorch executor for deep variants, or champion_residual_blend to train a fresh candidate and blend it into the checksum-verified frozen 0.612858 champion. Prefer champion_residual_blend once the retained champion is above the standalone FM family. Do not install packages or use undefined placeholders. The code field must show the concrete configuration corresponding to the selected typed experiment. Return only schema-valid JSON.",
            "input": json.dumps(context, sort_keys=True),
            "text": {"format": {"type": "json_schema", "name": "experiment_proposal", "strict": True, "schema": schema}},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as result:
                response = json.loads(result.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"OpenAI Responses API failed ({error.code}): {detail}") from error
        payload = json.loads(self._output_text(response))
        raw_usage = response.get("usage", {})
        output_details = raw_usage.get("output_tokens_details", {})
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens", 0)),
            "output_tokens": int(raw_usage.get("output_tokens", 0)),
            "reasoning_tokens": int(output_details.get("reasoning_tokens", 0)),
            "total_tokens": int(raw_usage.get("total_tokens", 0)),
        }
        return Proposal(**payload, usage=usage, response_id=response.get("id"))
