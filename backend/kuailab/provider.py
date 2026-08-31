from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import certifi


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
    strategy: str = "exploit"
    operator: str = "improve"
    component: str = "model"
    parent_iteration: int = 0
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    response_id: str | None = None
    research_sources: list[dict[str, str]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    web_search_used: bool = False
    executor_extension: dict[str, Any] = field(default_factory=dict)


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
            strategy=("exploit", "explore", "innovate")[(number - 1) % 3],
            operator="improve",
            component="model",
            parent_iteration=max(0, number - 1),
            alternatives=[
                {"strategy": "exploit", "title": "Refine current best", "operator": "improve", "component": "model", "rationale": "Exploit the strongest branch.", "expected_gain": 0.001},
                {"strategy": "explore", "title": "Try a distinct family", "operator": "draft", "component": "model", "rationale": "Preserve search diversity.", "expected_gain": 0.001},
                {"strategy": "innovate", "title": "Test a new signal", "operator": "ablate", "component": "reward", "rationale": "Probe an unattempted mechanism.", "expected_gain": 0.001},
            ],
        )


class OpenAIProvider:
    endpoint = "https://api.openai.com/v1/responses"
    academic_domains = (
        "arxiv.org",
        "dl.acm.org",
        "ieeexplore.ieee.org",
        "openreview.net",
        "proceedings.mlr.press",
        "proceedings.neurips.cc",
        "aclanthology.org",
        "link.springer.com",
    )

    def __init__(self, model: str, reasoning_effort: str, academic_search_enabled: bool = True):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.academic_search_enabled = academic_search_enabled

    @staticmethod
    def _ssl_context() -> ssl.SSLContext:
        """Build a verified TLS context, respecting an operator-supplied CA bundle."""
        cafile = os.getenv("SSL_CERT_FILE") or certifi.where()
        return ssl.create_default_context(cafile=cafile)

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

    @classmethod
    def _research_trace(cls, response: dict) -> tuple[list[dict[str, str]], list[str]]:
        """Return a compact, domain-checked audit trail for hosted web searches."""
        raw_sources: list[dict[str, Any]] = []
        queries: list[str] = []
        for item in response.get("output", []):
            if item.get("type") == "web_search_call":
                action = item.get("action") or {}
                query = action.get("query")
                if isinstance(query, str) and query.strip():
                    queries.append(query.strip())
                for candidate in action.get("queries", []):
                    if isinstance(candidate, str) and candidate.strip():
                        queries.append(candidate.strip())
                if isinstance(action.get("sources"), list):
                    raw_sources.extend(source for source in action["sources"] if isinstance(source, dict))
            if item.get("type") == "message":
                for content in item.get("content", []):
                    for annotation in content.get("annotations", []):
                        if annotation.get("type") == "url_citation":
                            raw_sources.append(annotation)

        sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for source in raw_sources:
            url = source.get("url")
            if not isinstance(url, str) or url in seen_urls:
                continue
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not any(
                host == domain or host.endswith(f".{domain}") for domain in cls.academic_domains
            ):
                continue
            seen_urls.add(url)
            title = source.get("title")
            sources.append({
                "title": title.strip() if isinstance(title, str) and title.strip() else host,
                "url": url,
                "domain": host,
            })
            if len(sources) >= 12:
                break
        return sources, list(dict.fromkeys(queries))[:8]

    @classmethod
    def _academic_search_options(cls) -> dict[str, Any]:
        return {
            "tools": [{
                "type": "web_search",
                "filters": {"allowed_domains": list(cls.academic_domains)},
            }],
            "tool_choice": "auto",
            "max_tool_calls": 2,
            "include": ["web_search_call.action.sources"],
        }

    @staticmethod
    def _proposal_schema() -> dict[str, Any]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "experiment_type", "hypothesis", "rationale", "change_summary", "code", "parameters", "acceptance", "abort_condition", "expected_gain", "strategy", "operator", "component", "parent_iteration", "alternatives", "executor_extension"],
            "properties": {
                "title": {"type": "string"},
                "experiment_type": {"type": "string", "enum": [
                    "fm_config", "fm_positive_weight", "fm_ensemble", "fm_pairwise", "fm_pairwise_blend",
                    "fm_deep_blend", "fm_temporal_deep_blend", "champion_residual_blend",
                    "multiview_neighbor_residual", "executor_incubation", "paper_signal_residual"
                ]},
                "hypothesis": {"type": "string"}, "rationale": {"type": "string"},
                "change_summary": {"type": "string"}, "code": {"type": "string"},
                "parameters": {"type": "object", "additionalProperties": False,
                    "required": [
                        "k", "lr", "epochs", "batch_size", "patience", "seed", "ensemble_seeds", "positive_weight",
                        "pairwise_lr", "pairwise_epochs", "pairwise_patience", "pairwise_seed", "blend_weight",
                        "deep_lr", "deep_epochs", "deep_patience", "deep_seed", "deep_hidden", "deep_dropout",
                        "deep_threads", "deep_blend_weight", "temporal_blend_weight",
                        "recency_half_life_days", "rad_aux_weight", "rad_score_weight", "ordinal_aux_weight",
                        "gauc_pair_weight", "champion_candidate_family", "champion_blend_weight",
                        "neighbor_views", "neighbor_view_weights", "neighbor_profile_mode", "neighbor_count",
                        "neighbor_similarity_power", "neighbor_idf_power", "neighbor_min_similarity",
                        "neighbor_smoothing", "neighbor_item_smoothing", "neighbor_blend_weight",
                        "paper_executor_slug", "paper_signals", "paper_signal_weights",
                        "paper_smoothing", "paper_item_smoothing", "paper_blend_weight"
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
                        "recency_half_life_days": {"type": "number", "minimum": 0, "maximum": 60},
                        "rad_aux_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "rad_score_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "ordinal_aux_weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "gauc_pair_weight": {"type": "number", "minimum": 0, "maximum": 0.5},
                        "champion_candidate_family": {
                            "type": "string",
                            "enum": [
                                "pointwise_fm", "pairwise_fm", "deepfm_blend",
                                "temporal_deepfm_blend", "slate_context_deepfm", "rad_deepfm",
                                "ordinal_watch_deepfm", "profile_deepfm", "gauc_deepfm"
                            ]
                        },
                        "champion_blend_weight": {"type": "number", "minimum": -0.25, "maximum": 0.25},
                        "neighbor_views": {
                            "type": "array", "minItems": 1, "maxItems": 5,
                            "items": {"type": "string", "enum": ["item", "author", "music", "tag", "video_type"]}
                        },
                        "neighbor_view_weights": {
                            "type": "array", "minItems": 1, "maxItems": 5,
                            "items": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "neighbor_profile_mode": {"type": "string", "enum": ["exposure", "positive", "signed"]},
                        "neighbor_count": {"type": "integer", "minimum": 10, "maximum": 120},
                        "neighbor_similarity_power": {"type": "number", "minimum": 0.5, "maximum": 4},
                        "neighbor_idf_power": {"type": "number", "minimum": 0, "maximum": 1},
                        "neighbor_min_similarity": {"type": "number", "minimum": 0, "maximum": 0.95},
                        "neighbor_smoothing": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "neighbor_item_smoothing": {"type": "number", "minimum": 0.1, "maximum": 200},
                        "neighbor_blend_weight": {"type": "number", "minimum": -0.25, "maximum": 0.25},
                        "paper_executor_slug": {"type": "string", "pattern": "^$|^[a-z][a-z0-9_]{2,47}$"},
                        "paper_signals": {
                            "type": "array", "minItems": 0, "maxItems": 6,
                            "items": {"type": "string", "enum": [
                                "user_item_affinity", "user_author_affinity", "user_music_affinity",
                                "user_tag_affinity", "user_video_type_affinity", "item_prior",
                                "author_prior", "music_prior", "tag_prior", "video_type_prior",
                                "repeat_penalty", "slate_author_frequency", "slate_music_frequency",
                                "duration_match"
                            ]}
                        },
                        "paper_signal_weights": {
                            "type": "array", "minItems": 0, "maxItems": 6,
                            "items": {"type": "number", "minimum": -1, "maximum": 1}
                        },
                        "paper_smoothing": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "paper_item_smoothing": {"type": "number", "minimum": 0.1, "maximum": 200},
                        "paper_blend_weight": {"type": "number", "minimum": -0.25, "maximum": 0.25}
                    }},
                "acceptance": {"type": "string"}, "abort_condition": {"type": "string"}, "expected_gain": {"type": "number"},
                "strategy": {"type": "string", "enum": ["exploit", "explore", "innovate"]},
                "operator": {"type": "string", "enum": ["draft", "improve", "ablate", "ensemble", "debug"]},
                "component": {"type": "string", "enum": ["features", "model", "loss", "training", "ensemble", "reward"]},
                "parent_iteration": {"type": "integer", "minimum": 0},
                "alternatives": {
                    "type": "array", "minItems": 3, "maxItems": 3,
                    "items": {
                        "type": "object", "additionalProperties": False,
                        "required": ["strategy", "title", "operator", "component", "rationale", "expected_gain"],
                        "properties": {
                            "strategy": {"type": "string", "enum": ["exploit", "explore", "innovate"]},
                            "title": {"type": "string"},
                            "operator": {"type": "string", "enum": ["draft", "improve", "ablate", "ensemble", "debug"]},
                            "component": {"type": "string", "enum": ["features", "model", "loss", "training", "ensemble", "reward"]},
                            "rationale": {"type": "string"},
                            "expected_gain": {"type": "number"}
                        }
                    }
                },
                "executor_extension": {
                    "type": "object", "additionalProperties": False,
                    "required": [
                        "requested", "slug", "paper_title", "paper_url", "family",
                        "method_summary", "why_new_executor", "signals", "signal_weights",
                        "smoothing", "entity_smoothing", "blend_weight", "resource_class",
                        "required_tests"
                    ],
                    "properties": {
                        "requested": {"type": "boolean"},
                        "slug": {"type": "string", "pattern": "^$|^[a-z][a-z0-9_]{2,47}$"},
                        "paper_title": {"type": "string"},
                        "paper_url": {"type": "string"},
                        "family": {"type": "string", "enum": ["declarative_signal_reranker_v1"]},
                        "method_summary": {"type": "string"},
                        "why_new_executor": {"type": "string"},
                        "signals": {
                            "type": "array", "minItems": 0, "maxItems": 6,
                            "items": {"type": "string", "enum": [
                                "user_item_affinity", "user_author_affinity", "user_music_affinity",
                                "user_tag_affinity", "user_video_type_affinity", "item_prior",
                                "author_prior", "music_prior", "tag_prior", "video_type_prior",
                                "repeat_penalty", "slate_author_frequency", "slate_music_frequency",
                                "duration_match"
                            ]}
                        },
                        "signal_weights": {
                            "type": "array", "minItems": 0, "maxItems": 6,
                            "items": {"type": "number", "minimum": -1, "maximum": 1}
                        },
                        "smoothing": {"type": "number", "minimum": 0.1, "maximum": 100},
                        "entity_smoothing": {"type": "number", "minimum": 0.1, "maximum": 200},
                        "blend_weight": {"type": "number", "minimum": -0.25, "maximum": 0.25},
                        "resource_class": {"type": "string", "enum": ["small", "medium", "large"]},
                        "required_tests": {
                            "type": "array", "minItems": 0, "maxItems": 6,
                            "items": {"type": "string", "enum": [
                                "validation_label_invariance", "temporal_fit_boundary",
                                "finite_output", "deterministic_output", "output_shape",
                                "resource_budget"
                            ]}
                        }
                    }
                }
            },
        }
        return schema

    def propose(self, context: dict) -> Proposal:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add a newly rotated key to your shell environment.")
        schema = self._proposal_schema()
        body = {
            "model": self.model,
            "reasoning": {"effort": self.reasoning_effort},
            "instructions": "You are the experiment-design component of an autonomous recommender-systems researcher. Optimize the strongest frozen, submission-safe KuaiRand-Pure long_view ranker. First produce exactly three distinct alternatives: one exploit, one explore, and one innovate. Then select exactly one as the returned proposal. Anchor it to one parent_iteration in the supplied search_tree and apply one atomic operator to one component. Use the supplied research_priors as directional evidence: translate their transferable mechanisms into outcome-free static approximations, but never claim an online/prequential metric as a static result and never use target-period outcomes at prediction time. Use ablation evidence and the method-card case library; never repeat an exhausted card or exact tested configuration. When academic web search is available, use it only when the supplied evidence has a genuine knowledge gap or the search has plateaued. Prefer primary papers, inspect the actual paper rather than relying on snippets, and use paper claims only as hypotheses. Never present a published metric as an observed KuaiRand result. Put paper title and URL in the rationale when a paper materially motivates the selected experiment. Propose a falsifiable, budget-aware experiment using the executor contract. Never claim a metric you have not observed. Use the trusted typed executors whenever they can express the hypothesis. If a materially useful paper mechanism cannot be expressed, you may select executor_incubation and request exactly one declarative_signal_reranker_v1 extension composed only of the allowed signals. Incubation does not train or report a score; it scaffolds and contract-tests the program. Only a later iteration may select paper_signal_residual, and its parameters must exactly copy an entry in approved_executor_extensions. For every non-incubation proposal set executor_extension.requested=false and use empty descriptive strings/arrays with safe numeric defaults. Prefer conservative zero-preferring residual weights. Do not install packages, emit executable model code, or use undefined placeholders. The code field must show the concrete typed configuration. Return only schema-valid JSON.",
            "input": json.dumps(context, sort_keys=True),
            "text": {"format": {"type": "json_schema", "name": "experiment_proposal", "strict": True, "schema": schema}},
        }
        if self.academic_search_enabled:
            body.update(self._academic_search_options())
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180, context=self._ssl_context()) as result:
                response = json.loads(result.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:800]
            raise RuntimeError(f"OpenAI Responses API failed ({error.code}): {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"OpenAI Responses API secure connection failed: {error.reason}") from error
        payload = json.loads(self._output_text(response))
        alternative_strategies = [item.get("strategy") for item in payload.get("alternatives", [])]
        if sorted(alternative_strategies) != ["exploit", "explore", "innovate"]:
            raise RuntimeError("Planner must return one exploit, one explore, and one innovate alternative")
        if payload.get("strategy") not in alternative_strategies:
            raise RuntimeError("Selected strategy must match one proposed alternative")
        matching = [
            item for item in payload["alternatives"]
            if item["strategy"] == payload["strategy"]
        ]
        if not matching or matching[0]["operator"] != payload.get("operator") or matching[0]["component"] != payload.get("component"):
            raise RuntimeError("Selected operator and component must match the chosen alternative")
        extension = payload.get("executor_extension", {})
        incubating = payload.get("experiment_type") == "executor_incubation"
        if bool(extension.get("requested")) != incubating:
            raise RuntimeError(
                "executor_extension.requested must be true only for executor_incubation"
            )
        if payload.get("experiment_type") == "paper_signal_residual":
            parameters = payload.get("parameters", {})
            if not parameters.get("paper_executor_slug") or not parameters.get("paper_signals"):
                raise RuntimeError(
                    "paper_signal_residual must name and copy an approved executor program"
                )
        valid_parents = {int(item["node"]) for item in context.get("search_tree", [])}
        if valid_parents and int(payload.get("parent_iteration", -1)) not in valid_parents:
            raise RuntimeError("parent_iteration must reference an existing search-tree node")
        raw_usage = response.get("usage", {})
        output_details = raw_usage.get("output_tokens_details", {})
        usage = {
            "input_tokens": int(raw_usage.get("input_tokens", 0)),
            "output_tokens": int(raw_usage.get("output_tokens", 0)),
            "reasoning_tokens": int(output_details.get("reasoning_tokens", 0)),
            "total_tokens": int(raw_usage.get("total_tokens", 0)),
        }
        sources, queries = self._research_trace(response)
        return Proposal(
            **payload,
            usage=usage,
            response_id=response.get("id"),
            research_sources=sources,
            search_queries=queries,
            web_search_used=bool(queries or sources),
        )
