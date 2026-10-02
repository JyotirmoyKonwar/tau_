"""Deterministic task analysis and candidate filtering for the optional router."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tau_agent.messages import AgentMessage, ImageContent, UserMessage
from tau_agent.tools import AgentTool
from tau_coding.context_window import estimate_context_usage
from tau_coding.provider_config import ProviderSettings, provider_model_supports_images
from tau_coding.router_config import RouterCandidate, RouterTier

TaskType = Literal[
    "explanation", "debugging", "tests", "code_generation", "refactoring", "architecture", "unknown"
]
Complexity = Literal["easy", "medium", "hard"]

_TASK_PATTERNS: tuple[tuple[TaskType, str], ...] = (
    ("architecture", r"\b(?:architect(?:ure)?|design|system design)\b"),
    ("debugging", r"\b(?:debug|fix|bug|error|crash|deadlock|race condition|failing)\b"),
    ("tests", r"\b(?:test|tests|pytest|unittest|coverage)\b"),
    ("refactoring", r"\b(?:refactor|restructure|simplify|cleanup)\b"),
    ("code_generation", r"\b(?:implement|create|build|add|write|generate)\b"),
    ("explanation", r"\b(?:explain|describe|what|why|how|summarize)\b"),
)
_HARD_PATTERN = re.compile(
    r"\b(?:deadlocks?|race condition|concurrency|distributed|migration)\b", re.I
)
_MEDIUM_PATTERN = re.compile(r"\b(?:debug|fix|implement|refactor|multiple files?)\b", re.I)


@dataclass(frozen=True, slots=True)
class TaskFeatures:
    task_type: TaskType
    complexity: Complexity
    prompt_length: int
    context_tokens: int
    required_capabilities: frozenset[str]


@dataclass(frozen=True, slots=True)
class CandidateFilterResult:
    eligible: tuple[RouterCandidate, ...]
    rejected: dict[RouterTier, str]


def analyze_turn(
    prompt: str,
    *,
    system: str = "",
    messages: tuple[AgentMessage, ...] = (),
    tools: tuple[AgentTool, ...] = (),
    has_image: bool = False,
) -> TaskFeatures:
    """Analyze a new prompt with the active context *before* that prompt is appended."""
    lower = prompt.lower()
    task_type: TaskType = "unknown"
    for candidate_type, pattern in _TASK_PATTERNS:
        if re.search(pattern, lower):
            task_type = candidate_type
            break

    context_tokens = estimate_context_usage(
        system=system, messages=(*messages, UserMessage(content=prompt)), tools=tools
    ).total_tokens
    if (
        len(prompt) >= 1200
        or context_tokens >= 16_000
        or _HARD_PATTERN.search(prompt)
        or task_type == "architecture"
    ):
        complexity: Complexity = "hard"
    elif (
        len(prompt) >= 300
        or _MEDIUM_PATTERN.search(prompt)
        or task_type in {"tests", "refactoring"}
    ):
        complexity = "medium"
    else:
        complexity = "easy"

    needs_image = has_image or any(
        isinstance(message, UserMessage)
        and isinstance(message.content, list)
        and any(isinstance(block, ImageContent) for block in message.content)
        for message in messages
    )
    capabilities = frozenset({"text", "image"} if needs_image else {"text"})
    return TaskFeatures(task_type, complexity, len(prompt), context_tokens, capabilities)


def filter_router_candidates(
    features: TaskFeatures,
    candidates: Sequence[RouterCandidate],
    settings: ProviderSettings,
) -> CandidateFilterResult:
    """Keep candidates known to fit context and required input modalities."""
    eligible: list[RouterCandidate] = []
    rejected: dict[RouterTier, str] = {}
    for candidate in candidates:
        if (
            candidate.context_window is not None
            and features.context_tokens >= candidate.context_window
        ):
            rejected[candidate.tier] = "context window too small"
            continue
        provider = next(
            (item for item in settings.providers if item.name == candidate.provider), None
        )
        if "image" in features.required_capabilities and (
            provider is None or not provider_model_supports_images(provider, candidate.model)
        ):
            rejected[candidate.tier] = "image input unavailable"
            continue
        eligible.append(candidate)
    return CandidateFilterResult(tuple(eligible), rejected)
