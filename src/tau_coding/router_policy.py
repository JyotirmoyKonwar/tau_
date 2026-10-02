"""Deterministic routing policies and scoring for three configured candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from tau_coding.router_config import RouterCandidate, RouterTier
from tau_coding.router_config import RouterPolicy as RouterMode
from tau_coding.router_features import Complexity, TaskFeatures, TaskType

_TIER_ORDER: tuple[RouterTier, ...] = ("small", "medium", "strong")
_SUCCESS_PRIORS: dict[Complexity, tuple[float, float, float]] = {
    "easy": (0.92, 0.96, 0.98),
    "medium": (0.65, 0.83, 0.93),
    "hard": (0.35, 0.65, 0.88),
}
_TASK_ADJUSTMENTS: dict[TaskType, float] = {
    "explanation": 0.03,
    "debugging": -0.04,
    "tests": 0.0,
    "code_generation": -0.02,
    "refactoring": -0.01,
    "architecture": -0.05,
    "unknown": 0.0,
}
_LATENCY_PROXIES: dict[RouterTier, float] = {"small": 1.0, "medium": 2.0, "strong": 3.0}
_ESTIMATED_OUTPUT_TOKENS = 1024


class RoutingError(ValueError):
    """A route cannot be selected from the supplied candidates or estimates."""


@dataclass(frozen=True, slots=True)
class CandidateEstimate:
    predicted_success: float
    estimated_cost: float | None
    estimated_latency: float | None


@dataclass(frozen=True, slots=True)
class RouterWeights:
    success: float = 1.0
    cost: float = 0.15
    latency: float = 0.05


_DEFAULT_WEIGHTS = RouterWeights()


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    tier: RouterTier
    provider: str
    model: str
    mode: RouterMode
    reason: str
    predicted_success: float


RouterStats = Mapping[RouterTier, CandidateEstimate]


class RoutingPolicy(Protocol):
    def select(
        self,
        features: TaskFeatures,
        candidates: Sequence[RouterCandidate],
        *,
        mode: RouterMode,
        stats: RouterStats,
        weights: RouterWeights = _DEFAULT_WEIGHTS,
        economy_margin: float = 0.05,
    ) -> RoutingDecision: ...


def prior_probability(features: TaskFeatures, tier: RouterTier) -> float:
    """Return a fixed, hand-tuned baseline prior; these are not measured results."""
    base = _SUCCESS_PRIORS[features.complexity][_TIER_ORDER.index(tier)]
    return min(0.99, max(0.01, base + _TASK_ADJUSTMENTS[features.task_type]))


def choose_candidate(
    candidates: Sequence[RouterCandidate],
    estimates: RouterStats,
    *,
    mode: RouterMode,
    weights: RouterWeights = _DEFAULT_WEIGHTS,
    economy_margin: float = 0.05,
) -> RouterCandidate:
    """Apply quality, economy, or balanced scoring to supplied model estimates."""
    if not candidates:
        raise RoutingError("No eligible router candidates")
    if not math.isfinite(economy_margin) or not 0 <= economy_margin <= 1:
        raise RoutingError("economy margin must be between 0 and 1")
    if any(
        not math.isfinite(value) or value < 0
        for value in (weights.success, weights.cost, weights.latency)
    ):
        raise RoutingError("router weights must be finite non-negative numbers")
    for candidate in candidates:
        estimate = estimates.get(candidate.tier)
        if estimate is None:
            raise RoutingError(f"Missing estimate for {candidate.tier}")
        if (
            not math.isfinite(estimate.predicted_success)
            or not 0 <= estimate.predicted_success <= 1
        ):
            raise RoutingError(f"Invalid success estimate for {candidate.tier}")
        for value in (estimate.estimated_cost, estimate.estimated_latency):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise RoutingError(f"Invalid cost or latency estimate for {candidate.tier}")

    ordered = sorted(candidates, key=lambda candidate: _TIER_ORDER.index(candidate.tier))
    if mode == "quality":
        return max(ordered, key=lambda candidate: estimates[candidate.tier].predicted_success)
    if mode == "economy":
        best = max(estimates[candidate.tier].predicted_success for candidate in ordered)
        close = [
            candidate
            for candidate in ordered
            if best - estimates[candidate.tier].predicted_success <= economy_margin
        ]
        priced = [
            candidate for candidate in close if estimates[candidate.tier].estimated_cost is not None
        ]
        return min(
            priced or close,
            key=lambda candidate: (
                estimates[candidate.tier].estimated_cost
                if estimates[candidate.tier].estimated_cost is not None
                else math.inf,
                _TIER_ORDER.index(candidate.tier),
            ),
        )
    if mode != "balanced":
        raise RoutingError(f"Unsupported router mode: {mode}")

    costs = _normalized([estimates[candidate.tier].estimated_cost for candidate in ordered])
    latencies = _normalized([estimates[candidate.tier].estimated_latency for candidate in ordered])
    return max(
        enumerate(ordered),
        key=lambda item: (
            weights.success * estimates[item[1].tier].predicted_success
            - weights.cost * costs[item[0]]
            - weights.latency * latencies[item[0]]
        ),
    )[1]


class HeuristicPolicy:
    """Fixed priors and tier latency proxies; historical statistics are ignored."""

    def select(
        self,
        features: TaskFeatures,
        candidates: Sequence[RouterCandidate],
        *,
        mode: RouterMode,
        stats: RouterStats,
        weights: RouterWeights = _DEFAULT_WEIGHTS,
        economy_margin: float = 0.05,
    ) -> RoutingDecision:
        del stats
        estimates = {
            candidate.tier: CandidateEstimate(
                predicted_success=prior_probability(features, candidate.tier),
                estimated_cost=_estimated_api_cost(candidate, features.context_tokens),
                estimated_latency=_LATENCY_PROXIES[candidate.tier],
            )
            for candidate in candidates
        }
        choice = choose_candidate(
            candidates, estimates, mode=mode, weights=weights, economy_margin=economy_margin
        )
        estimate = estimates[choice.tier]
        cost_text = (
            f"${estimate.estimated_cost:.6f} estimated API cost"
            if estimate.estimated_cost is not None
            else "unknown API cost"
        )
        return RoutingDecision(
            tier=choice.tier,
            provider=choice.provider,
            model=choice.model,
            mode=mode,
            reason=(
                f"{mode}: {estimate.predicted_success:.2f} baseline success prior, "
                f"{cost_text}, tier latency proxy"
            ),
            predicted_success=estimate.predicted_success,
        )


def _estimated_api_cost(candidate: RouterCandidate, input_tokens: int) -> float | None:
    if candidate.input_cost is None or candidate.output_cost is None:
        return None
    return (
        input_tokens * candidate.input_cost + _ESTIMATED_OUTPUT_TOKENS * candidate.output_cost
    ) / 1_000_000


def _normalized(values: Sequence[float | None]) -> list[float]:
    known_values = [value for value in values if value is not None]
    if not known_values:
        return [0.5 for _ in values]
    lowest, highest = min(known_values), max(known_values)
    if highest == lowest:
        return [0.5 if value is None else 0.0 for value in values]
    return [0.5 if value is None else (value - lowest) / (highest - lowest) for value in values]
