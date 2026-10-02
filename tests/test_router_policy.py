from dataclasses import asdict

import pytest

from tau_coding.router_config import RouterCandidate
from tau_coding.router_features import TaskFeatures
from tau_coding.router_policy import (
    CandidateEstimate,
    HeuristicPolicy,
    RouterWeights,
    RoutingError,
    RoutingPolicy,
    choose_candidate,
    prior_probability,
)


def _candidates() -> tuple[RouterCandidate, ...]:
    return (
        RouterCandidate("small", "local", "tiny", input_cost=0.1, output_cost=0.2),
        RouterCandidate("medium", "cloud", "mid", input_cost=0.4, output_cost=0.8),
        RouterCandidate("strong", "cloud", "large", input_cost=2.0, output_cost=4.0),
    )


def _features(task_type: str = "debugging", complexity: str = "hard") -> TaskFeatures:
    return TaskFeatures(task_type, complexity, 40, 1000, frozenset({"text"}))


def test_heuristic_policy_selects_from_eligible_candidates() -> None:
    policy: RoutingPolicy = HeuristicPolicy()

    decision = policy.select(_features(), _candidates(), mode="quality", stats={})

    assert (decision.tier, decision.provider, decision.model, decision.mode) == (
        "strong",
        "cloud",
        "large",
        "quality",
    )
    assert decision.reason
    assert asdict(decision)["tier"] == "strong"
    assert policy.select(_features(), _candidates()[:2], mode="quality", stats={}).tier == "medium"


def test_priors_account_for_task_type_and_complexity() -> None:
    assert prior_probability(_features("debugging", "hard"), "small") == pytest.approx(0.31)
    assert prior_probability(_features("explanation", "easy"), "small") == pytest.approx(0.95)
    assert prior_probability(_features("architecture", "hard"), "strong") == pytest.approx(0.83)


def test_economy_chooses_cheapest_within_success_margin() -> None:
    candidates = _candidates()
    estimates = {
        "small": CandidateEstimate(0.70, 0.01, 2.0),
        "medium": CandidateEstimate(0.88, 0.08, 4.0),
        "strong": CandidateEstimate(0.91, 0.50, 9.0),
    }

    assert choose_candidate(candidates, estimates, mode="economy").tier == "medium"
    stricter = choose_candidate(candidates, estimates, mode="economy", economy_margin=0.02)
    assert stricter.tier == "strong"


def test_balanced_weights_can_change_choice() -> None:
    policy = HeuristicPolicy()
    features = _features("explanation", "easy")

    assert policy.select(features, _candidates(), mode="balanced", stats={}).tier == "small"
    assert (
        policy.select(
            features,
            _candidates(),
            mode="balanced",
            stats={},
            weights=RouterWeights(success=1.0, cost=0.0, latency=0.0),
        ).tier
        == "medium"
    )


def test_ties_use_tier_order_even_if_candidates_arrive_reversed() -> None:
    candidates = tuple(reversed(_candidates()))
    estimates = {candidate.tier: CandidateEstimate(0.8, 0.1, 2.0) for candidate in candidates}

    assert choose_candidate(candidates, estimates, mode="quality").tier == "small"
    assert choose_candidate(candidates, estimates, mode="balanced").tier == "small"


def test_unpriced_candidates_are_not_treated_as_free() -> None:
    candidates = (
        RouterCandidate("small", "local", "unknown"),
        RouterCandidate("medium", "cloud", "priced", input_cost=1.0, output_cost=1.0),
    )
    estimates = {
        "small": CandidateEstimate(0.88, None, None),
        "medium": CandidateEstimate(0.91, 0.10, None),
    }

    assert choose_candidate(candidates, estimates, mode="economy").tier == "medium"


def test_no_eligible_candidate_fails_before_a_run() -> None:
    with pytest.raises(RoutingError, match="No eligible router candidates"):
        HeuristicPolicy().select(_features(), (), mode="balanced", stats={})
