"""Load and resolve the three model choices for Tau's opt-in router."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, cast

from tau_coding.paths import TauPaths
from tau_coding.provider_config import ProviderSettings

RouterTier = Literal["small", "medium", "strong"]
RouterPolicy = Literal["economy", "balanced", "quality"]
_TIERS: tuple[RouterTier, ...] = ("small", "medium", "strong")
_POLICIES = {"economy", "balanced", "quality"}


class RouterConfigError(ValueError):
    """An unusable router configuration or model choice."""


@dataclass(frozen=True, slots=True)
class RouterCandidate:
    """One configured provider/model pair and its optional metadata overrides."""

    tier: RouterTier
    provider: str
    model: str
    context_window: int | None = None
    input_cost: float | None = None
    output_cost: float | None = None


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """A version-one, three-tier router configuration."""

    default_policy: RouterPolicy
    tiers: tuple[RouterCandidate, RouterCandidate, RouterCandidate]


def load_router_config(paths: TauPaths | None = None) -> RouterConfig:
    """Read ``~/.tau/router.json`` without changing provider settings."""
    path = (paths or TauPaths()).router_config_path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RouterConfigError(f"Could not read router config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RouterConfigError(f"Invalid JSON in router config {path}: {exc}") from exc
    root = _object(data, "router config")
    _fields(root, {"schema_version", "default_policy", "tiers"}, "router config")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise RouterConfigError("router config schema_version must be 1")
    policy = root["default_policy"]
    if not isinstance(policy, str) or policy not in _POLICIES:
        raise RouterConfigError("default_policy must be economy, balanced, or quality")
    tier_data = _object(root["tiers"], "tiers")
    _fields(tier_data, set(_TIERS), "tiers")
    candidates = (
        _candidate("small", tier_data["small"]),
        _candidate("medium", tier_data["medium"]),
        _candidate("strong", tier_data["strong"]),
    )
    seen: dict[tuple[str, str], RouterTier] = {}
    for candidate in candidates:
        pair = (candidate.provider, candidate.model)
        if pair in seen:
            raise RouterConfigError(
                f"{candidate.tier} duplicates {seen[pair]} model "
                f"{candidate.provider}/{candidate.model}"
            )
        seen[pair] = candidate.tier
    return RouterConfig(default_policy=cast(RouterPolicy, policy), tiers=candidates)


def resolve_router_candidates(
    config: RouterConfig,
    settings: ProviderSettings,
    *,
    available_choices: Collection[tuple[str, str]] | None = None,
) -> tuple[RouterCandidate, RouterCandidate, RouterCandidate]:
    """Validate availability and fill missing metadata from Tau's provider catalog.

    Pass the session's current provider/model pairs as ``available_choices`` when
    present. The default uses configured providers, which is useful before a
    CodingSession exists. Unknown metadata stays unknown rather than invented.
    """
    if available_choices is None:
        available = {
            (provider.name, model) for provider in settings.providers for model in provider.models
        }
    else:
        available = set(available_choices)
    resolved: list[RouterCandidate] = []
    for choice in config.tiers:
        if (choice.provider, choice.model) not in available:
            raise RouterConfigError(
                f"{choice.tier} model is unavailable: {choice.provider}/{choice.model}"
            )
        provider = next((item for item in settings.providers if item.name == choice.provider), None)
        metadata = provider.model_metadata.get(choice.model) if provider is not None else None
        cost = metadata.cost if metadata is not None else {}
        context_window = choice.context_window
        if context_window is None and metadata is not None:
            context_window = metadata.context_window
        if context_window is None and provider is not None:
            context_window = provider.context_windows.get(choice.model)
        resolved.append(
            RouterCandidate(
                tier=choice.tier,
                provider=choice.provider,
                model=choice.model,
                context_window=context_window,
                input_cost=(
                    choice.input_cost if choice.input_cost is not None else cost.get("input")
                ),
                output_cost=(
                    choice.output_cost if choice.output_cost is not None else cost.get("output")
                ),
            )
        )
    return resolved[0], resolved[1], resolved[2]


def _candidate(tier: RouterTier, data: object) -> RouterCandidate:
    value = _object(data, tier)
    _fields(
        value,
        {"provider", "model"},
        tier,
        optional={"context_window", "input_cost", "output_cost"},
    )
    provider = _name(value["provider"], f"{tier}.provider")
    model = _name(value["model"], f"{tier}.model")
    context = value.get("context_window")
    if context is not None and (type(context) is not int or context <= 0):
        raise RouterConfigError(f"{tier}.context_window must be a positive integer")
    return RouterCandidate(
        tier=tier,
        provider=provider,
        model=model,
        context_window=context,
        input_cost=_cost(value.get("input_cost"), f"{tier}.input_cost"),
        output_cost=_cost(value.get("output_cost"), f"{tier}.output_cost"),
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RouterConfigError(f"{label} must be a JSON object")
    return value


def _fields(
    value: dict[str, object], required: set[str], label: str, *, optional: set[str] | None = None
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing:
        raise RouterConfigError(f"{label} is missing {', '.join(sorted(missing))}")
    if unknown:
        raise RouterConfigError(f"{label} has unknown fields: {', '.join(sorted(unknown))}")


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RouterConfigError(f"{label} must be a non-empty name")
    return value


def _cost(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RouterConfigError(f"{label} must be a non-negative number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise RouterConfigError(f"{label} must be a non-negative finite number") from exc
    if not math.isfinite(number) or number < 0:
        raise RouterConfigError(f"{label} must be a non-negative finite number")
    return number
