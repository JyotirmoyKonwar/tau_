import json
from pathlib import Path

import pytest

from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderModelMetadata,
    ProviderSettings,
)
from tau_coding.router_config import (
    RouterConfigError,
    load_router_config,
    resolve_router_candidates,
)


def _config_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "default_policy": "balanced",
        "tiers": {
            "small": {"provider": "local", "model": "small-model"},
            "medium": {"provider": "local", "model": "medium-model"},
            "strong": {"provider": "cloud", "model": "strong-model"},
        },
    }


def _write_config(tmp_path: Path, data: dict[str, object]) -> TauPaths:
    paths = TauPaths(home=tmp_path / ".tau")
    paths.home.mkdir()
    paths.router_config_path.write_text(json.dumps(data), encoding="utf-8")
    return paths


def _settings() -> ProviderSettings:
    return ProviderSettings(
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                models=("small-model", "medium-model"),
                default_model="small-model",
                context_windows={"small-model": 8192},
                model_metadata={
                    "small-model": ProviderModelMetadata(cost={"input": 0.1, "output": 0.2}),
                    "medium-model": ProviderModelMetadata(
                        context_window=32768,
                        cost={"input": 0.3, "output": 0.6},
                    ),
                },
            ),
            OpenAICompatibleProviderConfig(
                name="cloud", models=("strong-model",), default_model="strong-model"
            ),
        )
    )


def test_load_router_config_reads_three_tiers(tmp_path: Path) -> None:
    config = load_router_config(_write_config(tmp_path, _config_data()))

    assert config.default_policy == "balanced"
    assert [choice.model for choice in config.tiers] == [
        "small-model",
        "medium-model",
        "strong-model",
    ]


def test_load_router_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RouterConfigError, match="router.json"):
        load_router_config(TauPaths(home=tmp_path / ".tau"))


def test_load_router_config_rejects_missing_tier(tmp_path: Path) -> None:
    data = _config_data()
    assert isinstance(data["tiers"], dict)
    del data["tiers"]["medium"]

    with pytest.raises(RouterConfigError, match="medium"):
        load_router_config(_write_config(tmp_path, data))


def test_load_router_config_rejects_duplicate_models(tmp_path: Path) -> None:
    data = _config_data()
    assert isinstance(data["tiers"], dict)
    data["tiers"]["medium"] = {"provider": "local", "model": "small-model"}

    with pytest.raises(RouterConfigError, match="small.*medium|medium.*small"):
        load_router_config(_write_config(tmp_path, data))


@pytest.mark.parametrize(
    ("field", "value"),
    [("default_policy", "turbo"), ("schema_version", 2)],
)
def test_load_router_config_rejects_unsupported_values(
    tmp_path: Path, field: str, value: object
) -> None:
    data = _config_data()
    data[field] = value

    with pytest.raises(RouterConfigError, match=field):
        load_router_config(_write_config(tmp_path, data))


@pytest.mark.parametrize("override", [0, -1, True, "large"])
def test_load_router_config_rejects_invalid_context_override(
    tmp_path: Path, override: object
) -> None:
    data = _config_data()
    assert isinstance(data["tiers"], dict)
    data["tiers"]["small"] = {
        "provider": "local",
        "model": "small-model",
        "context_window": override,
    }

    with pytest.raises(RouterConfigError, match="context_window"):
        load_router_config(_write_config(tmp_path, data))


@pytest.mark.parametrize("price", [-1, True, "free", float("inf"), 10**400])
def test_load_router_config_rejects_invalid_price_override(tmp_path: Path, price: object) -> None:
    data = _config_data()
    assert isinstance(data["tiers"], dict)
    data["tiers"]["small"] = {"provider": "local", "model": "small-model", "input_cost": price}

    with pytest.raises(RouterConfigError, match="input_cost"):
        load_router_config(_write_config(tmp_path, data))


def test_load_router_config_rejects_unknown_field(tmp_path: Path) -> None:
    data = _config_data()
    assert isinstance(data["tiers"], dict)
    data["tiers"]["small"] = {"provider": "local", "model": "small-model", "contex_window": 16000}

    with pytest.raises(RouterConfigError, match="contex_window"):
        load_router_config(_write_config(tmp_path, data))


def test_resolve_router_candidates_uses_catalog_metadata(tmp_path: Path) -> None:
    config = load_router_config(_write_config(tmp_path, _config_data()))
    resolved = resolve_router_candidates(config, _settings())

    assert [(candidate.tier, candidate.provider, candidate.model) for candidate in resolved] == [
        ("small", "local", "small-model"),
        ("medium", "local", "medium-model"),
        ("strong", "cloud", "strong-model"),
    ]
    assert (resolved[0].context_window, resolved[0].input_cost, resolved[0].output_cost) == (
        8192,
        0.1,
        0.2,
    )
    assert (resolved[1].context_window, resolved[1].input_cost, resolved[1].output_cost) == (
        32768,
        0.3,
        0.6,
    )
    assert (resolved[2].context_window, resolved[2].input_cost, resolved[2].output_cost) == (
        None,
        None,
        None,
    )


def test_resolve_router_candidates_uses_overrides(tmp_path: Path) -> None:
    data = _config_data()
    assert isinstance(data["tiers"], dict)
    data["tiers"]["small"] = {
        "provider": "local",
        "model": "small-model",
        "context_window": 16000,
        "input_cost": 0.05,
        "output_cost": 0.15,
    }
    config = load_router_config(_write_config(tmp_path, data))

    small = resolve_router_candidates(config, _settings())[0]

    assert (small.context_window, small.input_cost, small.output_cost) == (16000, 0.05, 0.15)


def test_resolve_router_candidates_rejects_unavailable_choice(tmp_path: Path) -> None:
    config = load_router_config(_write_config(tmp_path, _config_data()))

    with pytest.raises(RouterConfigError, match="medium.*local/medium-model"):
        resolve_router_candidates(
            config,
            _settings(),
            available_choices={
                ("local", "small-model"),
                ("cloud", "strong-model"),
            },
        )
