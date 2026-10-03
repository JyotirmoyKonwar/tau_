from pathlib import Path

import pytest
from typer.testing import CliRunner

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage
from tau_ai import FakeProvider
from tau_coding import cli
from tau_coding.cli import app, run_print_mode
from tau_coding.paths import TauPaths
from tau_coding.provider_config import OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding.rendering import PrintOutputMode
from tau_coding.resources import TauResourcePaths
from tau_coding.router_config import RouterCandidate, RouterConfig
from tau_coding.router_policy import HeuristicPolicy, RoutingError
from tau_coding.session_manager import SessionManager


@pytest.mark.parametrize("model", ["auto", "auto:economy", "auto:balanced", "auto:quality"])
def test_cli_accepts_auto_model_in_print_mode(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    models: list[str | None] = []

    async def fake_run(*args: object, **kwargs: object) -> bool:
        models.append(args[1] if isinstance(args[1], str) else None)
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_run)

    result = CliRunner().invoke(app, ["--print", "--model", model, "hello"])

    assert result.exit_code == 0
    assert models == [model]


@pytest.mark.parametrize(
    "args",
    [
        ["--print", "--model", "auto:turbo", "hello"],
        ["--print", "--model", "auto", "--provider", "cloud", "hello"],
        ["--model", "auto", "hello"],
    ],
)
def test_cli_rejects_unsupported_auto_use(args: list[str]) -> None:
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 2
    assert "auto" in result.output.lower()


@pytest.mark.anyio
async def test_auto_wrapper_passes_mode_and_config_to_print_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local", models=("tiny", "mid", "large"), default_model="tiny"
            ),
        ),
    )
    config = RouterConfig(
        default_policy="balanced",
        tiers=(
            RouterCandidate("small", "local", "tiny"),
            RouterCandidate("medium", "local", "mid"),
            RouterCandidate("strong", "local", "large"),
        ),
    )
    seen: list[dict[str, object]] = []

    class ClosableFakeProvider(FakeProvider):
        async def aclose(self) -> None:
            pass

    async def fake_run_print_mode(**kwargs: object) -> bool:
        seen.append(kwargs)
        return True

    monkeypatch.setattr(cli, "load_provider_settings", lambda: settings)
    monkeypatch.setattr(cli, "load_router_config", lambda: config)
    monkeypatch.setattr(
        cli, "create_model_provider", lambda *args, **kwargs: ClosableFakeProvider([])
    )
    monkeypatch.setattr(cli, "run_print_mode", fake_run_print_mode)

    ok = await cli.run_openai_print_mode(
        "Hello",
        "auto:economy",
        tmp_path,
        session_manager=SessionManager(TauPaths(home=tmp_path / ".tau")),
    )

    assert ok is True
    assert seen[0]["model"] == "tiny"
    assert seen[0]["requested_model"] is None
    assert seen[0]["router_config"] is config
    assert seen[0]["router_mode"] == "economy"


@pytest.mark.anyio
async def test_print_auto_routes_before_fake_provider_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import tau_coding.session as session_module

    providers: list[FakeProvider] = []
    decisions: list[str] = []

    class CountingPolicy(HeuristicPolicy):
        def select(self, features, candidates, *, mode, stats, weights=None, economy_margin=0.05):
            decisions.append(mode)
            return super().select(
                features, candidates, mode=mode, stats=stats, economy_margin=economy_margin
            )

    class ClosableFakeProvider(FakeProvider):
        async def aclose(self) -> None:
            pass

    def fake_runtime_provider(*args: object, **kwargs: object) -> FakeProvider:
        provider = ClosableFakeProvider(
            [
                [
                    assistant_start(model="large"),
                    assistant_done(message=AssistantMessage(content="Done")),
                ]
            ]
        )
        providers.append(provider)
        return provider

    monkeypatch.setattr(session_module, "_create_runtime_provider", fake_runtime_provider)
    monkeypatch.setenv("ROUTER_TEST_KEY", "dummy")
    settings = ProviderSettings(
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                models=("tiny", "mid", "large"),
                default_model="tiny",
                api_key_env="ROUTER_TEST_KEY",
            ),
        )
    )
    config = RouterConfig(
        default_policy="quality",
        tiers=(
            RouterCandidate("small", "local", "tiny"),
            RouterCandidate("medium", "local", "mid"),
            RouterCandidate("strong", "local", "large"),
        ),
    )

    ok = await run_print_mode(
        prompt="Debug this concurrency deadlock",
        model="tiny",
        cwd=tmp_path,
        provider=FakeProvider([]),
        output=PrintOutputMode.text,
        provider_name="local",
        provider_settings=settings,
        runtime_provider_config=settings.providers[0],
        router_config=config,
        routing_policy=CountingPolicy(),
        resource_paths=TauResourcePaths(root=tmp_path / "resources", agents_root=None),
    )

    captured = capsys.readouterr()
    assert ok is True, captured
    assert providers[-1].calls[0][0] == "large"
    assert decisions == ["quality"]
    assert "strong" in captured.err
    assert captured.out == "Done\n"


@pytest.mark.anyio
async def test_print_auto_rejects_unfit_candidates_before_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ROUTER_TEST_KEY", "dummy")
    settings = ProviderSettings(
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                models=("tiny", "mid", "large"),
                default_model="tiny",
                api_key_env="ROUTER_TEST_KEY",
            ),
        )
    )
    config = RouterConfig(
        default_policy="balanced",
        tiers=(
            RouterCandidate("small", "local", "tiny", context_window=1),
            RouterCandidate("medium", "local", "mid", context_window=1),
            RouterCandidate("strong", "local", "large", context_window=1),
        ),
    )
    provider = FakeProvider([])

    with pytest.raises(RoutingError, match="No eligible router candidates.*context window"):
        await run_print_mode(
            prompt="Hello",
            model="tiny",
            cwd=tmp_path,
            provider=provider,
            provider_name="local",
            provider_settings=settings,
            router_config=config,
            resource_paths=TauResourcePaths(root=tmp_path / "resources", agents_root=None),
        )

    assert provider.calls == []
