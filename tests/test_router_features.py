import pytest

from tau_agent.messages import ImageContent, UserMessage
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderModelMetadata,
    ProviderSettings,
)
from tau_coding.router_config import RouterCandidate
from tau_coding.router_features import analyze_turn, filter_router_candidates


@pytest.mark.parametrize(
    ("prompt", "task_type", "complexity"),
    [
        ("Explain this function", "explanation", "easy"),
        ("Write tests for this function", "tests", "medium"),
        ("Implement a parser", "code_generation", "medium"),
        ("Refactor the parser", "refactoring", "medium"),
        ("Design the service architecture", "architecture", "hard"),
        ("Hello", "unknown", "easy"),
    ],
)
def test_analyze_turn_has_stable_task_categories(
    prompt: str, task_type: str, complexity: str
) -> None:
    features = analyze_turn(prompt)

    assert (features.task_type, features.complexity) == (task_type, complexity)


def test_analyze_turn_uses_prompt_and_context_length() -> None:
    assert analyze_turn("x" * 300).complexity == "medium"
    assert analyze_turn("x" * 1200).complexity == "hard"
    assert analyze_turn("Hello", messages=(UserMessage(content="x" * 64_000),)).complexity == "hard"


def test_analyze_turn_classifies_task_and_estimates_context() -> None:
    features = analyze_turn(
        "Debug why the worker deadlocks under load",
        system="You are a coding agent.",
        messages=(UserMessage(content="Earlier task"),),
    )

    assert features.task_type == "debugging"
    assert features.complexity == "hard"
    assert features.prompt_length == 41
    assert features.context_tokens > 0
    assert features.required_capabilities == frozenset({"text"})
    assert features == analyze_turn(
        "Debug why the worker deadlocks under load",
        system="You are a coding agent.",
        messages=(UserMessage(content="Earlier task"),),
    )


def test_analyze_turn_detects_image_requirement_from_session() -> None:
    message = UserMessage(content=[ImageContent(data="abc", mime_type="image/png")])

    features = analyze_turn("Explain this screenshot", messages=(message,))

    assert features.task_type == "explanation"
    assert features.required_capabilities == frozenset({"text", "image"})
    current_image = analyze_turn("Explain this screenshot", has_image=True)
    assert current_image.required_capabilities == frozenset({"text", "image"})


def test_filter_router_candidates_reports_context_and_capability_rejections() -> None:
    features = analyze_turn(
        "Explain this screenshot",
        messages=(UserMessage(content=[ImageContent(data="abc", mime_type="image/png")]),),
    )
    candidates = (
        RouterCandidate("small", "local", "small", context_window=8),
        RouterCandidate("medium", "local", "medium", context_window=1000),
        RouterCandidate("strong", "local", "strong", context_window=1000),
    )
    settings = ProviderSettings(
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                models=("small", "medium", "strong"),
                default_model="small",
                model_metadata={
                    "medium": ProviderModelMetadata(input=("text",)),
                    "strong": ProviderModelMetadata(input=("text", "image")),
                },
            ),
        )
    )

    result = filter_router_candidates(features, candidates, settings)

    assert [candidate.tier for candidate in result.eligible] == ["strong"]
    assert result.rejected == {
        "small": "context window too small",
        "medium": "image input unavailable",
    }
