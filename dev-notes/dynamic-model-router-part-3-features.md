# Dynamic Model Router, Part 3: turn analysis and candidate filtering

Part 3 adds deterministic analysis and filtering in `tau_coding`. It does not
select a model or call a provider. Part 5 will connect these functions to new
user turns. The reusable `tau_agent` harness remains unchanged.

## Features

`analyze_turn(prompt, system=..., messages=..., tools=..., has_image=...)`
returns task type, easy/medium/hard complexity, prompt character count,
estimated context tokens, and required input capabilities. `messages` contains
the active context before the new prompt; the function adds the prompt when
using Tau's existing context estimator. Set `has_image=True` for an image on
the new turn. Earlier user images in `messages` also require image input.

Task type uses ordered word rules: architecture, debugging, tests,
refactoring, code generation, then explanation. An unmatched prompt is
`unknown`. Complexity is hard for architecture, explicit concurrency or
migration terms, prompts of at least 1,200 characters, or estimated context of
at least 16,000 tokens. It is medium for prompts of at least 300 characters,
common coding verbs, tests, or refactoring. Other prompts are easy. These
rules are intentionally simple, stable baselines for later policies and
benchmarks; they do not claim to measure actual task difficulty.

`filter_router_candidates(features, candidates, provider_settings)` preserves
candidate order and reports why each rejected candidate could not satisfy the
turn. A known context window must exceed the estimated context size. Image
input must be explicitly supported in Tau's provider metadata. Unknown
context-window metadata remains unknown, so the candidate stays eligible;
the later provider may still reject an oversized request. Image token costs
are only as precise as Tau's existing context estimator.

## How to check

```bash
uv run --frozen pytest -q tests/test_router_features.py tests/test_router_config.py
uv run --frozen ruff check src/tau_coding/router_features.py tests/test_router_features.py
uv run --frozen mypy src/tau_coding/router_features.py
```

All checks use fake provider settings and session messages, with no inference
requests or API spend.
