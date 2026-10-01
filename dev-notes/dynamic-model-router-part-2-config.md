# Dynamic Model Router, Part 2: candidate configuration

Part 2 adds the router's configuration seam. It does not select a model or make
inference requests. This keeps provider setup and model switching in
`tau_coding`, while the portable `tau_agent` remains unchanged.

## File and shape

Copy [the example](examples/router.json) to `~/.tau/router.json`, then replace
the placeholder provider and model IDs with three distinct choices already
configured in Tau. The file requires `schema_version: 1`, `default_policy`
(`economy`, `balanced`, or `quality`), and exactly one `small`, `medium`, and
`strong` choice. No policy is active yet; later parts will connect this file to
the CLI and TUI.

Each choice can add `context_window`, `input_cost`, or `output_cost`. Context is
in tokens; costs are estimated USD per million tokens. Overrides take precedence
over Tau's provider catalog. A missing catalog value remains unknown. The
resolver checks that each provider/model pair is available, and reports the
unavailable tier before any agent run.

For the initial Hugging Face experiment, the provider/model IDs selected in the
project discussion are:

| Tier | Provider | Model |
| --- | --- | --- |
| Small | `huggingface` | `openai/gpt-oss-20b` |
| Medium | `huggingface` | `deepseek-ai/DeepSeek-V4-Flash` |
| Strong | `huggingface` | `deepseek-ai/DeepSeek-V4-Pro` |

Inference-provider availability and prices can change. Check live Hugging Face
metadata before recording benchmark costs; do not treat bundled catalog prices
as a bill. Tau's existing Hugging Face provider preferences can pin a backing
provider separately from this model-tier file.

## How to check

From the repository root, using the project `.venv`:

```bash
uv run --frozen pytest -q tests/test_router_config.py tests/test_paths.py
uv run --frozen ruff check .
uv run --frozen mypy
```

`load_router_config(TauPaths(...))` reads and validates the file.
`resolve_router_candidates(config, provider_settings, available_choices=...)`
fills context and prices from the provider catalog where no override is set.
`available_choices` can receive the active session's provider/model pairs when
Part 5 connects routing to `CodingSession`.
