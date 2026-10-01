# Dynamic Model Router, Part 1: baseline

This part installs Tau's development environment in the repository's `.venv` and
records the existing model-selection behavior. It adds tests and this note; it
does not add automatic routing.

## Current path through Tau

- The print-mode CLI accepts `--provider` and `--model`, resolves them during
  session startup, then calls `CodingSession.prompt()`.
- The TUI delegates a model choice to `CodingSession.select_provider_model()`.
- `select_provider_model()` builds the candidate runtime before publishing it,
  appends a provider-aware `ModelChangeEntry` and leaf, updates the active
  provider/model, and saves the choice as the manual default. Re-selecting the
  active pair does nothing.
- Session resume reconstructs the provider/model from the target session.
- `CodingSession.prompt()` expands input, runs the portable `AgentHarness`, and
  persists events. It already handles context-overflow compaction and a separate
  Hugging Face inference-provider route failover. Neither is cross-model routing.
- Provider/model metadata and usage accounting already live in `tau_coding`.
  The future router belongs there too; `tau_agent` should continue to know only
  the active provider and model.

## Checks

The new tests cover explicit print-mode choice forwarding, TUI delegation to
`select_provider_model()`, and that method's cross-provider publication and
idempotence. Existing tests cover resumed model state, provider errors, overflow
recovery, and Hugging Face route failover.

From the repository root, run:

```bash
uv sync --frozen --group dev
uv run --frozen pytest -q tests/test_cli.py tests/test_tui_app.py tests/test_coding_session.py
uv run --frozen ruff check tests/test_cli.py tests/test_tui_app.py tests/test_coding_session.py
uv run --frozen mypy
```

The next part can add router configuration. Before a router uses
`select_provider_model()`, it must make automatic switches distinct from manual
default changes.

Part 1 validation: 661 relevant tests passed; Ruff and mypy passed. All commands
used the repository `.venv` through `uv run --frozen`.
