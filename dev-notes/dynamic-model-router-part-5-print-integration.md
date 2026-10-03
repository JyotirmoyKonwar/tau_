# Dynamic Model Router, Part 5: print-mode integration

Part 5 connects the deterministic router to one print-mode user turn. Run
`tau --print --model auto "<prompt>"` to use the `default_policy` in
`~/.tau/router.json`, or `auto:economy`, `auto:balanced`, or `auto:quality` to
choose a mode for that invocation. `--router-engine heuristic` is explicit;
adaptive and Laya engines are not implemented yet.

Tau first prepares the session, so a resumed print turn can be analyzed with
its restored transcript and assembled system prompt. Informational slash
commands and terminal commands do not route. Before an agent prompt starts,
Tau resolves the three configured candidates against the session's available
provider/model choices, estimates task features, filters known context and
image incompatibilities, and asks the policy for one decision. The selected
provider/model enters Tau's existing session switch and agent run path. A
human-readable decision goes to stderr; text, JSON, and transcript stdout
retain their usual rendering.

An auto choice updates the active session model, including the model recorded
for later resume, but does not overwrite the user's saved manual default.
Manual `--provider`/`--model` operation still follows its original path. An
invalid router file, unavailable candidate, or empty eligible set reports an
error before an inference request. Phase 6 will add session-aware routing to
the interactive TUI; for now `--model auto` is print-mode only.

## How to check

```bash
uv run --frozen pytest -q tests/test_router_cli.py tests/test_cli.py
uv run --frozen ruff check src/tau_coding/cli.py src/tau_coding/session.py tests/test_router_cli.py
uv run --frozen mypy
```

The integration test uses fake provider responses and makes no API requests.
