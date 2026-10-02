# Dynamic Model Router, Part 4: deterministic policy

Part 4 adds model selection without changing Tau's CLI or agent loop. The
`RoutingPolicy` interface, `HeuristicPolicy`, shared `choose_candidate` scorer,
and `RoutingDecision` live in `tau_coding`. Part 5 will call this layer before
a new user turn begins. Later adaptive and Laya policies can implement the
same interface.

## Baseline estimates

The heuristic starts with hand-set success priors by complexity and tier:

| Complexity | Small | Medium | Strong |
| --- | ---: | ---: | ---: |
| Easy | 0.92 | 0.96 | 0.98 |
| Medium | 0.65 | 0.83 | 0.93 |
| Hard | 0.35 | 0.65 | 0.88 |

Task-type offsets are explanation +0.03, debugging -0.04, tests 0,
code generation -0.02, refactoring -0.01, architecture -0.05, and unknown 0.
Results are clamped to 0.01–0.99. These values are **assumptions**, not measured
success probabilities or confidence scores. They give the deterministic
baseline a reproducible task-conditioned preference; Phase 8 will smooth
verified observations against them.

API cost is estimated from the candidate's per-million-token input/output
rates, current estimated context tokens, and a fixed 1,024-token output
allowance. If either rate is missing, cost is unknown rather than zero. Until
telemetry provides observed latency, the heuristic uses relative tier proxies
of 1, 2, and 3. These are scoring proxies, not seconds or measurements.

## Modes

- **Quality** chooses the highest prior success estimate.
- **Economy** finds the best success estimate, keeps candidates within the
  default 0.05 margin, then chooses the cheapest priced one. Unpriced
  candidates are not treated as free. If none in the margin is priced, tier
  order provides a deterministic fallback.
- **Balanced** computes `success_weight × success - cost_weight × normalized
  cost - latency_weight × normalized latency`. Defaults are 1.0, 0.15, and
  0.05. Known costs and latencies use min–max normalization among eligible
  candidates. An unknown value gets a neutral 0.5 penalty; identical known
  values contribute no penalty. Weights and the economy margin are arguments
  to the policy so later CLI/config integration can expose them without
  changing the scorer.

Exact ties resolve in `small`, `medium`, `strong` order, regardless of input
order. An empty eligible set raises `RoutingError` before any agent run.
Every decision contains the provider/model pair, tier, mode, prior estimate,
and a short explanation. No provider or inference call is made here.

## How to check

```bash
uv run --frozen pytest -q tests/test_router_policy.py tests/test_router_features.py tests/test_router_config.py
uv run --frozen ruff check src/tau_coding/router_policy.py tests/test_router_policy.py
uv run --frozen mypy src/tau_coding/router_policy.py
```
