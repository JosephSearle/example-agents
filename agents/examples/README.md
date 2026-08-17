# agents/examples/

Applied, end-to-end demonstrations — either composing one or more
[`agents/patterns/*`](../patterns/README.md) packages into a concrete use case, or a standalone
applied use of a framework tier that doesn't (yet) have its own `agents/patterns/` entry. The
"how do I actually use this" counterpart to the patterns' "how does this technique work in
isolation."

| Example | Framework tier | Path |
|---|---|---|
| MLflow AI Issue Discovery | Tier 3 — `deepagents` | `agents/examples/experiment-analysis-agent` |

`experiment-analysis-agent` doesn't compose an existing `agents/patterns/*` package — it's this
repo's first applied use of `deepagents` (Tier 3 in the framework-tiering table), analyzing the
other agents' MLflow traces rather than building on their code. See its own README for what it
demonstrates.

## Convention

Each example is its own uv workspace member under `agents/examples/<name>/`, laid out the same
way as a pattern package:

```
agents/examples/<name>/
├── pyproject.toml
├── README.md
├── src/<name>/
└── tests/
```

Depend on the pattern(s) it builds on as real workspace packages — don't copy or fork their code:

```toml
[project]
dependencies = ["react-agent"]

[tool.uv.sources]
react-agent = { workspace = true }
```

Also depend on `agents-common` directly for checkpointing/observability/config, the same as a
pattern package would.

`agents/examples/*` is already registered in the root `pyproject.toml`
(`[tool.uv.workspace] members`), so a new example package is picked up by `uv sync` automatically
once it has a `pyproject.toml`.
