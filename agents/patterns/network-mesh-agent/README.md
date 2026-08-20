# network-mesh-agent — Network / Mesh topology

Peer agents that can all talk to each other, many-to-many, with no fixed hierarchy — each agent
decides which peer to hand off to next, at runtime. See
[docs/patterns/agent/network-mesh.md](../../../docs/patterns/agent/network-mesh.md) for the full
writeup, including its own honest framing: unlike Supervisor and Swarm, this topology has no
dedicated, actively-maintained first-party doc page or standalone library — reach for it last,
only once [Supervisor](../supervisor-agent/README.md)'s hierarchy and
[Swarm / Handoffs](../swarm-agent/README.md)'s peer-handoff both genuinely don't fit.

![Network / Mesh topology: five agents (Planner, Executor, Reviewer, Researcher, Validator) scattered in a loose non-symmetric layout with non-uniform connections including a route-back edge; any agent can route to any other, and the traversal path is decided at runtime rather than by a fixed graph shape](../../../public/images/agent/network-mesh.svg)

**Reach for this when:** there's genuinely no natural hierarchy (Supervisor) and no natural
handoff chain where one specialist should own the rest of the conversation (Swarm) — e.g. a small
set of peers that may need to go back and forth with each other before converging on an answer.

## Stack

Raw LangGraph `StateGraph`, built directly from the doc's own worked example — there's no
dedicated multi-agent-network library the way `langgraph-supervisor`/`langgraph-swarm` exist for
the other two topologies. Verified `StateGraph`/`add_conditional_edges` usage against current
LangChain docs via the `docs-langchain` and `reference-langchain` MCP servers rather than
training-data recollection. Same `agents-common` checkpointing/observability/config wiring as
every other pattern in this repo.

Unlike the doc's own `route_after_researcher` example (a second LLM call just to decide where to
go), each peer's single structured-output call produces its content *and* its routing signal in
one shot (`ResearchFinding.needs_critique`, `Critique.needs_more_research`) — the same
"structured output carries the routing decision" convention
[`evaluator-optimizer-agent`](../evaluator-optimizer-agent/README.md)'s `Evaluation.approved`
uses. The routing functions themselves are then pure Python.

## Shape

1. **`researcher`** — the entry point (though not a designated hub; it's just where every
   invocation happens to start). Investigates the task and decides for itself whether the finding
   needs the critic's review or is ready for the writer.
2. **`critic`** — reviews the researcher's finding and decides for itself whether the gaps it
   found are serious enough to send the task *back* to the researcher for another round, or
   whether it's ready for the writer as-is. This route-back edge — not present in any other
   multi-agent pattern in this repo — is the topology diagram's namesake feature: control can move
   backward as well as forward, decided locally by whichever agent just acted.
3. **`writer`** — synthesizes every peer's contribution (`MeshState.messages`) into the final
   answer. Always the last node to run.
4. A `research_rounds` counter caps the researcher<->critic loop
   (`DEFAULT_MAX_RESEARCH_ROUNDS = 2`) so a mesh that keeps deciding it needs another round
   can't recurse forever — same "cap the loop, force convergence" principle
   `evaluator-optimizer-agent` applies to its own generate<->evaluate loop.

Per the doc's own practical caveats — mesh communication scales O(N²), and a shared-state topology
needs an explicit merge strategy per field — this mesh stays deliberately small (three peers), and
every field in `MeshState` besides `messages` (an `operator.add`-accumulated field, same
convention as every other multi-writer field in this repo) has exactly one writer active at a
time, so there's no concurrent-write conflict to resolve.

## Running it

```bash
make up   # starts Postgres + MLflow + provisions the prompts/dataset/gateway route
uv run network-mesh-agent "Summarize recent trends in vector database indexing."
```

Prints the writer's final answer. Whether the mesh took the direct researcher->writer path or
looped through the critic first varies by task, with no change to the graph itself.

## Tests

```bash
make test-unit          # tests/unit — no external services; model is stubbed
make up
make test-integration   # tests/integration — real Postgres, checkpoint round-tripping
make provision-datasets
make test-eval          # tests/evals — calls a real model via the MLflow AI Gateway
```

- `tests/unit/test_graph.py` — asserts the direct researcher->writer path, the
  researcher->critic->writer path, the critic's route-back to the researcher actually firing, and
  that `max_research_rounds` forces convergence to the writer even when the critic keeps asking
  for more research.
- `tests/integration/test_checkpointing.py` — asserts the accumulated transcript and final answer
  survive a rebuild of the compiled graph against a real Postgres-backed checkpointer.
- `tests/evals/test_quality.py` — MLflow GenAI eval suite scoring whether the final answer is
  grounded in the mesh's own research/critique transcript, against the seed dataset at
  `packages/mlflow-server/datasets/network-mesh-agent.jsonl`. Unlike
  `orchestrator-workers-agent`'s `min_subtasks` lower bound, there's no per-record routing
  expectation to check — which path the mesh takes is itself a runtime decision.
