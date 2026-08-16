# agents/

Two directories, two different jobs:

- **[`patterns/`](patterns/README.md)** — reusable reference implementations of agentic patterns
  (ReAct, supervisor, swarm, deep agent). Each one is a standalone package you'd copy the
  *approach* from, or depend on directly.
- **[`examples/`](examples/README.md)** — applied, end-to-end demos that compose one or more
  `patterns/*` packages into a concrete use case. Examples depend on patterns as real workspace
  packages; they don't fork or copy pattern code.

Start in `patterns/` to see how a technique is implemented in isolation. Look in `examples/` to
see it used for something.
