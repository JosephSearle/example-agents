#!/usr/bin/env bash
# Runs one agent pattern with a sensible default query (or a caller-supplied override),
# with a colored banner around its output. Shared by `make demo`, `make demo-agent`, and
# `make demo-all` (see Makefile) so the agent/default-query table lives in exactly one place.
#
# Usage: scripts/demo_agent.sh <agent-name> [query ...]
#   - No query args: runs the agent's own default query below.
#   - Query args given: passed straight through to the agent's CLI in place of the default —
#     most agents take one string; map-reduce-agent (multiple topics) and
#     evaluator-optimizer-agent (task + criteria) take more than one, so pass as many words as
#     that agent's own `Usage:` line expects.
set -u

BOLD=$'\033[1m'
CYAN=$'\033[1;36m'
RED=$'\033[1;31m'
RESET=$'\033[0m'

# Every fully-implemented pattern under agents/patterns/ (see README's pattern table), in the
# same order `make demo-all` runs them.
AGENTS=(
  react-agent
  routing-agent
  prompt-chaining-agent
  parallelization-agent
  map-reduce-agent
  orchestrator-workers-agent
  evaluator-optimizer-agent
  supervisor-agent
  swarm-agent
  network-mesh-agent
)

usage() {
  echo "Usage: scripts/demo_agent.sh <agent-name> [query ...]" >&2
}

list_agents() {
  for a in "${AGENTS[@]}"; do echo "  - $a"; done
}

# Sets "$@" to this agent's default invocation args via `set --` directly, rather than returning
# a string to be re-split by the caller — most agents take exactly one query string (which must
# stay as one arg even though it contains spaces), while map-reduce-agent (multiple topics) and
# evaluator-optimizer-agent (task + criteria) take more than one. Splitting on word boundaries
# after the fact (e.g. via unquoted command substitution) would silently break the single-arg
# agents by handing their one query string to the CLI as many separate positional args instead.
set_default_args() {
  case "$1" in
    react-agent)
      set -- "What's 47 * 12, and does that number mean anything in dev slang?" ;;
    routing-agent)
      set -- "My last payment was charged twice, can I get a refund?" ;;
    prompt-chaining-agent)
      set -- "The history of the semicolon in programming languages" ;;
    parallelization-agent)
      set -- "Payment service returning 500s for ~15% of checkout requests since 14:02 UTC." ;;
    map-reduce-agent)
      set -- "cats" "the DMV" "JavaScript" ;;
    orchestrator-workers-agent)
      set -- "Add rate limiting to a public API — figure out what needs to change." ;;
    evaluator-optimizer-agent)
      set -- "Write a haiku about distributed systems" \
        "Must be exactly 5-7-5 syllables and mention consensus" ;;
    supervisor-agent)
      set -- "What's 12 * 8, and how many words are in this sentence?" ;;
    swarm-agent)
      set -- "I'd like a refund for invoice INV-1002." ;;
    network-mesh-agent)
      set -- "Summarize recent trends in vector database indexing." ;;
  esac
  printf '%s\0' "$@"
}

agent="${1:-}"
if [ -z "$agent" ]; then
  usage
  exit 1
fi
shift || true

is_known_agent=false
for a in "${AGENTS[@]}"; do
  if [ "$a" = "$agent" ]; then
    is_known_agent=true
    break
  fi
done

if [ "$is_known_agent" = false ]; then
  echo "${RED}${agent}${RESET} doesn't exist yet, please try one of the following:" >&2
  list_agents >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  # NUL-delimited round trip keeps each default arg intact even though some contain spaces — the
  # same reason `set_default_args` doesn't just `echo` a string for the caller to word-split.
  # `read -d ''` rather than `mapfile` since macOS ships bash 3.2, which lacks it.
  default_args=()
  while IFS= read -r -d '' arg; do
    default_args+=("$arg")
  done < <(set_default_args "$agent")
  set -- "${default_args[@]}"
fi

printf '%s▶ %s%s\n' "$CYAN$BOLD" "$agent" "$RESET"
uv run --package "$agent" "$agent" "$@"
status=$?
if [ "$status" -eq 0 ]; then
  printf '%s✓ %s done%s\n' "$CYAN" "$agent" "$RESET"
else
  printf '%s✗ %s failed (exit %d)%s\n' "$RED" "$agent" "$status" "$RESET" >&2
fi
exit "$status"
