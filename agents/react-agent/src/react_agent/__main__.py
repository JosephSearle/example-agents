"""CLI entrypoint: `uv run react-agent "<message>"`."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
import uuid

from agents_common import configure_mlflow, get_checkpointer

from react_agent.graph import EXPERIMENT_NAME, build_agent, extract_response

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

_MIN_ARGC = 2


def main() -> None:
    """Run one turn of the agent against a fresh thread and print the structured response."""
    if len(sys.argv) < _MIN_ARGC:
        print('Usage: react-agent "<message>"', file=sys.stderr)
        raise SystemExit(1)

    message = sys.argv[1]
    configure_mlflow(EXPERIMENT_NAME)

    with get_checkpointer() as checkpointer:
        agent = build_agent(checkpointer=checkpointer)
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
        print(extract_response(result))


if __name__ == "__main__":
    main()
