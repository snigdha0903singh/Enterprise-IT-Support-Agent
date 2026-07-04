from __future__ import annotations

from typing import Any, Callable

from agent.planner import AgentDecision
from tools.tools import TOOLS


ToolRegistry = dict[str, Callable[..., Any]]


def execute(decision: AgentDecision, tools: ToolRegistry) -> Any:
    """
    Execute exactly what the planner requested.

    This layer intentionally does not reason about user intent, choose fallback
    tools, or repair arguments. Planner mistakes should stay visible.
    """
    if decision.tool is None:
        return None

    if decision.tool not in tools:
        raise KeyError(f"Unknown tool requested by planner: {decision.tool}")

    tool = tools[decision.tool]
    return tool(**decision.arguments)
