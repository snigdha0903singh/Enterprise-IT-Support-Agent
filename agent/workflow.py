from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.executor import ToolRegistry, execute
from agent.planner import AgentDecision, PlannerLLM, plan
import json
from pathlib import Path
CACHE_FILE = Path("cache/agent_vanila_rag_runs.json")
CACHE_FILE.parent.mkdir(exist_ok=True)


@dataclass(frozen=True)
class AgentRun:
    query: str
    context: str | list[Any]
    decision: AgentDecision
    tool_output: Any
    answer: str


def run(
    query: str,
    retriever: Any,
    tools: ToolRegistry,
    tool_descriptions: Any | None = None,
) -> AgentRun:
    """
    Run the planner-agent pipeline:
    retrieve -> plan -> execute -> answer.
    """
    context = retrieve(query, retriever)
    decision = plan(
        query=query,
        retrieved_context=context,
        available_tools=tool_descriptions or tools,
    )
    tool_output = execute(decision, tools)
    answer = build_answer(decision, tool_output)

    # ------------------ Cache Agent Run ------------------

    if CACHE_FILE.exists():
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    cache[query] = {
        "context": (
            context
            if isinstance(context, str)
            else [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in context
            ]
        ),
        "decision": {
            "reasoning": decision.reasoning,
            "tool": decision.tool,
            "arguments": decision.arguments,
            "confidence": decision.confidence,
        },
        "tool_output": tool_output,
        "answer": answer,
    }

    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    # ---------------------------------- Cache Agent Run -------------------

    return AgentRun(
        query=query,
        context=context,
        decision=decision,
        tool_output=tool_output,
        answer=answer,
    )


def retrieve(query: str, retriever: Any) -> str | list[Any]:
    if retriever is None:
        return ""

    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)

    if callable(retriever):
        return retriever(query)

    raise TypeError("Retriever must be callable or expose invoke(query).")


def build_answer(decision: AgentDecision, tool_output: Any) -> str:
    if decision.tool is None:
        return "I do not have enough information to choose an action."

    if tool_output is None:
        return f"{decision.tool} completed."

    return str(tool_output)
