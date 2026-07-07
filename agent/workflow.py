from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.executor import ToolRegistry, execute
from agent.planner import AgentDecision, PlannerLLM, plan

from llm.wrapper import OpenRouterWrapper
from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("PLANNER_MODEL", "qwen/qwen3-32b")
tokens=int(os.getenv("PLANNER_MAX_TOKENS", "1024"))
llm = OpenRouterWrapper(model=model, tokens=tokens)


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
    llm: PlannerLLM | Callable[[str], Any],
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
        llm=llm,
    )
    tool_output = execute(decision, tools)
    answer = build_answer(decision, tool_output)

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
