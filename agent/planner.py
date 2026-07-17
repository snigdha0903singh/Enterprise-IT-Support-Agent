from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from agent.prompts import PLANNER_PROMPT
from llm.wrapper import OpenRouterWrapper
from dotenv import load_dotenv
import os
import re
load_dotenv()
model=os.getenv("PLANNER_MODEL")
tokens=int(os.getenv("PLANNER_MAX_TOKENS"))
# base_url=os.getenv("OPENROUTER_BASE_URL")
# api_key=os.getenv("OPENROUTER_API_KEY")
base_url=os.getenv("LOCAL_MODEL_BASE_URL")
api_key=os.getenv("LOCAL_MODEL_API_KEY")
llm = OpenRouterWrapper(model=model, tokens=tokens, api_key=api_key, base_url=base_url)


class PlannerLLM(Protocol):
    """Minimal interface expected from the LLM used by the planner."""

    def invoke(self, prompt: str) -> Any:
        ...


@dataclass(frozen=True)
class AgentDecision:
    reasoning: str
    tool: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentDecision":
        return cls(
            reasoning=str(data.get("reasoning", "")).strip(),
            tool=data.get("tool"),
            arguments=dict(data.get("arguments") or {}),
            confidence=_clamp_confidence(data.get("confidence", 0.0)),
        )


def plan(
    query: str,
    retrieved_context: str | list[Any],
    available_tools: dict[str, Callable[..., Any]] | list[str] | tuple[str, ...],
) -> AgentDecision:
    """
    Ask the LLM to choose the next tool call.

    This is the only agent layer that should reason about intent, tool choice,
    argument extraction, and confidence.
    """
    prompt = PLANNER_PROMPT.format(
        query=query,
        context=_format_context(retrieved_context),
        tools=_format_tools(available_tools),
    )
    raw_response = _invoke_llm(llm, prompt)
    print(f"model used for planning: {llm.model}")
    return AgentDecision.from_dict(_parse_json_response(raw_response))


def _invoke_llm(llm: PlannerLLM | Callable[[str], Any], prompt: str) -> Any:
    if callable(llm) and not hasattr(llm, "invoke"):
        return llm(prompt)
    return llm.invoke(prompt)


def _parse_json_response(response: Any) -> dict[str, Any]:
    text = getattr(response, "content", response)

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    # Remove markdown code fences
    if text.startswith("```"):
        text = _strip_code_fence(text)

    # Remove // comments
    text = re.sub(r"//.*", "", text)

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    text = text.strip()

    try:
        parsed = json.loads(text)
        tool = parsed.get("tool")
        if isinstance(tool, str):
            tool = tool.strip()
            if tool.lower() in {"null", "none", ""}:
                parsed["tool"] = None
            else:
                parsed["tool"] = tool
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner response was not valid JSON:\n{text}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError("Planner response must be a JSON object.")

    return parsed


def _strip_code_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        # Remove opening fence (``` or ```json)
        lines = lines[1:]

        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines)

    return text.strip()


def _format_context(retrieved_context: str | list[Any]) -> str:
    if isinstance(retrieved_context, str):
        return retrieved_context

    formatted = []
    for index, item in enumerate(retrieved_context, start=1):
        page_content = getattr(item, "page_content", None)
        metadata = getattr(item, "metadata", None)

        if page_content is None:
            formatted.append(f"[{index}] {item}")
            continue

        source = ""
        if metadata:
            source_value = metadata.get("source") or metadata.get("_id")
            if source_value:
                source = f" source={source_value}"
        formatted.append(f"[{index}{source}]\n{page_content}")

    return "\n\n".join(formatted)


def _format_tools(
    available_tools: dict[str, Callable[..., Any]] | list[str] | tuple[str, ...],
) -> str:
    if isinstance(available_tools, dict):
        formatted_tools = []
        for tool_name in sorted(available_tools):
            tool = available_tools[tool_name]
            description = getattr(tool, "description", None)
            arguments = getattr(tool, "arguments", None)

            if description and arguments:
                formatted_arguments = ", ".join(
                    f"{name}: {details}" for name, details in arguments.items()
                )
                formatted_tools.append(
                    f"- {tool_name}: {description} Arguments: {formatted_arguments}"
                )
            elif description:
                formatted_tools.append(f"- {tool_name}: {description}")
            else:
                formatted_tools.append(f"- {tool_name}")

        return "\n".join(formatted_tools)

    return "\n".join(f"- {tool_name}" for tool_name in available_tools)


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    return min(1.0, max(0.0, confidence))
