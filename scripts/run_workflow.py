
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.workflow import run
from tools.tools import TOOLS, TOOL_DEFINITIONS
from retrieval.retriever import get_retriever

retriever = get_retriever()
from llm.wrapper import OpenRouterWrapper
from dotenv import load_dotenv
import os
load_dotenv()
model=os.getenv("PLANNER_MODEL", "qwen/qwen3-32b")
tokens=int(os.getenv("PLANNER_MAX_TOKENS", "1024"))
llm = OpenRouterWrapper(model=model, tokens=tokens)

result = run(
    query="Give Priya Nair access to Polaris Reporting.",
    retriever=retriever,
    llm=llm,
    tools=TOOLS,
    tool_descriptions=TOOL_DEFINITIONS,
)

print("Agent result:", result.decision)
print("Tool output:", result.tool_output)
print("Final answer:", result.answer)